from __future__ import annotations
import copy
import re
from .base import ModelBase, gguf
from .qwen import Qwen3_5TextModel

@ModelBase.register('SenseNovaLoopForConditionalGeneration')
class SenseNovaLoopModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.SENSENOVA_LOOP
    no_mtp = True

    def __init__(self, dir_model, *args, **kwargs):
        hp = copy.deepcopy(kwargs.pop('hparams', None) or ModelBase.load_hparams(dir_model, False))
        loop, mhc = hp['loop_settings'], hp['mhc']
        self.loop = loop
        self.mhc = mhc
        self.physical_count = hp['text_config']['num_hidden_layers']
        start, end, count = loop['start'], loop['end'], loop['eval_count']
        if loop['counts'] != [count] or not 0 <= start < end <= self.physical_count:
            raise ValueError('Unsupported loop schedule')
        if mhc['start'] != start or mhc['end'] != end or mhc['streams'] != 2 or mhc['fixed_identity'] or mhc.get('pre_scale', 1.0) != 1.0:
            raise ValueError('Only two-stream mHC covering the repeated block is supported')
        self.plan = list(range(start)) + list(range(start, end)) * count + list(range(end, self.physical_count))
        hp['text_config']['layer_types'] = [hp['text_config']['layer_types'][i] for i in self.plan]
        hp['text_config']['num_hidden_layers'] = len(self.plan)
        hp['text_config']['mtp_num_hidden_layers'] = 0
        super().__init__(dir_model, *args, hparams=hp, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        for name, value in [('loop_start', self.loop['start']), ('loop_end', self.loop['end']),
                            ('loop_count', self.loop['eval_count']), ('physical_layer_count', self.physical_count),
                            ('mhc_streams', 2), ('mhc_iterations', self.mhc['iterations'])]:
            self.gguf_writer.add_uint32('sensenova_loop.' + name, value)

    def modify_tensors(self, data_torch, name, bid):
        match = re.search(r'\.mhc_(attn|ffn)\.(weight|bias)$', name)
        if match:
            assert bid is not None
            kind, suffix = match.groups()
            part = 'fn' if suffix == 'weight' else 'base'
            pairs = [(f'blk.{bid}.hc_{kind}_{part}.{suffix}', data_torch)]
        else:
            pairs = super().modify_tensors(data_torch, name, bid)
        for mapped, tensor in pairs:
            match = re.match(r'blk\.(\d+)\.(.*)', mapped)
            if not match:
                yield mapped, tensor
                continue
            physical = int(match.group(1))
            for virtual, original in enumerate(self.plan):
                if original == physical:
                    yield f'blk.{virtual}.{match.group(2)}', tensor

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if '.hc_attn_' in new_name or '.hc_ffn_' in new_name or new_name.endswith('.ssm_conv1d.weight'):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
