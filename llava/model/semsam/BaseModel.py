import os
import logging

import torch
import torch.nn as nn

from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), save_path)

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location='cpu')
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        if 'model' in state_dict:
            state_dict=state_dict['model']
            state_dict={k[6:]:v for k,v in state_dict.items()}

        # if self.opt['MODEL']['LLAMA'].get('lora_r',0)>0:
        #     new_sd = dict()
        #     for k,v in state_dict.items():
        #         if k.startswith("llama."):
        #             if k.startswith("llama.base_model."):
        #                 new_sd=state_dict
        #                 break
        #             new_sd[k.replace("llama.","llama.base_model.model.")]=v
        #         else:
        #             new_sd[k]=v
        # else:
        #     new_sd = state_dict
        new_sd = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(new_sd, strict=False)
        return self
