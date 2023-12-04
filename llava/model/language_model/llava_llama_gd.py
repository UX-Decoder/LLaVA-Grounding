#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
IGNORE_INDEX=-100
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, LlavaMetaForCausalLM_gd,LlavaMetaForCausalLM_gd_interactive

import transformers
# @dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances,tokenizer):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :tokenizer.model_max_length]
        labels = labels[:, :tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

        if 'image_clip' in instances[0]:
            images = [instance['image_clip'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

class LlavaLlamaForCausalLM_gd(LlamaForCausalLM, LlavaMetaForCausalLM_gd):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(self,**batched_inputs):
        # print(kwargs.keys())
        # images_for_llava=torch.stack([inp['image_clip'] for inp in batched_inputs['flickr']])
        collator=DataCollatorForSupervisedDataset()

        if 'refcoco' in batched_inputs:
            if 'vg' in batched_inputs:
                llava_inputs = collator(batched_inputs['vg']+batched_inputs['refcoco'],
                                        tokenizer=batched_inputs['refcoco'][0]['tokenizer'])
            else:
                llava_inputs = collator( batched_inputs['refcoco'],
                                    tokenizer=batched_inputs['refcoco'][0]['tokenizer'])
        elif 'coco' in batched_inputs:
            llava_inputs=collator(batched_inputs['flickr']+batched_inputs['coco'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        else:
            llava_inputs=collator(batched_inputs['flickr'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        llava_inputs['seg_inputs']=batched_inputs
        return self.forward_inner(**llava_inputs)

    def forward_inner(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        seg_inputs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        ground_idx_coco=[]
        ground_idx = [torch.argwhere(lb == 32002)[:, 0] for lb in labels]
        if 'refcoco' in seg_inputs:
            if 'vg' in seg_inputs:
                vg_len=len(seg_inputs['vg'])
                ground_idx_flickr = ground_idx[:vg_len]
                padded_ground_idx_flickr = torch.nn.utils.rnn.pad_sequence(ground_idx_flickr,
                                                                           batch_first=True,
                                                                           padding_value=-1)
                padded_mask_flickr = padded_ground_idx_flickr != -1
                padded_ground_idx_flickr[padded_ground_idx_flickr == -1] = 0
                # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
                hidden_states = outputs[0]
                hidden_states_flickr = hidden_states[:vg_len]
                ground_hs_flickr = torch.gather(hidden_states_flickr, 1,
                                                padded_ground_idx_flickr[..., None].repeat(1, 1,
                                                                                           hidden_states_flickr.shape[
                                                                                               -1]))
                seg_inputs['vg_text_embeddings'] = (ground_hs_flickr, padded_mask_flickr)
            flickr_len = len(seg_inputs['refcoco'])
            ##########flickr
            # if self.seg_model.model.coco_only:
            ground_idx_flickr = ground_idx[vg_len:vg_len+flickr_len] if 'vg' in seg_inputs else ground_idx[:flickr_len]
            padded_ground_idx_flickr = torch.nn.utils.rnn.pad_sequence(ground_idx_flickr,
                                                                       batch_first=True,
                                                                       padding_value=-1)
            padded_mask_flickr = padded_ground_idx_flickr != -1
            padded_ground_idx_flickr[padded_ground_idx_flickr == -1] = 0
            # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
            hidden_states = outputs[0]
            hidden_states_flickr = hidden_states[vg_len:vg_len+flickr_len] if 'vg' in seg_inputs else hidden_states[:flickr_len]
            ground_hs_flickr = torch.gather(hidden_states_flickr, 1, padded_ground_idx_flickr[..., None].repeat(1, 1,
                                                                                                                hidden_states_flickr.shape[
                                                                                                                    -1]))
            seg_inputs['refcoco_text_embeddings'] = (ground_hs_flickr, padded_mask_flickr)
            # seg_inputs['flickr']=seg_inputs['refcoco']
        else:
            flickr_len=len(seg_inputs['flickr'])
            ground_idx = [torch.argwhere(lb == 32002)[:, 0] for lb in labels]
            zero_mask = [0 if len(idx) == 0 else 1 for idx in ground_idx]
            ##########flickr
            # if self.seg_model.model.coco_only:
            ground_idx_flickr=ground_idx[:flickr_len]
            padded_ground_idx_flickr = torch.nn.utils.rnn.pad_sequence(ground_idx_flickr,
                                                     batch_first=True,
                                                     padding_value=-1)
            padded_mask_flickr=padded_ground_idx_flickr!=-1
            padded_ground_idx_flickr[padded_ground_idx_flickr==-1]=0
            # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
            hidden_states = outputs[0]
            hidden_states_flickr=hidden_states[:flickr_len]
            ground_hs_flickr=torch.gather(hidden_states_flickr,1,padded_ground_idx_flickr[...,None].repeat(1,1,hidden_states_flickr.shape[-1]))
            seg_inputs['flickr_text_embeddings']=(ground_hs_flickr,padded_mask_flickr)

            ##########coco
            ground_idx_coco = ground_idx[flickr_len:]
            if len(ground_idx_coco)>0:
                for i,(idx,data) in enumerate(zip(ground_idx_coco,seg_inputs['coco'])):
                    mask=data['grounding_mask']
                    ground_idx_coco[i]=idx[mask[:len(idx)]]
                padded_ground_idx_coco = torch.nn.utils.rnn.pad_sequence(ground_idx_coco,
                                                                           batch_first=True,
                                                                           padding_value=-1)
                padded_mask_coco = padded_ground_idx_coco != -1

                padded_ground_idx_coco[padded_ground_idx_coco == -1] = 0

                hidden_states = outputs[0]
                hidden_states_coco = hidden_states[flickr_len:]
                ground_hs_coco = torch.gather(hidden_states_coco, 1, padded_ground_idx_coco[..., None].repeat(1, 1,
                                                                                                                    hidden_states_coco.shape[
                                                                                                                        -1]))
                seg_inputs['coco_text_embeddings'] = (ground_hs_coco, padded_mask_coco)

        ground_loss=self.seg_model(seg_inputs)
        if self.seg_model.model.coco_only and len(ground_idx_coco)>0:
            logits = self.lm_head(hidden_states_coco)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            if self.seg_model.model.coco_only and len(ground_idx_coco) > 0:
                shift_labels = labels[..., 1:][flickr_len:].contiguous()
            else:
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        ground_loss['llava']=loss
        ground_loss['loss_total']=sum(ground_loss.values())
        return CausalLMOutputWithPast(
            loss=ground_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

class LlavaLlamaForCausalLM_joint(LlavaLlamaForCausalLM_gd):
    def forward(self,**batched_inputs):
        # print(kwargs.keys())
        # images_for_llava=torch.stack([inp['image_clip'] for inp in batched_inputs['flickr']])
        collator=DataCollatorForSupervisedDataset()
        assert 'refcoco' in batched_inputs and 'flickr' in batched_inputs and 'llava' in batched_inputs
        for data in batched_inputs['llava']:
            data['image_clip']=data['image']
        llava_inputs = collator( batched_inputs['flickr']+batched_inputs['refcoco']+batched_inputs['llava'],
                                tokenizer=batched_inputs['refcoco'][0]['tokenizer'])

        # if 'refcoco' in batched_inputs:
        #     llava_inputs = collator( batched_inputs['refcoco'],
        #                             tokenizer=batched_inputs['refcoco'][0]['tokenizer'])
        # elif 'coco' in batched_inputs:
        #     llava_inputs=collator(batched_inputs['flickr']+batched_inputs['coco'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        # else:
        #     llava_inputs=collator(batched_inputs['flickr'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        llava_inputs['seg_inputs']=batched_inputs
        return self.forward_inner(**llava_inputs)

    def forward_inner(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        seg_inputs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        ground_idx_coco=[]
        # if 'refcoco' in seg_inputs:
        flickr_len = len(seg_inputs['flickr'])
        ground_idx = [torch.argwhere(lb == 32002)[:, 0] for lb in labels]
        ##########flickr
        # if self.seg_model.model.coco_only:
        ground_idx_flickr = ground_idx[:flickr_len]
        padded_ground_idx_flickr = torch.nn.utils.rnn.pad_sequence(ground_idx_flickr,
                                                                   batch_first=True,
                                                                   padding_value=-1)
        padded_mask_flickr = padded_ground_idx_flickr != -1
        padded_ground_idx_flickr[padded_ground_idx_flickr == -1] = 0
        # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
        hidden_states = outputs[0]
        hidden_states_flickr = hidden_states[:flickr_len]
        ground_hs_flickr = torch.gather(hidden_states_flickr, 1, padded_ground_idx_flickr[..., None].repeat(1, 1,
                                                                                                            hidden_states_flickr.shape[
                                                                                                                -1]))
        seg_inputs['flickr_text_embeddings'] = (ground_hs_flickr, padded_mask_flickr)
        # seg_inputs['flickr']=seg_inputs['refcoco']
        # else:
        #################################################
        #################################################
        refcoco_len=len(seg_inputs['refcoco'])
        ground_idx = [torch.argwhere(lb == 32002)[:, 0] for lb in labels]
        ##########flickr
        ground_idx_refcoco=ground_idx[flickr_len:flickr_len+refcoco_len]
        padded_ground_idx_refcoco = torch.nn.utils.rnn.pad_sequence(ground_idx_refcoco,
                                                 batch_first=True,
                                                 padding_value=-1)
        padded_mask_refcoco=padded_ground_idx_refcoco!=-1
        padded_ground_idx_refcoco[padded_ground_idx_refcoco==-1]=0
        # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
        # hidden_states = outputs[0]
        hidden_states_refcoco=hidden_states[flickr_len:flickr_len+refcoco_len]
        ground_hs_refcoco=torch.gather(hidden_states_refcoco,1,padded_ground_idx_refcoco[...,None].repeat(1,1,hidden_states_refcoco.shape[-1]))
        seg_inputs['refcoco_text_embeddings']=(ground_hs_refcoco,padded_mask_refcoco)



        ground_loss=self.seg_model(seg_inputs)
        # if self.seg_model.model.coco_only and len(ground_idx_coco)>0:
        #     logits = self.lm_head(hidden_states_coco)
        # else:
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            if self.seg_model.model.coco_only and len(ground_idx_coco) > 0:
                shift_labels = labels[..., 1:][flickr_len:].contiguous()
            else:
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        ground_loss['llava']=loss
        ground_loss['loss_total']=sum(ground_loss.values())
        return CausalLMOutputWithPast(
            loss=ground_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlavaLlamaForCausalLM_joint_2st(LlavaLlamaForCausalLM_gd):
    def forward(self,**batched_inputs):
        # print(kwargs.keys())
        # images_for_llava=torch.stack([inp['image_clip'] for inp in batched_inputs['flickr']])
        collator=DataCollatorForSupervisedDataset()
        assert 'coco' in batched_inputs and 'flickr' in batched_inputs and 'llava' in batched_inputs
        for data in batched_inputs['llava']:
            data['image_clip']=data['image']
        llava_inputs = collator( batched_inputs['flickr']+batched_inputs['coco']+batched_inputs['llava'],
                                tokenizer=batched_inputs['coco'][0]['tokenizer'])

        # if 'refcoco' in batched_inputs:
        #     llava_inputs = collator( batched_inputs['refcoco'],
        #                             tokenizer=batched_inputs['refcoco'][0]['tokenizer'])
        # elif 'coco' in batched_inputs:
        #     llava_inputs=collator(batched_inputs['flickr']+batched_inputs['coco'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        # else:
        #     llava_inputs=collator(batched_inputs['flickr'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        llava_inputs['seg_inputs']=batched_inputs
        return self.forward_inner(**llava_inputs)

    def forward_inner(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        seg_inputs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        flickr_len = len(seg_inputs['flickr'])
        ground_idx = [torch.argwhere(lb == 32002)[:, 0] for lb in labels]
        ##########flickr
        # if self.seg_model.model.coco_only:
        ground_idx_flickr = ground_idx[:flickr_len]
        padded_ground_idx_flickr = torch.nn.utils.rnn.pad_sequence(ground_idx_flickr,
                                                                   batch_first=True,
                                                                   padding_value=-1)
        padded_mask_flickr = padded_ground_idx_flickr != -1
        padded_ground_idx_flickr[padded_ground_idx_flickr == -1] = 0
        # ground_idx=[[-1] if len(idx)==0 else idx for idx in ground_idx]
        if self.seg_model.model.detach_seg:
            hidden_states = outputs[0].detach()
        else:
            hidden_states = outputs[0]
        hidden_states_flickr = hidden_states[:flickr_len]
        ground_hs_flickr = torch.gather(hidden_states_flickr, 1, padded_ground_idx_flickr[..., None].repeat(1, 1,
                                                                                                            hidden_states_flickr.shape[
                                                                                                                -1]))
        seg_inputs['flickr_text_embeddings'] = (ground_hs_flickr, padded_mask_flickr)

        ##########coco
        coco_len = len(seg_inputs['coco'])
        ground_idx_coco = ground_idx[flickr_len:flickr_len+coco_len]
        if len(ground_idx_coco) > 0:
            for i, (idx, data) in enumerate(zip(ground_idx_coco, seg_inputs['coco'])):
                mask = data['grounding_mask']
                ground_idx_coco[i] = idx[mask[:len(idx)]]
            padded_ground_idx_coco = torch.nn.utils.rnn.pad_sequence(ground_idx_coco,
                                                                     batch_first=True,
                                                                     padding_value=-1)
            padded_mask_coco = padded_ground_idx_coco != -1

            padded_ground_idx_coco[padded_ground_idx_coco == -1] = 0

            # hidden_states = outputs[0]
            hidden_states_coco = hidden_states[flickr_len:flickr_len+coco_len]
            ground_hs_coco = torch.gather(hidden_states_coco, 1, padded_ground_idx_coco[..., None].repeat(1, 1,
                                                                                                          hidden_states_coco.shape[
                                                                                                              -1]))
            seg_inputs['coco_text_embeddings'] = (ground_hs_coco, padded_mask_coco)
        ground_loss = self.seg_model(seg_inputs)
        hidden_states_ = outputs[0]
        if self.seg_model.model.coco_only and len(ground_idx_coco) > 0:
            logits = self.lm_head(hidden_states_[flickr_len:])
        else:
            logits = self.lm_head(hidden_states_)
        ############################################################

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            if self.seg_model.model.coco_only and len(ground_idx_coco) > 0:
                shift_labels = labels[..., 1:][flickr_len:].contiguous()
            else:
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        ground_loss['llava']=loss
        ground_loss['loss_total']=sum(ground_loss.values())
        ignore_list=[f'_{i}' for i in range(1,10)]
        ignore_list.append('interm')
        for key in list(ground_loss.keys()):
            if not key.endswith('_0') and key!='llava' and key !='loss_total':
                ground_loss.pop(key)
        return CausalLMOutputWithPast(
            loss=ground_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr(LlamaForCausalLM, LlavaMetaForCausalLM_gd_interactive):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(self,**batched_inputs):
        # print(kwargs.keys())
        # images_for_llava=torch.stack([inp['image_clip'] for inp in batched_inputs['flickr']])
        collator=DataCollatorForSupervisedDataset()
        # assert 'coco' in batched_inputs and 'flickr' in batched_inputs and 'llava' in batched_inputs and 'interactive' in batched_inputs
        # for data in batched_inputs['llava']:
        #     data['image_clip']=data['image']
        llava_inputs = collator( batched_inputs['interactive'],
                                tokenizer=batched_inputs['interactive'][0]['tokenizer'])

        # if 'refcoco' in batched_inputs:
        #     llava_inputs = collator( batched_inputs['refcoco'],
        #                             tokenizer=batched_inputs['refcoco'][0]['tokenizer'])
        # elif 'coco' in batched_inputs:
        #     llava_inputs=collator(batched_inputs['flickr']+batched_inputs['coco'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        # else:
        #     llava_inputs=collator(batched_inputs['flickr'],tokenizer=batched_inputs['flickr'][0]['tokenizer'])
        llava_inputs['seg_inputs']=batched_inputs
        res1= self.forward_inner(**llava_inputs)
        loss_dict1=res1.loss
        prefix1='coco.'
        res1.loss=res1['loss']={prefix1+k:v for k,v in loss_dict1.items()}
        if 'interactiveref' in batched_inputs:
            llava_inputs = collator( batched_inputs['interactiveref'],
                                    tokenizer=batched_inputs['interactive'][0]['tokenizer'])
            batched_inputs['interactive']=batched_inputs['interactiveref']
            llava_inputs['seg_inputs']=batched_inputs
            res2= self.forward_inner(**llava_inputs)
            loss_dict2=res2.loss
            prefix2='refcoco.'
            res2.loss=res2['loss']={prefix2+k:v for k,v in loss_dict2.items()}
            res1.loss.update(res2.loss)
            res1.loss['loss_total']=res1.loss['coco.loss_total']+res1.loss['refcoco.loss_total']
        else:
            res1.loss['loss_total'] = res1.loss['coco.loss_total']
        return res1


    def forward_inner(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        seg_inputs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        obj_feats,inter_losses=self.interactive_model.float().forward(seg_inputs['interactive'],detach=False)
        obj_feats=[obj_feats[i][seg_inputs['interactive'][i]['grounding_index']][None] for i in range(len(obj_feats))]
        num_it=len(seg_inputs['interactive'])
        _, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,obj_feats=obj_feats,num_it=num_it)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

  
        hidden_states_ = outputs[0]

        logits = self.lm_head(hidden_states_)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            # if self.seg_model.model.coco_only and len(ground_idx_coco) > 0:
            #     shift_labels = labels[..., 1:][flickr_len:].contiguous()
            # else:
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        ground_loss=dict()
        ground_loss['llava']=loss
        # for k,v in inter_losses.items():
        #     print(v.dtype)
        inter_losses={k:inter_losses[k].to(float) for k in inter_losses.keys()}
        ground_loss.update(inter_losses)
        # import pdb;pdb.set_trace()
        ground_loss['loss_total']=sum(ground_loss.values())
        ignore_list=[f'_{i}' for i in range(1,10)]
        ignore_list.append('interm')
        for key in list(ground_loss.keys()):
            if not key.endswith('_0') and key!='llava' and key !='loss_total':
                ground_loss.pop(key)
        return CausalLMOutputWithPast(
            loss=ground_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_gd)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_joint)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_joint_2st)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr)