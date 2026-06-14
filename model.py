
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

import numpy as np
import bitsandbytes as bnb

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "output_embeds", "logits", "past_key_values"],
)
MAX_N_LATENT = 8


class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        device
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_id = eos_token_id
        self.kv_cache = None
        self.device = device

        if isinstance(base_causallm, GPT2LMHeadModel):
            self._base_transformer_attr = "transformer"
        else:
            self._base_transformer_attr = "model"

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            model_embedding = self.base_causallm.get_input_embeddings()
            self.embedding = bnb.nn.StableEmbedding(
                model_embedding.num_embeddings,
                model_embedding.embedding_dim,
                padding_idx=model_embedding.padding_idx,
            )
            self.embedding.weight.data.copy_(model_embedding.weight.data)
            self.embedding.norm = nn.Identity()
            self.base_causallm.set_input_embeddings(self.embedding)
            self.base_causallm.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        reset_kv_cache=False,
    ):
        # KV cache is not implemented for this demo
        if reset_kv_cache:
            self.kv_cache = None

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(self.device)

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.to(self.device)

        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            labels = labels.to(self.device)

        model_dtype = next(self.base_causallm.parameters()).dtype
        input_embeds = self.embedding(input_ids).to(dtype=model_dtype)

        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists = [
            [
                idx[1].item()
                for idx in latent_indices
                if idx[0].item() == batch_idx
            ]
            for batch_idx in range(input_ids.shape[0])
        ]
        max_n_latents = max((len(latents) for latents in latent_lists), default=0)

        for pass_idx in range(max_n_latents):
            filling_indices = [
                (batch_idx, latents[pass_idx])
                for batch_idx, latents in enumerate(latent_lists)
                if len(latents) > pass_idx
            ]
            if not filling_indices:
                continue

            latent_pos_set = {latent_pos for _, latent_pos in filling_indices}
            if len(latent_pos_set) != 1:
                raise ValueError(
                    "Latent positions are not aligned across the batch. "
                    "Use LatentCollator(start_id=...) to left-pad examples so "
                    "<bot> and latent tokens share the same indices."
                )
            latent_pos = filling_indices[0][1]

            outputs = self.base_causallm(
                inputs_embeds=input_embeds[:, :latent_pos, :],
                attention_mask=attention_mask[:, :latent_pos],
                position_ids=position_ids[:, :latent_pos],
                output_hidden_states=True,
            )
            previous_hidden = outputs.hidden_states[-1][:, -1, :]

            updated_embeds = input_embeds.clone()
            for batch_idx, current_latent_pos in filling_indices:
                updated_embeds[batch_idx, current_latent_pos, :] = previous_hidden[
                    batch_idx
                ]
            input_embeds = updated_embeds

        outputs = self.base_causallm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        loss = None
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return Outputs(
            loss=loss,
            inputs_embeds=input_embeds,
            output_embeds=outputs.hidden_states[-1],
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

        # # 1. Find <bot> and <eot>
        # idx_bot = torch.where(input_ids == self.start_latent_id)[0].item()
        # idx_eot = torch.where(input_ids == self.end_latent_id)[0].item()

        # # 2. Compute everything before <bot> in one go
        # init_embeds = self.embedding(input_ids[:idx_bot+1]).unsqueeze(0)

        # # 3. From <bot>, compute one step at a time for each <latent>
        # embeds = init_embeds
        # for i in range(idx_bot, idx_eot):
        #     outputs  = self.base_causallm(inputs_embeds=embeds,
        #                                 attention_mask=attention_mask[:i],
        #                                 position_ids=position_ids[:i],
        #                                 output_hidden_states=True)
            
        #     next_embeds = outputs.hidden_states[-1][:, -1, :]       # hidden state for next predicted latent token
        #     embeds = torch.concat([embeds, next_embeds])
        
        # # 4. Run a final pass including <eot>
        # outputs = self.base_causallm(inputs_embeds=embeds,
        #                                 attention_mask=attention_mask[:idx_eot+1],
        #                                 position_ids=position_ids[:idx_eot+1],
        #                                 output_hidden_states=True)

        # return outputs.hidden_states[-1]
    
    def generate(self, input_ids, max_new_tokens=32):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        assert input_ids.shape[0] == 1, "generate currently supports batch size 1"

        # First pass to obtain embeddings until the <eot> token
        outputs = self.forward(input_ids)
        output_embeds = outputs.inputs_embeds

        # Calculate next token and its corresponding input embedding
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        ans_tokens = [next_token]
        if next_token != self.eos_token_id:
            new_embed = self.embedding(
                torch.tensor(next_token, device=self.device)
            ).to(dtype=output_embeds.dtype).view(1, 1, -1)
            output_embeds = torch.cat([output_embeds, new_embed], dim=1)

        # Repeat until <eos> or max_new_tokens
        for i in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                break
            outputs = self.base_causallm(inputs_embeds=output_embeds)
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            ans_tokens.append(next_token)
            if next_token == self.eos_token_id:
                break
            new_embed = self.embedding(
                torch.tensor(next_token, device=self.device)
            ).to(dtype=output_embeds.dtype).view(1, 1, -1)
            output_embeds = torch.cat([output_embeds, new_embed], dim=1)

        return ans_tokens
