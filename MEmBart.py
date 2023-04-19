import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FiDmBart(MBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, **kwargs):
        self.model.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            **kwargs
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap mBart encoder to obtain a Fusion-in-Decoder model.
        """
        self.model.encoder = EncoderWrapper(self.model.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.model.encoder = self.model.encoder.encoder
        layers = []
        for mod in self.model.encoder.layers:
            layers.append(mod.module)
        layers = nn.ModuleList(layers)
        self.model.encoder.layers = layers

    def load_mbart(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.model.encoder.encoder.layers:
            mod.use_checkpoint = use_checkpoint

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for mBart Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape # (B, N*L)
        passage_length = total_length // self.n_passages # N
        input_ids = input_ids.view(bsz*self.n_passages, passage_length) # (B*N, L)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length) # (B*N, L)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:] # (B, N*L, D)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages*passage_length, -1) # (B, N*L, D)
        return outputs

def apply_checkpoint_wrapper(mbart_encoder, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    layers = []
    for mod in mbart_encoder.layers:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        layers.append(wrapped_mod)
    layers = nn.ModuleList(layers)
    mbart_encoder.layers = layers


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, layer_head_mask, output_attentions, **kwargs)
        return output


if __name__ == '__main__':
    mBart = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    model = FiDmBart(mBart.config)
    model.load_mbart(mBart.state_dict())
    print("")
