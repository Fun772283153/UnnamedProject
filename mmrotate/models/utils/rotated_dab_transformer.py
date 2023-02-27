import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence, build_transformer_layer_sequence)
from mmdet.models.utils.transformer import inverse_sigmoid
from .builder import ROTATED_TRANSFORMER
from .misc import get_sine_pos_embed

@ROTATED_TRANSFORMER.register_module()
class RotatedDabDetrTransformer(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None
    ) -> None:
        super().__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, anchor_box_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )

        num_queries = anchor_box_embed.shape[0]
        target = torch.zeros(num_queries, bs, self.embed_dims, device=anchor_box_embed.device)

        hidden_state, reference_boxes = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,
        )
        return hidden_state, reference_boxes


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedDabDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        post_norm_cfg=dict(type='LN'),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            self.post_norm = None

        self.query_scale = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)
    
    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        for layer in self.layers:
            position_sclaes = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_sclaes,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs
            )
        if self.post_norm is not None:
            query = self.post_norm(query)
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedDabDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        modulate_hw_attn: bool = True,
        post_norm_cfg=dict(type='LN'),
        return_intermediate: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.query_sclae = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)
        self.ref_point_head = MLP(int(5 / 2 * self.embed_dims), self.embed_dims, self.embed_dims, 2)

        self.bbox_embed = None

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dims, self.embed_dims, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None
        
        for idx in range(self.num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        anchor_box_embed=None,
        **kwargs,
    ):
        intermediate = []

        reference_boxes = anchor_box_embed.sigmoid()
        intermediate_ref_boxes = [reference_boxes]

        for idx, layer in enumerate(self.layers):
            obj_cneter = reference_boxes[..., : self.embed_dims]
            query_sine_embed = get_sine_pos_embed(obj_cneter)
            query_pos = self.ref_point_head(query_sine_embed)

            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_sclae(query)
            
            query_sine_embed = query_sine_embed[..., : self.embed_dims] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dims // 2:] *= (ref_hw_cond[..., 0] / obj_cneter[..., 2]).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dims // 2] *= (ref_hw_cond[..., 1] / obj_cneter[..., 3]).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )
            if self.bbox_embed is not None:
                offsets = self.bbox_embed(query)
                offsets[..., : self.embed_dims] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., : self.embed_dims].sigmoid()
                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)

        if self.post_norm is not None:
            query = self.post_norm(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_ref_boxes).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_boxes.unsqueeze(0).transpose(1, 2)]
        return query.unsqueeze(0)




class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x