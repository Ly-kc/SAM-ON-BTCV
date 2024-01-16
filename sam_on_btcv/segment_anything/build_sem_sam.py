# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, SemanticMaskDecoder, PromptEncoder, Sam, TwoWayTransformer




def build_sem_sam_vit_h(checkpoint=None, need_semantic = True):
    return _build_sem_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        need_semantic=need_semantic,
    )


build_sem_sam = build_sem_sam_vit_h


def build_sem_sam_vit_l(checkpoint=None, need_semantic = True):
    return _build_sem_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        need_semantic=need_semantic,
    )


def build_sem_sam_vit_b(checkpoint=None, need_semantic = True):
    return _build_sem_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        need_semantic=need_semantic,
    )


sem_sam_model_registry = {
    "default": build_sem_sam_vit_h,
    "vit_h": build_sem_sam_vit_h,
    "vit_l": build_sem_sam_vit_l,
    "vit_b": build_sem_sam_vit_b,
}


def _build_sem_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    need_semantic = True,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=SemanticMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            output_semantic=need_semantic,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        sam = safe_load_weights(sam, checkpoint)
    return sam


def safe_load_weights(module, checkpoints):
    with open(checkpoints, "rb") as f:
        state_dict = torch.load(f)
    module.load_state_dict(state_dict, strict=False)
    #警告不匹配的参数
    for name, param in module.named_parameters():
        if(name not in state_dict):
            print(f'{name} not in state_dict')
    return module