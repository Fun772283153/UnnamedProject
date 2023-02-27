angle_version = 'oc'
_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='RotatedDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    bbox_head=dict(
        type='RotatedDabDetrHead',
        num_query=300,
        num_classes=15,
        in_channels=2048,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        angle_version=angle_version,
        random_refpoints_xy=False,
        transformer=dict(
            type='RotatedDabDetrTransformer',
            encoder=dict(
                type='RotatedDabDetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='DabDetrMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            batch_first=False,
                        )
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU')
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                ),
            ),
            decoder=dict(
                type='RotatedDabDetrTransformerDecoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='ConditionalSelfAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.,
                            batch_first=False,
                        ),
                        dict(
                            type='ConditionalCrossAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.,
                            batch_first=False,
                        )
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU')
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            temperature=20,
            num_feats=128,
            normalize=True,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            gamma=2.0,
            alpha=0.25
        ),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0,
        ),
        loss_iou=dict(
            type='RotatedIoULoss',
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='RotatedHuangarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            reg_cost=dict(type='RBBoxL1Cost', weight=1.0, box_format='xywha'),
            iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=1.0)
        ),
    ),
    test_cfg=dict(),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, filter_empty_gt=True, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        # custom_keys={
        #     'backbone': dict(lr_mult=0.1, decay_mylt=1.0),
        #     'bbox_head': dict(lr_mult=1., decay_mult=1.0)}
    )
            # '.bbox_head': dict(lr_mult=1.0, decay_mult=1.0),})
    #         'module.bbox_head.transformer.encoder': dict(lr_mult=1.0, decay_mult=1.0),
    #         'module.bbox_head.transformer.decoder': dict(lr_mult=1.0, decay_mult=1.0),})
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(warmup=None, policy='step', step=[380])
runner = dict(type='EpochBasedRunner', max_epochs=400)
find_unused_parameters = True