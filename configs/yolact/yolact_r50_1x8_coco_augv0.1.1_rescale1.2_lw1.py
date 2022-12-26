_base_ = '../_base_/default_runtime.py'

# model settings
img_size = 550
model = dict(
    type='YOLACT',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # do not freeze stem
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,  # update the statistics of bn
        zero_init_residual=False,
        style='pytorch',
        init_cfg=None),   # dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[550.0 / x for x in [69, 35, 18, 9, 5]],
            centers=[(550 * 0.5 / x, 550 * 0.5 / x)
                     for x in [69, 35, 18, 9, 5]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=256,
        num_protos=32,
        num_classes=80,
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=80,
        in_channels=256,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        # smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
        additional_loss=['aug_loss'],
        lambda_weight=[1, 1, 1],
        wandb=dict(
            log=dict(
                features_list=[  # 'neck.fpn_convs.0.conv',
                ],
                vars=['log_vars']
            ))
        ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        top_k=200,
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/ws/data/coco/'
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38], to_rgb=True)

train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='RandomAffine2',
             max_rotate_degree=0.0,
             max_translate_ratio=0.1,
             scaling_ratio_range=(1.0, 1.2),
             max_shear_degree=0.0,
             border=(0, 0),
             border_val=(114, 114, 114),
             min_bbox_size=2,
             min_area_ratio=0.2,
             max_aspect_ratio=20,
             bbox_clip_border=True,
             skip_filter=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'aug_parameters']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train1 = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',
    pipeline=train_pipeline1,
    ),
train2 = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/instances_train2017.json',
    img_prefix=data_root + 'train2017/',
    pipeline=train_pipeline2,),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConDataset',
        datasets=[train1, train2]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.1,
    step=[2, 4])
runner = dict(type='EpochBasedRunner', max_epochs=5)

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=[
                    'neck.fpn_convs.0',
                    'neck.fpn_convs.1',
                    'neck.fpn_convs.2',
             ]),]

cudnn_benchmark = True
evaluation = dict(metric=['bbox', 'segm'])

log_config = dict(interval=50,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "KT_AI", 'entity': "kaist-url-ai28",
                                              'name': "yolact_r50_1x8_coco_rescale1.2_lw1",
                                              },
                           log_map_every_iter=False,
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )

load_from = '/ws/data/log_kt/pretrained/yolact_r50_1x8_coco_20200908-f38d58df.pth'