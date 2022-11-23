_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (640, 640)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5),
                   additional_loss=['aug_loss'],  # ['frame_loss'], ['aug_loss']       # additional_loss parameter
                   lambda_weight=[0.5, 0.5, 0.5], # [0.5, 0.5, 0.5, 0.5],
                   wandb=dict(
                       log=dict(
                           features_list=[  # 'neck.fpn_convs.0.conv',
                           ],
                           vars=['log_vars']
                       ))
                   ),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = '/ws/data/coco/'
dataset_type = 'CocoDataset'

# train_pipeline = [
#     dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     # According to the official implementation, multi-scale
#     # training is not considered here but in the
#     # 'mmdet/models/detectors/yolox.py'.
#     dict(type='Resize', img_scale=img_scale, keep_ratio=True),
#     dict(
#         type='Pad',
#         pad_to_square=True,
#         # If the image is three-channel, the pad value needs
#         # to be set separately for each channel.
#         pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
#
# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.json',
#         img_prefix=data_root + 'train2017/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=False,
#     ),
#     pipeline=train_pipeline)
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Pad',
#                 pad_to_square=True,
#                 pad_val=dict(img=(114.0, 114.0, 114.0))),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
#
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4,
#     persistent_workers=True,
#     train=train_dataset,
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomAffine2',
             max_rotate_degree=0.0,
             max_translate_ratio=0.1,
             scaling_ratio_range=(1.0, 1.5),
             max_shear_degree=0.0,
             border=(0, 0),
             border_val=(114, 114, 114),
             min_bbox_size=2,
             min_area_ratio=0.2,
             max_aspect_ratio=20,
             bbox_clip_border=True,
             skip_filter=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'aug_parameters']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(848, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
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
    workers_per_gpu=1,
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
        pipeline=test_pipeline,
        ))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 10
num_last_epochs = 1
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.005)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=['neck.out_convs.0.activate',
                     'neck.out_convs.1.activate',
                     'neck.out_convs.2.activate',
             ]),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=50,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "KT_AI", 'entity': "kaist-url-ai28",
                                              'name': "yolox_s_8x8_10e_coco_augv0.1.1",
                                              },
                           log_map_every_iter=False,
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )

load_from = '/ws/data/OpenLORIS/pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'