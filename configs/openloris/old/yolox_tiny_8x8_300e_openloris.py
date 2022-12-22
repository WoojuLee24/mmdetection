_base_ = './yolox_s_8x8_300e_openloris.py'

# model settings
model = dict(
    random_size_range=(10, 20),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96))

img_scale = (640, 640)

# dataset settings
data_root = '/ws/data/OpenLORIS/'
dataset_type = 'CocoDataset'


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
    ann_file=data_root + 'annotations/yolox_s_8x8_300e_openloris_train.json',
    img_prefix=data_root + 'train/',
    # ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
    # img_prefix=data_root + 'val/',
    pipeline=train_pipeline1,
    ),
train2 = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/yolox_s_8x8_300e_openloris_train.json',
    img_prefix=data_root + 'train/',
    # ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
    # img_prefix=data_root + 'val/',
    pipeline=train_pipeline2,),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ConDataset',
        datasets=[train1, train2]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/yolox_s_8x8_300e_openloris_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/yolox_s_8x8_300e_openloris_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        ))


# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.0001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 5
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
    min_lr_ratio=0.0005)

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
        priority=49),
    
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
                                              'name': "yolox_s_8x8_5e_openloris_augv0.1.1",
                                              },
                           log_map_every_iter=False,
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ])

load_from = '/ws/data/OpenLORIS/pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
