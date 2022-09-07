# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/ws/data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomAffine',
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
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
        ann_file=data_root +
                     'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root + 'leftImg8bit/train/',
        pipeline=train_pipeline1
        )
train2 = dict(
        type=dataset_type,
        ann_file=data_root +
                     'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root + 'leftImg8bit/train/',
        pipeline=train_pipeline2
        )
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConDataset',
        datasets=[train1, train2]),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),)
evaluation = dict(metric=['bbox', 'segm'])
