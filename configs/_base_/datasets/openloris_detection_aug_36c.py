# dataset settings
dataset_type = 'CocoDataset'
data_root = '/ws/data/OpenLORIS/'
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

classes = ['person',
           'backpack', 'umbrella', 'handbag', 'tie',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

train1 = dict(
    type=dataset_type,
    # ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_train.json',
    # img_prefix=data_root + 'train/',
    ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
    img_prefix=data_root + 'val/',
    pipeline=train_pipeline1,
    classes=classes,
    ),
train2 = dict(
    type=dataset_type,
    # ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_train.json',
    # img_prefix=data_root + 'train/',
    ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
    img_prefix=data_root + 'val/',
    pipeline=train_pipeline2,
    classes=classes
    ),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ConDataset',
        datasets=[train1, train2]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/mask_rcnn_r50_fpn_2x_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox')



