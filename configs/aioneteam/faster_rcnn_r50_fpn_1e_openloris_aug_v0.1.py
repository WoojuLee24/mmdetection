model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        wandb=dict(
            layer_list=[], log=dict(features_list=[], vars=['log_vars'])),
        additional_loss=['aug_loss']),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = '/ws/data/OpenLORIS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomAffine2',
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
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'aug_parameters'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train1 = ({
    'type':
    'CocoDataset',
    'ann_file':
    '/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
    'img_prefix':
    '/ws/data/OpenLORIS/val/',
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'with_bbox': True
    }, {
        'type': 'RandomFlip',
        'flip_ratio': 0.0
    }, {
        'type': 'Normalize',
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True
    }, {
        'type': 'Pad',
        'size_divisor': 32
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type': 'Collect',
        'keys': ['img', 'gt_bboxes', 'gt_labels']
    }]
}, )
train2 = ({
    'type':
    'CocoDataset',
    'ann_file':
    '/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
    'img_prefix':
    '/ws/data/OpenLORIS/val/',
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'with_bbox': True
    }, {
        'type': 'RandomAffine2',
        'max_rotate_degree': 0.0,
        'max_translate_ratio': 0.1,
        'scaling_ratio_range': (1.0, 1.5),
        'max_shear_degree': 0.0,
        'border': (0, 0),
        'border_val': (114, 114, 114),
        'min_bbox_size': 2,
        'min_area_ratio': 0.2,
        'max_aspect_ratio': 20,
        'bbox_clip_border': True,
        'skip_filter': True
    }, {
        'type': 'RandomFlip',
        'flip_ratio': 0.0
    }, {
        'type': 'Normalize',
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True
    }, {
        'type': 'Pad',
        'size_divisor': 32
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type': 'Collect',
        'keys': ['img', 'gt_bboxes', 'gt_labels', 'aug_parameters']
    }]
}, )
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ConDataset',
        datasets=[
            ({
                'type':
                'CocoDataset',
                'ann_file':
                '/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
                'img_prefix':
                '/ws/data/OpenLORIS/val/',
                'pipeline': [{
                    'type': 'LoadImageFromFile'
                }, {
                    'type': 'LoadAnnotations',
                    'with_bbox': True
                }, {
                    'type': 'RandomFlip',
                    'flip_ratio': 0.0
                }, {
                    'type': 'Normalize',
                    'mean': [123.675, 116.28, 103.53],
                    'std': [58.395, 57.12, 57.375],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'DefaultFormatBundle'
                }, {
                    'type': 'Collect',
                    'keys': ['img', 'gt_bboxes', 'gt_labels']
                }]
            }, ),
            ({
                'type':
                'CocoDataset',
                'ann_file':
                '/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
                'img_prefix':
                '/ws/data/OpenLORIS/val/',
                'pipeline': [{
                    'type': 'LoadImageFromFile'
                }, {
                    'type': 'LoadAnnotations',
                    'with_bbox': True
                }, {
                    'type': 'RandomAffine2',
                    'max_rotate_degree': 0.0,
                    'max_translate_ratio': 0.1,
                    'scaling_ratio_range': (1.0, 1.5),
                    'max_shear_degree': 0.0,
                    'border': (0, 0),
                    'border_val': (114, 114, 114),
                    'min_bbox_size': 2,
                    'min_area_ratio': 0.2,
                    'max_aspect_ratio': 20,
                    'bbox_clip_border': True,
                    'skip_filter': True
                }, {
                    'type': 'RandomFlip',
                    'flip_ratio': 0.0
                }, {
                    'type': 'Normalize',
                    'mean': [123.675, 116.28, 103.53],
                    'std': [58.395, 57.12, 57.375],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'DefaultFormatBundle'
                }, {
                    'type':
                    'Collect',
                    'keys':
                    ['img', 'gt_bboxes', 'gt_labels', 'aug_parameters']
                }]
            }, )
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
        img_prefix='/ws/data/OpenLORIS/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(848, 480),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json',
        img_prefix='/ws/data/OpenLORIS/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(848, 480),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLogger',
            wandb_init_kwargs=dict(
                project='KT_AI',
                entity='kaist-url-ai28',
                name='faster_rcnn_r50_fpn_1e_openloris_aug_v0.1'),
            log_map_every_iter=False,
            interval=500,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=5)
    ])
custom_hooks = [
    dict(
        type='FeatureHook',
        layer_list=['neck.fpn_convs.0.conv', 'neck.fpn_convs.1.conv', 'neck.fpn_convs.2.conv', 'neck.fpn_convs.3.conv'])
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/ws/data/OpenLORIS/pretrained/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '/ws/data/log/openloris/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1'
auto_resume = False
gpu_ids = range(0, 1)
