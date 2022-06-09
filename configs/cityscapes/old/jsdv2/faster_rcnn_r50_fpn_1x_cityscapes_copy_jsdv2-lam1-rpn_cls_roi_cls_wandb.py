_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_ai28.py',
    '../../_base_/datasets/cityscapes_detection_augmix_without_obj_translation.py',
    '../../_base_/schedules/ai28.py',
    '../../_base_/default_runtime.py'
]

'''
[OPTIONS]
  model
  * loss_cls/loss_bbox.additional_loss
    : [None, 'jsd', 'jsdy', 'jsdsy']
  * train_cfg.wandb.log.features_list 
    : [None, "rpn_head.rpn_cls", "neck.fpn_convs.0.conv", "neck.fpn_convs.1.conv", "neck.fpn_convs.2.conv", "neck.fpn_convs.3.conv"] 
'''

model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0,
            additional_loss='jsdv2', additional_loss_weight_reduce=False,
            lambda_weight=1, temper=1, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0,
                       additional_loss='None', lambda_weight=1, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0,
                additional_loss='jsdv2', additional_loss_weight_reduce=False,
                lambda_weight=1, temper=1, wandb_name='roi_cls'),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0,
                           additional_loss='None', lambda_weight=1, wandb_name='roi_bbox'))),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars'],
                ))))

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

rpn_loss_cls = model['rpn_head']['loss_cls']
rpn_loss_bbox = model['rpn_head']['loss_bbox']
roi_loss_cls = model['roi_head']['bbox_head']['loss_cls']
roi_loss_bbox = model['roi_head']['bbox_head']['loss_bbox']

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/ws/data/cityscapes/'
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ###Insert AugMix###
    dict(type='AugMix', no_jsd=False, aug_list='copy', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'img3', 'gt_bboxes', 'gt_labels']),
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
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
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
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "ai28",
                                              'name': f"augmix_copy_lambda{rpn_loss_cls['lambda_weight']}_rpn.{rpn_loss_cls['additional_loss']}.{rpn_loss_bbox['additional_loss']}_"
                                                      f"roi.{roi_loss_cls['additional_loss']}.{roi_loss_bbox['additional_loss']}",
                                              'config': {
                                                  'loss_type(rpn_cls)': f"{rpn_loss_cls['type']}({rpn_loss_cls['additional_loss']})",
                                                  'loss_type(rpn_bbox)': f"{rpn_loss_bbox['type']}({rpn_loss_bbox['additional_loss']})",
                                                  'loss_type(roi_cls)': f"{roi_loss_cls['type']}({roi_loss_cls['additional_loss']})",
                                                  'loss_type(roi_bbox)': f"{roi_loss_bbox['type']}({roi_loss_bbox['additional_loss']})",
                                                  'aug_list': f"augmix_copy"
                                              }},
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
