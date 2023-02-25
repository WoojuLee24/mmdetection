_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ws/external/configs/_base_/datasets/coco_detection.py',
    '/ws/external/configs/_base_/schedules/schedule_1x.py',
    '/ws/external/configs/_base_/default_runtime.py'
]

#############
### MODEL ###
#############
model = dict(
    rpn_head=dict(
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        wandb=dict(log=dict(features_list=[], vars=['log_vars'])),
        analysis_list=[]
    )
)

###############
### DATASET ###
###############
# dataset settings
data_root = '/ws/data/coco/'
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix_detection_faster'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMixDetectionFaster', num_views=1, version='2.2',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg,
         num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3),
         pre_blur=True, fillmode='var_blur', sigma_ratio=1 / 8,
         mixture_width=1, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))

################
### RUN TIME ###
################
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(
    type='EpochBasedRunner', max_epochs=12)

###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

train_version = 'v4'
dataset = 'coco'
pipeline = 'augmix.det2.2.4'
loss_type = 'none'
rpn_loss = 'none.none'
roi_loss = 'none.none'
lambda_weight = ''

name = f"{train_version}_{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[dict(type='TextLoggerHook')])

############
### LOAD ###
############
