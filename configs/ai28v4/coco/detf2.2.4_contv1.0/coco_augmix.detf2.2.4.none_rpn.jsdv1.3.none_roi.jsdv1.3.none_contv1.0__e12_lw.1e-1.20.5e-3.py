_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ws/external/configs/_base_/datasets/coco_detection.py',
    '/ws/external/configs/_base_/schedules/ai28.py',
    # '/ws/external/configs/_base_/schedules/schedule_1x.py',
    '/ws/external/configs/_base_/default_runtime.py'
]
num_views=2
num_classes=80
#############
### MODEL ###
#############
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
            additional_loss='jsdv1_3_2aug', lambda_weight=0.1, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                       additional_loss="None", lambda_weight=0.0, wandb_name='rpn_bbox')),
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_head=dict(
            num_classes=80,
            type='Shared2FCContrastiveHead',
            with_cont=True,
            cont_predictor_cfg=dict(num_linear=2, feat_channels=256, return_relu=True),
            out_dim_cont=256,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=20, wandb_name='roi_cls', log_pos_ratio=True),
            loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0, wandb_name='roi_bbox'),
            loss_cont=dict(type='ContrastiveLossPlus', version='1.0', loss_weight=0.005, num_views=num_views,
                           memory=0, num_classes=81, dim=256, min_samples=5))),
    train_cfg=dict(
        wandb=dict(log=dict(features_list=[], vars=['log_vars'])),
        analysis_list=[]
    )
)

###############
### DATASET ###
###############
# dataset settings
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix_detection_faster'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMixDetectionFaster', num_views=num_views, version='2.2',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg,
         num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3),
         pre_blur=False, fillmode='var_blur', sigma_ratio=1 / 8,
         mixture_width=1, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2',
                               'gt_bboxes', 'gt_bboxes2', 'gt_labels']),
]
data = dict(
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(pipeline=train_pipeline))

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
optimizer_config = dict(grad_clip=None, detect_anomalous_params=True)

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
loss_type = 'plus'
rpn_loss = 'jsdv1.3.none'
roi_loss = 'jsdv1.3.none.contv1.0'
lambda_weight = '1e-1.20.5e-3'

name = f"{train_version}_{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[dict(type='TextLoggerHook')])

############
### LOAD ###
############
