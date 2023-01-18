_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ws/external/configs/_base_/datasets/coco_detection.py',
    '/ws/external/configs/_base_/schedules/schedule_1x.py',
    '/ws/external/configs/_base_/default_runtime.py'
]
#############
### MODEL ###
#############
num_views = 2
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
            additional_loss='jsdv1_3_2aug', lambda_weight=0.1, wandb_name='rpn_cls',
            additional_loss2=None, lambda_weight2=0),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                       additional_loss="None", lambda_weight=0, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=10, wandb_name='roi_cls', log_pos_ratio=False,
                additional_loss2='None', lambda_weight2=0),
            loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0, wandb_name='roi_bbox'))),
    train_cfg=dict(
        wandb=dict(log=dict(features_list=[], vars=['log_vars'])),
        analysis_list=[
            dict(type='loss_weight', outputs=dict()),
            dict(type='bbox_head_loss',
                 log_list=['acc_pos', 'acc_neg', 'acc_orig', 'acc_aug2', 'acc_aug3', 'consistency']),
        ]
    )
)
###############
### DATASET ###
###############
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMix', no_jsd=False, num_views=num_views,
         aug_list='augmentations', aug_severity=3, **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
    train=dict(dataset=dict(pipeline=train_pipeline)),
)

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
pipeline = 'augmix.augs'
loss_type = 'plus'
rpn_loss = 'jsdv1.3.none'
roi_loss = 'jsdv1.3.none'
lambda_weight = '1e-1.10'

name = f"{train_version}_{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28v4", 'entity': "kaist-url-ai28",
                                              'name': name,
                                              'config': {
                                                  # data pipeline
                                                  'data pipeline': f"{pipeline}",
                                                  # losses
                                                  'loss type(rpn)': f"{rpn_loss}",
                                                  'loss type(roi)': f"{roi_loss}",
                                                  # parameters
                                                  'epoch': runner['max_epochs'],
                                                  'lambda_weight': lambda_weight,
                                              }},
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )

############
### LOAD ###
############
