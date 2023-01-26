_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn_ai28.py',
    '/ws/external/configs/_base_/datasets/cityscapes_detection.py',
    '/ws/external/configs/_base_/schedules/ai28.py',
    '/ws/external/configs/_base_/default_runtime.py'
]

'''
[NAMING] (updated 22.06.09)
  data pipeline:    [original, augmix(copy, wotrans)]
  loss:             [none, augmix, plus]
  rpn:              [rpn(none.none, jsd.none, ...)]
  roi:              [roi(none.none, jsd.none, ...)]
  parameters:       [e1, lw(e.g.,1e-4), wr(true, false)]
                     > e1: 1 epoch
                     > lw: lambda weight
                     > wr: weight reduce  
[OPTIONS] (updated 22.06.09)
  model
  * loss_cls/loss_bbox.additional_loss
    : [None, 'jsd', 'jsdy', 'jsdv1_1', 'jsdv2']
  * train_cfg.wandb.log.features_list 
    : [None, "rpn_head.rpn_cls", "neck.fpn_convs.0.conv", "neck.fpn_convs.1.conv", "neck.fpn_convs.2.conv", "neck.fpn_convs.3.conv"]
  * train_cfg.wandb.log.vars
    : ['log_vars'] 
'''

#############
### MODEL ###
#############
num_views = 2
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
            additional_loss='jsdv1_3_2aug', lambda_weight=0.1, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                       additional_loss="None", lambda_weight=0.0001, wandb_name='rpn_bbox')),
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_head=dict(
            type='Shared2FCContrastiveHead',
            with_cont=True,
            cont_predictor_cfg=dict(num_linear=2, feat_channels=256, return_relu=True),
            out_dim_cont=256,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=10, wandb_name='roi_cls', log_pos_ratio=True),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0001, wandb_name='roi_bbox'),
            loss_cont=dict(type='ContrastiveLossPlus', version='0.2', loss_weight=0.005, num_views=num_views,
                           memory=0, num_classes=9, dim=256,
                           iou_th=0.25, iou_act='clip')
        ),
    ),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars']),
        ),
        analysis_list=[
            dict(type='loss_weight', outputs=dict()),
            dict(type='bbox_head_loss', log_list=['acc_pos', 'acc_neg', 'acc_orig', 'acc_aug2', 'acc_aug3', 'consistency']),
        ]
    ))

###############
### DATASET ###
###############
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMixDetection', num_views=num_views, version='1.4',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', # 'img3',
                               'gt_bboxes', 'gt_bboxes2', # 'gt_bboxes3',
                               'gt_labels']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline)),)

################
### RUN TIME ###
################
runner = dict(
    type='EpochBasedRunner', max_epochs=2)  # actual epoch = 8 * 8 = 64
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[1])

###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

pipeline = 'augmix.det1.4'
loss_type = 'plus'
rpn_loss = 'jsdv1.3.none'
roi_loss = 'jsdv1.3.none.contv0.2'
lambda_weight = '1e-1.10.5e-3'

name = f"{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "kaist-url-ai28",
                                              'name': "augmix.det1.4_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none_contv0.2_clip0.25__e2_lw.1e-1.10.5e-3",
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
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
