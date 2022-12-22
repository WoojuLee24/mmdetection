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
'''

#############
### MODEL ###
#############
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0
            , additional_loss='None', lambda_weight=0.0, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0
                       , additional_loss="None", lambda_weight=0.0, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            num_cls_convs=0, num_cls_fcs=1,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0
                , additional_loss='None', lambda_weight=0.0, wandb_name='roi_cls'),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0
                           , additional_loss="None", lambda_weight=0.0, wandb_name='roi_bbox')),
        loss_feat=dict(type='ContrastiveLossPlus', version='0.0.4', loss_weight=1.0)
    ),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars']),
        )))

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
    dict(type='RandAutoAugment', policy='0.1', num_gen=2, min_depth=1, max_depth=3, min_level=2, max_level=8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'img3',
                               'gt_bboxes', 'gt_bboxes2', 'gt_bboxes3',
                               'gt_labels', 'gt_labels2', 'gt_labels3',
                               'gt_instance_inds', 'gt_instance_inds2', 'gt_instance_inds3',
                               ]),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
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

pipeline = 'randautoaug.0.1'
loss_type = 'plus'
rpn_loss = 'none.none'
roi_loss = 'none.none'
cont_loss = '0.0.4'
lambda_weight = '0.0.1'

name = f"{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}_cont.{cont_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "kaist-url-ai28",
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
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

