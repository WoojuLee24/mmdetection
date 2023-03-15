_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '../_base_/datasets/s-dgod.py',
    '../_base_/default_runtime.py'
]
num_views=2
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    rpn_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=0.001, wandb_name='rpn_cls'),
            loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0001, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=1, wandb_name='roi_cls', log_pos_ratio=True),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0001, wandb_name='roi_bbox'),
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
    dict(type='AugMixDetectionFaster', num_views=num_views, version='2.7.all',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg,
         num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1/0.3),
         pre_blur=True, fillmode='var_blur', sigma_ratio=1/8, mixture_width=1,),
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

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)    # original: 0.01
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = * 2 => 8 , 16
lr_config = dict(policy='step', step=[4, 8])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 10 * 2 = 20


###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

pipeline = 'augmix.detf2.2.4'
loss_type = 'plus'
rpn_loss = 'jsdv1.3.none'
roi_loss = 'jsdv1.3.none'
lambda_weight = '1e-1.10'

name = f"{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"


print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      # dict(type='WandbLogger',
                      #      wandb_init_kwargs={'project': "AI28", 'entity': "kaist-url-ai28",
                      #                         'name': "faster_rcnn_r101_fpn_augmix.det1.4_jsdv1.3_lw1e-1.10_1x_s-dgod",
                      #                         'config': {
                      #                             # data pipeline
                      #                             'data pipeline': f"{pipeline}",
                      #                             # losses
                      #                             'loss type(rpn)': f"{rpn_loss}",
                      #                             'loss type(roi)': f"{roi_loss}",
                      #                             # parameters
                      #                             'epoch': runner['max_epochs'],
                      #                             'lambda_weight': lambda_weight,
                      #                         }},
                      #      interval=500,
                      #      log_checkpoint=True,
                      #      log_checkpoint_metadata=True,
                      #      num_eval_images=5),
                  ]
                  )