_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '../_base_/datasets/s-dgod.py',
    '../_base_/default_runtime.py'
]
num_views=1
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
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
    dict(type='AugMixDetectionFaster', num_views=1, version='1.4.5.1.sdgod',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg,
         fillmode='blur', radius=10, ratios=(0.3, 1 / 0.3),
         num_bboxes=(3, 10), scales=(0.01, 0.2)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]
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

pipeline = 'augmix.det1.4'
loss_type = 'plus'
rpn_loss = 'jsdv1.3.none'
roi_loss = 'jsdv1.3.none'
lambda_weight = '1e-1.0.5'

name = "faster_rcnn_r101_fpn_augmix.det1.4_jsdv1.3_lw1e-1.10_1x_s-dgod"

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