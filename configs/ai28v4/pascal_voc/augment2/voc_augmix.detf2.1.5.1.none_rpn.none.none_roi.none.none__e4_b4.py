_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ws/external/configs/_base_/datasets/voc0712.py',
    '/ws/external/configs/_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=20)),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars']),
        ),
        analysis_list=[
            # dict(type='loss_weight', outputs=dict()),
            # dict(type='bbox_head_loss',
            #      log_list=['acc_pos', 'acc_neg', 'acc_orig', 'consistency']),
        ]
    )
)

###############
### DATASET ###
###############
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix_detection_faster'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMixDetectionFaster', num_views=1, version='2.1',
         aug_severity=3, mixture_depth=(1,6), **img_norm_cfg,
         num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3),
         pre_blur=True, fillmode='var_blur', sigma_ratio=1 / 6,
         mixture_width=1, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(dataset=dict(pipeline=train_pipeline))
)

################
### RUN TIME ###
################
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

train_version = 'v4'
dataset = 'voc'
pipeline = 'augmix.det2.1.5.1'
loss_type = 'none'
rpn_loss = 'none.none'
roi_loss = 'none.none'
lambda_weight = ''

name = f"{train_version}_{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      # dict(type='WandbLogger',
                      #      wandb_init_kwargs={'project': "AI28", 'entity': "kaist-url-ai28",
                      #                         'name': name,
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