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
        wandb=dict(log=dict( features_list=[], vars=['log_vars'])),
        analysis_list=[
            dict(type='loss_weight', outputs=dict()),
            dict(type='bbox_head_loss',
                 log_list=['acc_pos', 'acc_neg', 'acc_orig', 'consistency']),
        ]
    )
)

###############
### DATASET ###
###############
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,)

################
### RUN TIME ###
################
# optimizer
# lr is set for a batch size of 8
optimizer_config = dict(grad_clip=None)
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

dataset = 'coco'
pipeline = 'none'
loss_type = 'none'
rpn_loss = 'none.none'
roi_loss = 'none.none'
lambda_weight = ''

name = f"{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28v3", 'entity': "kaist-url-ai28",
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
