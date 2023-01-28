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
    samples_per_gpu=2, workers_per_gpu=2,)

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

train_version = 'v5'
dataset = 'coco'
pipeline = 'none'
loss_type = 'none'
rpn_loss = 'none.none'
roi_loss = 'none.none'
lambda_weight = ''

name = f"{train_version}_{dataset}_{pipeline}.{loss_type}_rpn.{rpn_loss}_roi.{roi_loss}__e{str(runner['max_epochs'])}_lw.{lambda_weight}"

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

############
### LOAD ###
############
