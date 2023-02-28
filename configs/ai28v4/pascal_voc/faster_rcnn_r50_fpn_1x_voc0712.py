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
data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
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

dataset = 'voc'
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