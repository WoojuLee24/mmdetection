_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/cityscapes_detection_augmix.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(init_cfg=None),
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossAugMix', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1LossAugMix', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLossAugMix', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        augmix=dict(
            layer_list=["rpn_head.rpn_cls",
                        "roi_head.bbox_head.fc_cls",]),
        wandb=dict(),
        jsd_loss_parameter=0.001,
        is_debugging=True,
        loss_type_list={'rpn_head.rpn_cls': 'jsd_new',
                        'roi_head.bbox_head.fc_cls': 'jsd_new'}))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=1)  # actual epoch = 8 * 8 = 64
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=["rpn_head.rpn_cls", "rpn_head.rpn_reg",
                    "roi_head.bbox_head.fc_cls", "roi_head.bbox_head.fc_reg"]),
]
log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "ai28",
                                              'name': "augmix_with_jsdy-rpn-roi-cls"},
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
