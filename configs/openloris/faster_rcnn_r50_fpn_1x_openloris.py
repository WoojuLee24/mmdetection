_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/openloris_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=['neck.fpn_convs.0.conv',
                     'neck.fpn_convs.1.conv',
             ])
]
log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                  ])

load_from = '/ws/data/OpenLORIS/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'