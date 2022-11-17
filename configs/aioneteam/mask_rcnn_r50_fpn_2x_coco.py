_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=["neck.fpn_convs.0.conv",
                     "neck.fpn_convs.1.conv",
                     "neck.fpn_convs.2.conv",
                     "neck.fpn_convs.3.conv"
                     ]),
]
