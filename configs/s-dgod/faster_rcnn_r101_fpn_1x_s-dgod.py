_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/s-dgod.py',
    '../_base_/default_runtime.py'
]
num_views=2
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(bbox_head=dict(num_classes=7)))
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)    # original: 0.01
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[4, 8])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 10 * 2 = 20
