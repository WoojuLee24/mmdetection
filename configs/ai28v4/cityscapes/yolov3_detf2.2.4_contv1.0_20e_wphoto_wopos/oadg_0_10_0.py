_base_ = [
    '/ws/external/configs/ai28v4/cityscapes/yolov3_detf2.2.4_contv1.0_20e_wphoto_wopos/_base_yolov3_d53_mstrain-1024_20e_detf_oadg_wopos.py'
    ]

num_views = 2
wo_pos = True
jsd_conf_weight = 0.0
jsd_cls_weight = 10.0
cont_weight = 0.0

wandb_name = f"yolov3_20e_oadg_{int(jsd_conf_weight)}_{int(jsd_cls_weight)}_{int(cont_weight)}_wphoto_wopos"

# model settings
model = dict(
    bbox_head=dict(
        wo_pos=wo_pos,
        jsd_conf_weight=jsd_conf_weight,
        jsd_cls_weight=jsd_cls_weight,
        cont_cfg=dict(type='1.0', loss_weight=cont_weight, dim=256)))

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/ws/data/cityscapes/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(800, 800), (1024, 1024)], # [(480, 480), (608, 608)],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'), # include?
    dict(type='AugMixDetectionFaster', num_views=num_views, version='2.2.4',
         aug_severity=3, mixture_depth=-1, **img_norm_cfg,
         num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3),
         pre_blur=True, fillmode='var_blur', sigma_ratio=1 / 8, mixture_width=1, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2',
                               'gt_bboxes', 'gt_bboxes2',
                               'gt_labels', 'gt_labels2'])
]
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root + 'leftImg8bit/train/',
        pipeline=train_pipeline))
