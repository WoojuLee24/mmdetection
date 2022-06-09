_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_ai28.py',
    '../../_base_/datasets/cityscapes_detection_augmix_without_obj_translation.py',
    '../../_base_/schedules/ai28.py',
    '../../_base_/default_runtime.py'
]

'''
[OPTIONS]
  model
  * loss_cls/loss_bbox.additional_loss
    : [None, 'jsd', 'jsdy']
  * train_cfg.wandb.log.features_list 
    : [None, "rpn_head.rpn_cls", "neck.fpn_convs.0.conv", "neck.fpn_convs.1.conv", "neck.fpn_convs.2.conv", "neck.fpn_convs.3.conv"] 
'''

model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0
            , additional_loss='jsd', lambda_weight=0.0001, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0
                       , additional_loss=None, lambda_weight=0.0001, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0
                , additional_loss=None, lambda_weight=0.0001, wandb_name='roi_cls'),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0
                           , additional_loss=None, lambda_weight=0.0001, wandb_name='roi_bbox'))),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=["rpn_head.rpn_cls"],
                vars=['log_vars'],
                ))))

custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

rpn_loss_cls = model['rpn_head']['loss_cls']
rpn_loss_bbox = model['rpn_head']['loss_bbox']
roi_loss_cls = model['roi_head']['bbox_head']['loss_cls']
roi_loss_bbox = model['roi_head']['bbox_head']['loss_bbox']

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "ai28",
                                              'name': f"augmix_wo_trans.rpn.{rpn_loss_cls['additional_loss']}.{rpn_loss_bbox['additional_loss']}_"
                                                      f"roi.{roi_loss_cls['additional_loss']}.{roi_loss_bbox['additional_loss']}_",
                                              'config': {
                                                  'loss_type(rpn_cls)': f"{rpn_loss_cls['type']}({rpn_loss_cls['additional_loss']})",
                                                  'loss_type(rpn_bbox)': f"{rpn_loss_bbox['type']}({rpn_loss_bbox['additional_loss']})",
                                                  'loss_type(roi_cls)': f"{roi_loss_cls['type']}({roi_loss_cls['additional_loss']})",
                                                  'loss_type(roi_bbox)': f"{roi_loss_bbox['type']}({roi_loss_bbox['additional_loss']})",
                                                  'aug_list': f"augmix_wo_trans"
                                              }},
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
