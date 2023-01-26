from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os
from mmdet.datasets.coco import COCO
from mmdet.datasets.cityscapes import CityscapesDataset

idx = 0
ann_file = '/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'

# category = 'augment/'
# filename = 'city_augmix.augs_none_rpn.none.none_roi.none.none__e8'
# checkpoint = 'epoch_8.pth'
# category = 'augment/'
# filename = 'city_augmix.det1.4_plus_rpn.jsdv1.3.none_roi.jsdv1.3.none__e2_lw.1e-1.20'
# checkpoint = 'epoch_2.pth'
category = ''
filename = 'faster_rcnn_r50_fpn_1x_cityscapes__e8'
checkpoint = 'epoch_8.pth'

# corruption = 'frost/5'
corruption = 'gaussian_noise/5'
img = f'/ws/data/cityscapes-c/{corruption}/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
work_dir = f'/ws/data2/dshong/grad_cam/{category}{filename}/{corruption.replace("/", "")}'
config = f'/ws/data2/dshong/ai28v4/{category}{filename}/{filename}.py'
checkpoint = f'/ws/data2/dshong/ai28v4/{category}{filename}/{checkpoint}'
device = 'cuda:0'

'''
coco = COCO(ann_file)
coco.CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
def load_annotations(ann_file):
    # The order of returned `cat_ids` will not
    # change with the order of the CLASSES
    coco.cat_ids = coco.get_cat_ids(cat_names=coco.CLASSES)

    coco.cat2label = {cat_id: i for i, cat_id in enumerate(coco.cat_ids)}
    coco.img_ids = coco.get_img_ids()
    data_infos = []
    coco.total_ann_ids = []
    for i in coco.img_ids:
        info = coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        data_infos.append(info)
        ann_ids = coco.get_ann_ids(img_ids=[i])
        coco.total_ann_ids.extend(ann_ids)
    assert len(set(coco.total_ann_ids)) == len(
        coco.total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    return data_infos

coco.data_infos = load_annotations(ann_file)
img = f"/ws/data/cityscapes/leftImg8bit/train/{coco.data_infos[idx]['file_name']}"
ann_ids = coco.total_ann_ids[idx]
ann_info = coco.load_anns(ann_ids)

def _parse_ann_info(self, img_info, ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue
        if ann['category_id'] not in self.cat_ids:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        bboxes_ignore=gt_bboxes_ignore,
        masks=gt_masks_ann,
        seg_map=img_info['segm_file'])

    return ann
ann = _parse_ann_info(coco, coco.data_infos[idx], ann_info)
'''

# build the model from a config file and a checkpoint file
model = init_detector(config, checkpoint, device=device)
model.grad_cam = True
# test a single image
image = cv2.imread(img)
height, width, channels = image.shape

model.train_cfg = {}
# hello = inference_detector(model, img)
results, x_backone, x_fpn = inference_detector(model, img)

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# GradCam for backbone features
feature_index = 0
for feature in x_backone:
    feature_index += 1
    P = torch.sigmoid(feature)
    P = P.cpu().detach().numpy()
    P = np.maximum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    P = P.squeeze(0)
    print(P.shape)

    P = P[10, ...]  # 挑选一个通道
    print(P.shape)

    cam = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image = np.uint8(255 * heatmap)

    cv2.imwrite(f'{work_dir}/' + 'stage_' + str(feature_index) + '_heatmap.jpg', heatmap_image)
    result = cv2.addWeighted(image, 0.8, heatmap_image, 0.3, 0)
    cv2.imwrite(f'{work_dir}/' + 'stage_' + str(feature_index) + '_result.jpg', result)

# GradCam for FPN features
feature_index = 1
for feature in x_fpn:
    feature_index += 1
    P = torch.sigmoid(feature)
    P = P.cpu().detach().numpy()
    P = np.maximum(P, 0)
    P = (P - np.min(P)) / (np.max(P) - np.min(P))
    P = P.squeeze(0)
    P = P[2, ...]
    print(P.shape)
    cam = cv2.resize(P, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)
    heatmap_image = np.uint8(255 * heatmap)

    cv2.imwrite(f'{work_dir}/' + 'P' + str(feature_index) + '_heatmap.jpg', heatmap_image)  # 生成图像
    result = cv2.addWeighted(image, 0.8, heatmap_image, 0.4, 0)
    cv2.imwrite(f'{work_dir}/' + 'P' + str(feature_index) + '_result.jpg', result)

# Visualize prediction results
import matplotlib.pyplot as plt
def pixel2inch(pixel):
    PIXEL2INCH = 0.0104166667
    return pixel * PIXEL2INCH

height, width = image.shape[0], image.shape[1]
height_inch, width_inch = pixel2inch(height), pixel2inch(width)
fig, ax = plt.subplots(1, 1, figsize=(width_inch, height_inch,))

if torch.is_tensor(image):
    image = image.permute(1, 2, 0).cpu().detach()
ax.imshow(image, aspect='auto')
ax.axis('off') # 축 없애기
ax.set_xticks([]); ax.set_yticks([]) # 틱 없애기
fig.tight_layout() # 공백을 잘 배치
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

from thirdparty.dscv.utils.detection_utils import visualize_bboxes_xy

for i in range(len(results[0])):
    bboxes = results[0][i]
    visualize_bboxes_xy(bboxes, fig=fig, ax=ax, color_idx=i)
fig.savefig(f'{work_dir}/prediction_result.jpg')
plt.close(fig)

