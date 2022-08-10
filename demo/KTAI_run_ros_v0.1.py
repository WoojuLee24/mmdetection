import argparse

import cv2
import torch
import numpy as np

import mmcv
from mmdet.apis import inference_detector, init_detector

import time


import rospy
from gpSLAM_ICRA.msg import node, group
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


bridge = CvBridge()
# pub = rospy.Publisher('/node/combined/deep', node, queue_size=10)
pub = rospy.Publisher('/node/combined/grouped/deep', group, queue_size=10)


def parse_args():
    parser = argparse.ArgumentParser(description='KTAI perception ros run')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    # parser.add_argument('--config', default='../configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py', help='Config file')
    # parser.add_argument('--checkpoint', default='../weights/cascade_mask_rcnn_swin_small_patch4_window7.pth', help='Checkpoint file')

    parser.add_argument('--config', default='../configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='../weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth', help = 'Checkpoint file')

    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--score-thr', type=float, default=0.35, help='bbox score threshold')
    args = parser.parse_args()
    return args



args = parse_args()

device = torch.device(args.device)

np.random.seed(42)
mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]

model = init_detector(args.config, args.checkpoint, device=device)

# cv2.namedWindow("result", cv2.WINDOW_GUI_EXPANDED)

# SETTINGS
bbox_color = (72, 101, 241)
text_color = (72, 101, 241)
score_thr = args.score_thr
class_names = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
target_names = ['person', 'chair', 'couch', 'table', 'tv']


def evalimage(img):
    t0 = time.time()
    result = inference_detector(model, img)
    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)

    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_label = np.zeros(img.shape[:2], dtype=np.int32)
    if segms is not None:
        for i, (bbox, segm, label) in enumerate(zip(bboxes, segms, labels)):
            if class_names[label] in target_names:
                # bbox_int = bbox.astype(np.int32)
                # bbox_locs = [(bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3])]
                # img = cv2.rectangle(img, bbox_locs[0], bbox_locs[1], bbox_color, 2)
                #
                # label_text = class_names[label] if class_names is not None else f'class {label}'
                # if len(bbox) > 4:
                #     label_text += f'|{bbox[-1]:.02f}'
                # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)
                # # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)
                #
                # color_mask = mask_colors[label]
                mask = segm.astype(bool)
                # img[mask] = img[mask] * 0.5 + color_mask * 0.5
                #
                target_index = target_names.index(class_names[label]) + 1
                mask_label[mask] = target_index

    print(f"{time.time() - t0: .3f}s")

    # cv2.imshow("result", img)
    # cv2.waitKey(1)

    return mask_label


def callback_node(data: group):
    # for i in range(data.nodes):
    for node in data.nodes:
        img = bridge.compressed_imgmsg_to_cv2(node.image)
        seg_pred = evalimage(img).astype(np.uint8)
        node.segmentImage = bridge.cv2_to_compressed_imgmsg(seg_pred)
        node.flagSegmented.data = True
    pub.publish(data)

    # cv2.imshow("img", img)
    # cv2.imwrite("img", img)

    # seg_pred = evalimage(net, img)

    # newNode = data
    # newNode.segmentImage = bridge.cv2_to_compressed_imgmsg(seg_pred)
    # newNode.flagSegmented.data = True
    # pub.publish(newNode)


if __name__ == '__main__':
    try:
        rospy.init_node('deepNet', anonymous=True)
        # rospy.Subscriber("/node/combined/raw", node, callback_node,queue_size=100)
        rospy.Subscriber("/node/combined/grouped/raw", group, callback_node, queue_size=100)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
