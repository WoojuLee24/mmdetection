#!/usr/bin/env python3.6

import rospy, sys, setuptools, torch, time
import numpy as np
from mmdet.apis import init_detector, inference_detector # init_detector_with_feature
import mmcv
# from gpSLAM_ICRA.msg import node
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--name', default='fast', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--show', action='store_true', help='debug option')
    parser.add_argument('--work-dir', action='store_true', help='debug option')

    args = parser.parse_args()

    return args


print("------ Please wait model loading ------")

# name = 'yolo'  # yolo, fast, yoloX

# ##### Yolo-v3 #####
# if name == 'yolo':
#     # config_file = '/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/config/yolov3_d53_mstrain-608_7e_coco_augv0.1.1.py'
#     # checkpoint_file = '/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/config/epoch_7.pth'
#     config_file = '/ws/external/configs/yolo/yolov3_d53_mstrain-608_7e_coco_augv0.1.1.py'
#     checkpoint_file = '/ws/data/log_kt/checkpoints/yolo/yolov3_d53_mstrain-608_7e_coco_augv0.1.1-test-lw50-lr0.0001/epoch_7.pth'
# ##### Faster-rcnn-augv0.1 #####
# elif name == 'fast':
#     config_file = '/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/config/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.py'
#     checkpoint_file = '/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/config/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.pth'
# ##### YoloX-s #####
# elif name == 'yoloX':
#     config_file = '/home/ktaioneteam/git/mmdetection/configs/openloris/yolox_s_8x8_300e_openloris.py'
#     checkpoint_file = '/home/ktaioneteam/data/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'


args = parse_args()
mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]


if args.name =='yolact':
    print("yolact img shape is fixed to 550")

mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]

model = init_detector(args.config, args.checkpoint, device='cuda:0')

score_thr = 0.3

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

bridge = CvBridge()
pub_featureImg = rospy.Publisher('/featureImg', Image, queue_size=10)
pub_segImg = rospy.Publisher('/segImg', Image, queue_size=10)


def evalimage(img):
    result = inference_detector(model, img)

    features = model.fpn_features
    # print("features shape1: ", features[0].size())
    # print("features shape2: ", features[1].size())

    if len(result) == 2:
        bbox_result, segm_result = result
    else:
        bbox_result = result
        segm_result = None
    bboxes = np.vstack(bbox_result)
    labels_ = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels_)

    # print("bboxes: ", bboxes_)
    # print("labels: ", labels_)

    # print("bboxes shape: ", np.shape(bboxes))
    # print("labels shape: ", np.shape(labels))

    # Draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        # print('segms shape1: ', np.shape(segms))
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
        # print('segms shape2: ', np.shape(segms))
    # print('segms: ',segms)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        # print('inds: ', inds)
        # print('segms shape4: ', np.shape(segms))
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_segms = np.zeros(img.shape[:2], dtype=np.int32)
    mask_bboxes = []
    mask_labels = []

    if segms is not None:
        for i, (bbox, segm, label) in enumerate(zip(bboxes, segms, labels)):
            if class_names[label] in target_names:
                mask = segm.astype(bool)
                target_index = target_names.index(class_names[label]) + 1
                mask_segms[mask] = target_index
                color_mask = mask_colors[label]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

                mask_bboxes.append(bbox)
                mask_labels.append(label)

    else:
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if class_names[label] in target_names:
                target_index = target_names.index(class_names[label]) + 1
                mask_bboxes.append(bbox)
                mask_labels.append(label)

    if len(mask_bboxes) > 0:
        mask_bboxes = np.vstack(mask_bboxes)
        mask_labels = np.array(mask_labels)
    else:
        mask_bboxes = np.array(mask_bboxes)
        mask_labels = np.array(mask_labels)

    return mask_bboxes, mask_labels, mask_segms, features, img


def callback_img(data):
    img = bridge.compressed_imgmsg_to_cv2(data)
    img_copy = img.copy()

    t0 = time.time()
    bboxes, labels, segms, feats, mask_image = evalimage(img_copy)
    t1 = time.time()
    print(f"inference time: {t1 - t0: .3f}s")

    print("segms shape: ", np.shape(segms))



    # print(np.shape(bboxes))
    bboxes = bboxes.flatten()

    t0 = time.time()
    feature_points = []
    for key, feature in enumerate(feats):
        feature = feature[0]
        c, h, w = feature.size()
        # print("key: ", key)
        print("value before shape: ", feature.size())
        feature = feature[:128]
        # print("value after shape: ", feature.size())
        feature_ch = feature.mean(dim=0)
        th = 0.0
        while (1):
            inds = ((feature_ch - feature_ch.mean()) / feature_ch.std() >= th).nonzero()
            # inds = inds[inds[:, 0] > 0]
            # inds = inds[inds[:, 0] < w]
            # inds = inds[inds[:, 1] > 0]
            # inds = inds[inds[:, 1] < h]


            inds = inds[inds[:, 0] > int(w/9)]
            inds = inds[inds[:, 1] > int(h/9)]
            inds = inds[inds[:, 0] < w - int(w/9)]
            inds = inds[inds[:, 1] < h - int(h/9)]

            # inds = inds[w - int(w / 9) > inds[:, 0] > int(w / 9)]
            # inds = inds[h - int(h / 9) >inds[:, 1] > int(h / 9)]


            if np.shape(inds)[0] > 50:
                th = th + 0.1
            else:
                break
        feature_selected = feature[:, inds[:, 0], inds[:, 1]]  # (C, N)
        feature_selected = feature_selected.transpose(1, 0)  # (N, C)
        feature_point = torch.cat([inds, feature_selected], dim=1)  # (N, 2+C)
        feature_point = feature_point.flatten()  # (N*(2+C),)
        feature_points.append(feature_point)
    t1 = time.time()
    print(f"feature time: {t1 - t0: .3f}s")
    print("-----")

    ## debug
    img_debug = img.copy()
    img_H = np.shape(img_debug)[0]  # 480
    img_W = np.shape(img_debug)[1]  # 848

    # bbox : (X1, X2, Y1, Y2, Score)
    for i in range(0, len(bboxes), 5):
        img_debug = cv2.rectangle(img_debug, (int(bboxes[i]), int(bboxes[i + 1])),
                                  (int(bboxes[i + 2]), int(bboxes[i + 3])), (255, 0, 0), 3)

    channel = 128
    xy = 2
    stride = channel + xy
    feat_0 = feature_points[0].cpu().numpy()
    feat_1 = feature_points[1].cpu().numpy()
    feat_2 = feature_points[2].cpu().numpy()

    N_feat0 = len(feat_0) // stride
    N_feat1 = len(feat_1) // stride
    N_feat2 = len(feat_2) // stride

    # print("N_feat0: ", N_feat0)
    # print("N_feat1: ", N_feat1)

    if args.name == 'yolo':
        scale_x_0 = 11
        scale_y_0 = 19
        scale_x_1 = 22
        scale_y_1 = 38
        scale_x_2 = 44
        scale_y_2 = 76
    elif args.name == 'yolact':
        scale_x_0 = 69
        scale_y_0 = 69
        scale_x_1 = 35
        scale_y_1 = 35
        scale_x_2 = 18
        scale_y_2 = 18

    elif args.name == 'fast':
        scale_x = 120  # 120, 60, 30
        scale_y = 216  # 216, 108, 54
        scale_x = 80  # 80, 40, 20
        scale_y = 80  # 80, 40, 20

    ## feature 정보 이미지로 만들기
    # featImg = np.zeros((img_H, img_W), dtype=np.uint8)  # 480 * 848

    t0 = time.time()
    for i in range(N_feat0):
        feat_0[i * stride] = int(round(feat_0[i * stride] * img_H / scale_x_0))
        feat_0[i * stride+1] = int(round(feat_0[i * stride+1] * img_W / scale_y_0))
        # print('y, x ',feat_0[i * stride], feat_0[i * stride+1])
        img_debug = cv2.circle(img_debug, (int(feat_0[i * stride+1]), int(feat_0[i * stride])), 6, (0, 0, 255), -1)
        # print('y, x ',feat_0[i * stride], feat_0[i * stride+1])

    for i in range(N_feat1):
        feat_1[i * stride] = round(feat_1[i * stride] * img_H / scale_x_1)
        feat_1[i * stride + 1] = round(feat_1[i * stride + 1] * img_W / scale_y_1)
        img_debug = cv2.circle(img_debug, (int(feat_1[i * stride+1]), int(feat_1[i * stride])), 6, (0, 255, 0), -1)
        # print('y, x ',feat_1[i * stride], feat_1[i * stride+1])

    for i in range(N_feat2):
        feat_2[i * stride] = round(feat_2[i * stride] * img_H / scale_x_2)
        feat_2[i * stride + 1] = round(feat_2[i * stride + 1] * img_W / scale_y_2)
        img_debug = cv2.circle(img_debug, (int(feat_2[i * stride + 1]), int(feat_2[i * stride])), 6, (255, 0, 0), -1)


        #### 아래 부분은 피쳐정보를 segmentation image 처럼 이미지화하기 위함.
        #### 굳이 필요없음.
        # feat_sum = 0
        # for j in range(channel):
        #     feat_sum = feat_sum + feat_0[i*stride + j].item()
        # feat_sum = feat_sum / 1
        # if round(feat_sum) < 0:
        #     featImg[up_x, up_y] = 0
        # elif round(feat_sum) > 255:
        #     featImg[up_x, up_y] = 255
        # else: featImg[up_x, up_y] = round(feat_sum)
        # print("x, y, f", up_x, up_y, feat_sum, featImg[up_x, up_y])

    img_msg = bridge.cv2_to_imgmsg(img_debug, encoding="passthrough")
    seg_msg = bridge.cv2_to_imgmsg(mask_image, encoding="passthrough")

    # img_msg2 = bridge.cv2_to_imgmsg(featImg, encoding="passthrough")

    pub_featureImg.publish(img_msg)
    pub_segImg.publish(seg_msg)


if __name__ == '__main__':
    # print("Torch version")
    # print(torch.__version__)
    # print("Python version")
    # print(sys.version)
    # print("Version info.")
    # print(sys.version_info)
    print("------ Start !!! ------")
    try:
        rospy.init_node('deepNet', anonymous=True)
        rospy.Subscriber("/rgb/image_rect_color/compressed", CompressedImage, callback_img, queue_size=100)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass