#!/usr/bin/env python3.6

import rospy, sys, setuptools, torch, time
import numpy as np
from mmdet.apis import init_detector, inference_detector, init_detector_with_feature
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

# show = True

args = parse_args()

model = init_detector(args.config, args.checkpoint, device='cuda:0')


score_thr = 0.75

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

prev_feature = torch.zeros(256, 120, 216, dtype=torch.float32).cuda()

def evalimage(img):
    result = inference_detector(model, img)

    features = model.fpn_features

    if len(result) == 2:
        bbox_result, segm_result = result
    else:
        bbox_result = result
        segm_result = None
    bboxes_ = np.vstack(bbox_result)
    labels_ = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels_ = np.concatenate(labels_)

    # print("bboxes: ", bboxes)
    # print("bboxes shape: ", np.shape(bboxes))
    # print("labels: ", labels)
    # print("labels shape: ", np.shape(labels))

    bboxes = []
    labels = []

    for i, (bbox, label) in enumerate(zip(bboxes_, labels_)):
        if class_names[label] in target_names:
            # target_index = target_names.index(class_names[label]) + 1
            bboxes.append(bbox)
            labels.append(label)
    print("labels: ", labels)

    # print("bboxes_target: ", bboxes_target)
    # print("labels_target: ", labels_target)

    bboxes = np.vstack(bboxes)
    labels = np.array(labels)

    # print("bboxes_target: ", bboxes_target)
    # print("labels_target: ", labels_target)

    # Draw segmentation masks
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
                target_index = target_names.index(class_names[label]) + 1
                mask_label[mask] = target_index
    return bboxes, labels, features


def callback_img(data):
    img = bridge.compressed_imgmsg_to_cv2(data)
    img_copy = img.copy()
    print('img_shape:', np.shape(img_copy))

    t0 = time.time()
    with torch.no_grad():
        bboxes, labels, feats = evalimage(img_copy)
    t1 = time.time()
    print(f"inference time: {t1 - t0: .3f}s")

    # print(np.shape(bboxes))
    bboxes = bboxes.flatten()

    t0 = time.time()
    feature_points = []
    feature_maps = []

    if args.name == 'yolo':
        selected_feature = 2
    elif args.name == 'fast':
        selected_feature = 0

    with torch.no_grad():
        for i, feature in enumerate(feats):
            feature = feature[0]

            if args.show and i==selected_feature:
                feature_npy = feature.mean(dim=0).cpu().detach().numpy()
                # plt.imsave(f'/ws/data/log_kt/demo/debug/{t0}_img.png', img_copy)
                plt.imsave(f'/ws/data/log_kt/demo/debug/{t0}_{i}.png', feature_npy, cmap='gray')
                # print('feature_npy: ', feature_npy)


            # feature_point_cat = []
            # if i == selected_feature:
            #     for j in range(4):
            #         print(f"feature {j} shape: ", feature.size())
            #         feature_ = feature[j*128:(j+1)*128]
            #         print(f"feature {j} shape slice: ", feature_.size())
            #
            #         # get feature points
            #         inds = get_feature_percentage(feature_)
            #         print("inds size: ", inds.size())
            #         # inds = get_feature_thres_value(feature, thres=0.05)
            #
            #         # print("inds: ", inds)
            #         feature_selected = feature[:, inds[:, 0], inds[:, 1]]  # (C, N)
            #         feature_selected = feature_selected.transpose(1, 0)  # (N, C)
            #         feature_point = torch.cat([inds, feature_selected], dim=1)  # (N, 2+C)
            #         feature_point = feature_point.flatten()  # (N*(2+C),)
            #         print('feature point shape: ', feature_point.size())
            #         feature_point_cat.append(feature_point)
            #
            #     feature_point_cat = torch.cat(feature_point_cat)
            #     print('feature point cat shape: ', feature_point_cat.size())
            # feature_points.append(feature_point_cat)

            # get feature points v1
            feature = feature[:128]
            inds = get_feature_percentage(feature)

            feature_selected = feature[:, inds[:, 0], inds[:, 1]]  # (C, N)
            feature_selected = feature_selected.transpose(1, 0)  # (N, C)
            feature_point = torch.cat([inds, feature_selected], dim=1)  # (N, 2+C)
            feature_point = feature_point.flatten()  # (N*(2+C),)
            feature_points.append(feature_point)
            feature_maps.append(feature)
            print('feature_point shape: ', feature_point.size())



        print("feature_points: ", len(feature_points))

        t1 = time.time()
        print(f"feature time: {t1 - t0: .3f}s")
        print("-----")

        ## debug
        img_debug = img.copy()
        img_H = np.shape(img_debug)[0]  # 480
        img_W = np.shape(img_debug)[1]  # 848

        print('img_H: ', img_H)
        print('img_W: ', img_W)

        # bbox : (X1, X2, Y1, Y2, Score)
        # print('bboxes:', bboxes)

        for i in range(0, len(bboxes), 5):
            img_debug = cv2.rectangle(img_debug, (int(bboxes[i]), int(bboxes[i + 1])),
                                      (int(bboxes[i + 2]), int(bboxes[i + 3])), (255, 0, 0), 3)

        channel = 128
        xy = 2
        stride = channel + xy
        if args.name == 'yolo':
            scale_x = 76
            scale_y = 44
            feat_0 = feature_points[2]
            print("feat_0 shape: ", feat_0.size())
            N_feat0 = len(feat_0) // stride
        elif args.name == 'fast':
            scale_x = 216 #120  # 120, 60, 30
            scale_y = 120 #216  # 216, 108, 54
            feat_0 = feature_points[0]
            print("feat_0 shape: ", feat_0.size())
            N_feat0 = len(feat_0) // stride
        elif args.name == 'yoloX':
            scale_x = 80  # 80, 40, 20
            scale_y = 80  # 80, 40, 20


        ## feature 정보 이미지로 만들기
        # featImg = np.zeros((img_H, img_W), dtype=np.uint8)  # 480 * 848

        t0 = time.time()
        for i in range(N_feat0):


            print("x, y", feat_0[i*stride+1].item(), feat_0[i*stride].item())
            up_y = round(feat_0[i * stride].item() * img_H / scale_y)
            up_x = round(feat_0[i * stride + 1].item() * img_W / scale_x)
            # up_x = round(feat_0[i * stride].item() * img_W / scale_x)
            # up_y = round(feat_0[i * stride + 1].item() * img_H / scale_y)
            print("x, y", feat_0[i*stride+1].item(), feat_0[i*stride].item())

            img_debug = cv2.circle(img_debug, (up_x, up_y), 6, (0, 0, 255), -1)
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


        plt.imsave(f'/ws/data/log_kt/demo/debug/{t0}_debug.png', img_debug)

        img_msg = bridge.cv2_to_imgmsg(img_debug, encoding="passthrough")
        # img_msg2 = bridge.cv2_to_imgmsg(featImg, encoding="passthrough")

        pub_featureImg.publish(img_msg)


def get_feature_percentage(feature):
    # print("value before shape: ", feature.size())
    feature = feature[:128]
    # print("value after shape: ", feature.size())
    feature_ch = feature.mean(dim=0)

    th = 0.0

    while (1):
        inds = ((feature_ch - feature_ch.mean()) / feature_ch.std() >= th).nonzero()
        inds = inds[inds[:, 0] > 0]
        inds = inds[inds[:, 1] > 0]

        if np.shape(inds)[0] > 500:
            th = th + 0.1
        else:
            break

    return inds


def get_feature_percentage2(feature, th=0.5):
    # print("value before shape: ", feature.size())
    # print("value after shape: ", feature.size())

    C, H, W = feature.size()

    indexes = []
    for i in range(C):
        feat = feature[i]
        feat_mean = feat.mean()
        feat_std = feat.std()
        # print("feat: ", feat.size())
        # print("mean: ", feat_mean.size())
        # print("std: ", feat_std.size())

        while (1):
            inds = ((feat - feat_mean) / feat_std >= th).nonzero()
            # print("inds size: ", inds.size())
            inds = inds[inds[:, 0] > 0]
            inds = inds[inds[:, 1] > 0]

            if np.shape(inds)[0] > 20:
                th = th + 0.1
            else:
                break
        # print("inds size: ", inds.size())
        indexes.append(inds)
    indexes = torch.cat(indexes, dim=0)

    print("indexes size: ", indexes.size())

    return indexes


def get_feature_thres_value(feature, thres=0.5):

    feature_ch = feature.mean(dim=0)
    inds = ((feature_ch - feature_ch.mean()) >= thres).nonzero()

    return inds


if __name__ == '__main__':

    print("------ Start !!! ------")
    try:
        rospy.init_node('deepNet', anonymous=True)
        rospy.Subscriber("/rgb/image_rect_color/compressed", CompressedImage, callback_img, queue_size=100)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass