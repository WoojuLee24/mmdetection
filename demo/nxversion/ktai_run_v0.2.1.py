#!/usr/bin/env python3.6

import rospy, sys, torch, time
import numpy as np
from mmdet.apis import init_detector, inference_detector, init_detector_with_feature
import mmcv
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt

from sensor_msgs.msg import CompressedImage, Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32, Int32
from std_msgs.msg import Float32MultiArray

"""
from KTAI_run_ros_v0.2 py
feature images to points
"""

##### Faster-rcnn #####
# config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

##### Faster-rcnn-augv0.1 #####
# config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.py'
# checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.pth'

##### Mask-rcnn #####
# config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/mask_rcnn_r50_fpn_2x_coco.py'
# checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'


##### YoloX-s #####
config_file = '/home/ktaioneteam/git/mmdetection/configs/openloris/yolox_s_8x8_300e_openloris.py'
checkpoint_file = '/home/ktaioneteam/data/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

##### YoloX-tiny #####
# config_file = '/home/ktaioneteam/git/mmdetection/configs/openloris/yolox_tiny_8x8_300e_openloris.py'
# checkpoint_file = '/home/ktaioneteam/data/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

print("config file: ", config_file)

score_thr = 0.75

# model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = init_detector_with_feature(config_file, checkpoint_file, device='cuda:0')

bridge = CvBridge()
# pub = rospy.Publisher('/node/combined/deep', node, queue_size=10)
# pubSeg = rospy.Publisher('/deep/seg', Image, queue_size=10)
pubBB = rospy.Publisher('/deep/bbox', Float32MultiArray, queue_size=10)
pubFeat0 = rospy.Publisher('/deep/feat0', Float32MultiArray, queue_size=10)
pubFeat1 = rospy.Publisher('/deep/feat1', Float32MultiArray, queue_size=10)
pubFeat2 = rospy.Publisher('/deep/feat2', Float32MultiArray, queue_size=10)
pubFeat3 = rospy.Publisher('/deep/feat3', Float32MultiArray, queue_size=10)

np.random.seed(42)
mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]
bbox_color = (72, 101, 241)
text_color = (72, 101, 241)
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


# pub_ori_img = rospy.Publisher('test1', Image, queue_size=10)
# pub_img = rospy.Publisher('test2', Image, queue_size=10)
# pub_bb = rospy.Publisher('test3', numpy_msg(Float32), queue_size=10)
# pub_mask = rospy.Publisher('test4', numpy_msg(Int32), queue_size=10)

def evalimage(img):
    # t0 = time.time()
    result = inference_detector(model, img)

    features = model.features
    ##### debug #####
    # print("features: ", features)
    # for key, value in model.features.items():
    #    print("key: ", key)
    #    print("value shape: ", value[0].size())

    if len(result) == 2:
        bbox_result, segm_result = result
    else:
        bbox_result = result
        segm_result = None
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
                bbox_int = bbox.astype(np.int32)
                bbox_locs = [(bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3])]
                img = cv2.rectangle(img, bbox_locs[0], bbox_locs[1], bbox_color, 2)

                label_text = class_names[label] if class_names is not None else f'class {label}'
                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)
                # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)

                color_mask = mask_colors[label]
                mask = segm.astype(bool)
                # img.setflags(write=1)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

                target_index = target_names.index(class_names[label]) + 1
                mask_label[mask] = target_index
    else:
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if class_names[label] in target_names:
                bbox_int = bbox.astype(np.int32)
                bbox_locs = [(bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3])]
                img = cv2.rectangle(img, bbox_locs[0], bbox_locs[1], bbox_color, 2)

                label_text = class_names[label] if class_names is not None else f'class {label}'
                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)

    # print(f"{time.time() - t0: .3f}s")

    return bboxes, mask_label, features


def callback_image(data):
    t0 = time.time()
    img = bridge.imgmsg_to_cv2(data)
    img_copy = img.copy()
    bboxes, mask_label, features = evalimage(img_copy)
    t1 = time.time()
    print(f"inference time: {t1 - t0: .3f}s")

    #########################################################
    ####BBOX: (X1, X2, Y1, Y2, Score), BBOX shape: (5, )#####
    #########################################################
    # print("bbox type: ", type(bboxes))
    # print("bbox shape: ", np.shape(bboxes))
    # print("bbox dtype: ", bboxes.dtype)
    # print("bboxes: ", bboxes)

    bboxes = bboxes.flatten()
    bbMsg = Float32MultiArray()
    bbMsg.data = bboxes
    pubBB.publish(bbMsg)

    # for key, value in features.items():
    #    print("key: ", key)
    #    print("value shape: ", value[0].size())
    #######################################################################
    #####neck.fpn_convs.0.conv [1, 256, 192, 336] -> [16515072, ] ######### BCHW
    #####neck.fpn_convs.1.conv [1, 256, 96, 168] -> [4128768, ]############
    #####neck.fpn_convs.2.conv [1, 256, 48, 84] -> [1032192, ]########################
    #####neck.fpn_convs.3.conv [1, 256, 24, 42] -> [258048, ] ########################
    #####neck.fpn_convs.4.conv [1, 256, 12, 21] -> [64512, ]########################
    #######################################################################

    feature_points = []
    for key, feature in features.items():
        feature = feature[0][0]
        feature_ch = feature.mean(dim=0)
        inds = ((feature_ch - feature_ch.mean()) / feature_ch.std() >= 2.56).nonzero()
        feature_selected = feature[:, inds[:, 0], inds[:, 1]]  # (C, N)
        feature_selected = feature_selected.transpose(1, 0)  # (N, C)
        feature_point = torch.cat([inds, feature_selected], dim=1)  # (N, 2+C)
        feature_point = feature_point.flatten()  # (N*(2+C),)
        print(f"feature {key} flatten shape: ", np.shape(feature_point))
        feature_points.append(feature_point)
    t2 = time.time()
    print(f"image2feature time: {t2 - t1: .3f}s")

    feat0Msg = Float32MultiArray()
    feat0Msg.data = feature_points[0]
    pubFeat0.publish(feat0Msg)

    feat1Msg = Float32MultiArray()
    feat1Msg.data = feature_points[1]
    pubFeat1.publish(feat1Msg)

    feat2Msg = Float32MultiArray()
    feat2Msg.data = feature_points[2]
    pubFeat2.publish(feat2Msg)

    # feat3Msg = Float32MultiArray()
    # feat3Msg.data = feature_points[3]
    # pubFeat3.publish(feat3Msg)

    t3 = time.time()
    print(f"feature publishing time: {t3 - t2: .6f}s")
    print(f"total time: {t3 - t0: .3f}s")

    # print(mask_label.dtype)

    # cv2.imwrite("/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/results/origin_imgs/img_{}_{}.png".format(data.header.stamp.secs, data.header.stamp.nsecs), img)
    # cv2.imwrite("/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/results/imgs/img_{}_{}.png".format(data.header.stamp.secs, data.header.stamp.nsecs), img_copy)
    # np.save("/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/results/bboxes/bb_{}_{}".format(data.header.stamp.secs, data.header.stamp.nsecs), bboxes)
    # np.save("/home/ktaioneteam/all_ws/catkin_ws/src/kt_ros/results/segms/mask_{}_{}".format(data.header.stamp.secs, data.header.stamp.nsecs), mask_label)

    # pub_ori_img.publish(bridge.cv2_to_imgmsg(img))
    # pub_img.publish(bridge.cv2_to_imgmsg(img_copy))
    # pub_bb.publish(bboxes)
    # pub_mask.publish(mask_label)


if __name__ == '__main__':
    print("Torch version")
    print(torch.__version__)
    print("Python version")
    print(sys.version)
    print("Version info.")
    print(sys.version_info)
    print("Ready to start!!!")
    try:
        rospy.init_node('kt_ros', anonymous=True)
        rospy.Subscriber("/d400/color/image_raw", Image, callback_image, queue_size=10)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass