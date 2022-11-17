#!/usr/bin/env python3.6

import rospy, sys, torch, time, signal, math
import threading
import numpy as np
from mmdet.apis import init_detector, inference_detector, init_detector_with_feature
import mmcv
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt

from sensor_msgs.msg import CompressedImage, Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
from kt_ros.msg import Node
from mmdeploy_python import Detector


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class kt_ros():
    def __init__(self):

        rospy.init_node('kt_ros', anonymous=True)
        ##### Faster-rcnn #####
        self.config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        ##### Faster-rcnn-augv0.1 #####
        # self.config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.py'
        # self.checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/faster_rcnn_r50_fpn_1e_openloris_aug_v0.1.pth'
        ##### Mask-rcnn #####
        # self.config_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/mask_rcnn_r50_fpn_2x_coco.py'
        # self.checkpoint_file = '/home/ktaioneteam/git/mmdetection/configs/aioneteam/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'

        self.score_thr = 0.75
        # self.model = init_detector_with_feature(self.config_file, self.checkpoint_file, device='cuda:0')
        self.model = Detector('/home/ktaioneteam/git/mmdeploy/results/faster_rcnn2/', 'cuda:0')
        # self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')

        self.bridge = CvBridge()

        self.sub_img = rospy.Subscriber("/d400/color/image_raw", Image, self.image_cb, queue_size=1)
        self.pub_result = rospy.Publisher('/deep/result', Node, queue_size=1)

        np.random.seed(43)
        self.mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]
        self.bbox_color = (72, 101, 241)
        self.text_color = (72, 101, 241)
        self.class_names = (
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        self.target_names = ['person', 'chair', 'couch', 'table', 'tv']

        self.img_buffer = np.empty((0, 480, 848, 3), np.uint8)
        self.time_buffer = np.empty((0,), float)
        self.infer_end_time = 0.0

        self.verbose = True
        self.rate = rospy.Rate(5)
        self.mutex = threading.Lock()

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)
        self.mutex.acquire()
        self.img_buffer = np.append(self.img_buffer, np.array([img]), axis=0)
        self.time_buffer = np.append(self.time_buffer, msg.header.stamp.to_sec())
        self.mutex.release()
        # if self.verbose:
        #     print("img_buffer shape: ", np.shape(self.img_buffer))
        #     print("time_buffer shape: ", np.shape(self.time_buffer))

    def inference_image(self, img):
        result = inference_detector(self.model, img)
        features = self.model.features

        if len(result) == 2:
            bbox_result, segm_result = result
        else:
            bbox_result = result
            segm_result = None
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        mask_label = np.zeros(img.shape[:2], dtype=np.int32)
        if segms is not None:
            for i, (bbox, segm, label) in enumerate(zip(bboxes, segms, labels)):
                if self.class_names[label] in self.target_names:
                    target_index = self.target_names.index(self.class_names[label]) + 1
                    mask_label[mask] = target_index
        return bboxes, mask_label, features

    def get_nearest_img(self):
        i = 0
        for img in self.img_buffer:
            if self.infer_end_time < img.header.stamp.to_time() and i > 0:
                delta_prev = abs(self.img_buffer[i - 1].header.stamp.to_time() - self.infer_end_time)
                delta_next = abs(self.img_buffer[i].header.stamp.to_time() - self.infer_end_time)
                if delta_prev < delta_next:
                    return self.img_buffer[i - 1]
                else:
                    return self.img_buffer[i]
            i = i + 1
        return self.img_buffer[-1]

    def evalimage(self):
        # 기록된 시간과 가장 가까운 이미지 time 찾기
        ## 만약 없다면 (처음 이미지라면) 그냥 front() 사용
        # 가장 가까운 이미지 전 이미지는 전부 pop해서 버리기
        # if self.infer_end_time != 0:
        #     get_nearest_img()
        if len(self.img_buffer) == 0:
            return

        # 처음 이미지 pop해서 받아오기
        # self.mutex.acquire()
        # if self.verbose:
        #     print("befor pop img_buffer: ", np.shape(self.img_buffer))
        infer_img = self.img_buffer[-1]
        # print(np.shape(infer_img))
        infer_start_time = self.time_buffer[-1]  # secs.nsecs
        # self.img_buffer = np.delete(self.img_buffer, 0)
        # if self.verbose:
        #     print("after pop img_buffer: ", np.shape(self.img_buffer))
        self.img_buffer = np.empty((0, 480, 848, 3), np.uint8)
        self.time_buffer = np.empty((0,), float)
        # self.mutex.release()

        t0 = time.time()
        # 버퍼의 가장 처음 이미지 inference 수행
        bboxes, mask_label, feats = self.inference_image(infer_img)
        feat_0 = feats["neck.fpn_convs.0.conv"][0].detach().cpu().numpy()
        feat_1 = feats["neck.fpn_convs.1.conv"][0].detach().cpu().numpy()
        bboxes = bboxes.flatten()
        feat_0 = feat_0.flatten()
        feat_1 = feat_1.flatten()
        if self.verbose:
            print("shape of bboxes: ", np.shape(bboxes))
            print("shape of feature 0: ", np.shape(feat_0))
            print("shape of feature 1: ", np.shape(feat_1))
        # inference 수행 중인 이미지 시간 저장해두기
        infering_time = time.time() - t0
        self.infer_end_time = infer_start_time + infering_time
        if self.verbose:
            print("start time: ", infer_start_time)
            print("ing time: ", infering_time)
            print("end time: ", self.infer_end_time)

        # inference 끝나면, 저장해둔 이미지 시간 기록해서 msg로 캡슐화
        # publish 하기.
        t0 = time.time()
        output = Node()
        t0 = time.time()
        o_sec = math.trunc(infer_start_time)
        o_nsec = math.trunc((infer_start_time - o_sec) * 10000000)
        t0 = time.time()
        output.header.stamp = rospy.Time(o_sec, o_nsec)
        output.bbox.data = bboxes
        # output.feat0.data = feat_0
        # output.feat1.data = feat_1
        t0 = time.time()
        self.pub_result.publish(output)
        if self.verbose:
            # print(output.header)
            # print(np.shape(output.bbox))
            # print(np.shape(output.feat0))
            # print(np.shape(output.feat1))
            print("publish time: ", time.time() - t0)
        return


if __name__ == '__main__':
    # print("Torch version")
    # print(torch.__version__)
    # print("Python version")
    # print(sys.version)
    # print("Version info.")
    # print(sys.version_info)

    kt_ros_ = kt_ros()
    print("Ready to start!!!")
    time.sleep(0.2)
    while 1:
        try:
            kt_ros_.evalimage()
            kt_ros_.rate.sleep()
        except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt):
            sys.exit(0)