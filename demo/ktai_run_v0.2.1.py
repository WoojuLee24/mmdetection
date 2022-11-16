import argparse

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

import mmcv
from mmdet.apis import inference_detector, init_detector_with_feature

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
import random


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    # parser.add_argument('--config', default='../configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py', help='Config file')
    # parser.add_argument('--checkpoint', default='../weights/cascade_mask_rcnn_swin_small_patch4_window7.pth', help='Checkpoint file')

    parser.add_argument('--config', default='../configs/aioneteam/faster_rcnn_r50_fpn_1x_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='/media/ktaioneteam/DISK/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth', help='Checkpoint file')
    parser.add_argument('--visualize', type=bool, default=False, help='visualization option')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--mp4', type=str, default='/media/ktaioneteam/DISK/video/20210729_sq01_1280x720.mp4', help='mp4 source file')
    parser.add_argument('--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument('--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def interpolate_features(features, size, mode='nearest'):
    features_resized = dict()
    for key, feature in features.items():
        feature_resized = F.interpolate(feature[0], size, mode=mode)
        features_resized[key] = feature_resized[0]
    return features_resized

def main():
    args = parse_args()

    device = torch.device(args.device)

    np.random.seed(42)
    mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(200)]

    model = init_detector_with_feature(args.config, args.checkpoint, device=device)

    # camera = cv2.VideoCapture(args.camera_id)
    camera = cv2.VideoCapture(args.mp4)
    camera.set(cv2.CAP_PROP_POS_FRAMES, 100)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter('/media/ktaioneteam/DISK/video/output.mp4', fourcc, 18, (1920, 1080))

    # cv2.namedWindow("result", cv2.WINDOW_GUI_EXPANDED)

    # print('Press "Esc", "q" or "Q" to exit.')

    # SETTINGS
    bbox_color = (72, 101, 241)
    text_color = (72, 101, 241)
    score_thr = 0.35
    class_names = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    target_names = ['person', 'chair', 'couch', 'table', 'cell phone']
    frame = 0
    while True:
        t0 = time.time()
        model.features.clear()
        ret_val, img = camera.read()
        if not ret_val:
            break
        result = inference_detector(model, img)
        t1 = time.time()
        print('inference time: ', t1-t0)
        features = model.features
        H, W, C = np.shape(img)
        # features = interpolate_features(features, size=(H, W))
        t2 = time.time()
        print('interpolation time: ', t2-t1)
        # features['neck.fpn_convs.0.conv'] = features['neck.fpn_convs.0.conv'][0][0]
        # feature debug
        feature_points = dict()
        for key, feature in features.items():
            feature = feature[0][0]
            inds = ((feature - feature.mean()) / feature.std() >= 2.56).nonzero()
            feature_selected = feature[inds[:, 0], inds[:, 1], inds[:, 2]]
            feature_selected = feature_selected.unsqueeze(dim=1)
            feature_point = torch.cat([inds, feature_selected], dim=1)
            feature_points[key] = feature_point
        t3 = time.time()
        print(f"feature processing time: {t3 - t2: .3f}s")
        feature_npy = features['neck.fpn_convs.0.conv'][0][0].cpu().numpy()
        t4 = time.time()
        print(f"npy conversion time: {t4 - t3: .3f}s")

        bbox_result = result
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        print(f"(total time: {t3 - t0: .3f}s)")

        # # draw segmentation masks
        segms = None
        # if segm_result is not None and len(labels) > 0:  # non empty
        #     segms = mmcv.concat_list(segm_result)
        #     if isinstance(segms[0], torch.Tensor):
        #         segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        #     else:
        #         segms = np.stack(segms, axis=0)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
        if args.visualize:
            feature_npy = feature_npy.mean(axis=0)
            feature_cv2 = cv2.normalize(feature_npy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if segms is not None:
                mask_label = np.zeros(img.shape[:2], dtype=np.int32)
                for i, (bbox, segm, label) in enumerate(zip(bboxes, segms, labels)):
                    if class_names[label] in target_names:
                        # color_mask = mask_colors[label]
                        mask = segm.astype(bool)
                        # img[mask] = img[mask] * 0.5 + color_mask * 0.5
                        mask_label[mask] = label
            else:
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    if class_names[label] in target_names:
                        bbox_int = bbox.astype(np.int32)
                        bbox_locs = [(bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3])]

                        # img bbox
                        img = cv2.rectangle(img, bbox_locs[0], bbox_locs[1], bbox_color, 2)

                        label_text = class_names[label] if class_names is not None else f'class {label}'
                        if len(bbox) > 4:
                            label_text += f'|{bbox[-1]:.02f}'
                        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color)

                        # feature bbox
                        feature_npy = cv2.rectangle(feature_cv2, bbox_locs[0], bbox_locs[1], bbox_color, 2)

                        label_text = class_names[label] if class_names is not None else f'class {label}'
                        if len(bbox) > 4:
                            label_text += f'|{bbox[-1]:.02f}'
                        cv2.putText(feature_npy, label_text, (bbox_int[0], bbox_int[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                    text_color)

        # print(f"(total time: {time.time() - t0: .3f}s)")
        cv2.imwrite(f'/ws/data/OpenLORIS/debug/result_{frame}.png', img)
        plt.imsave(f'/ws/data/OpenLORIS/debug/feature_{frame}.png', feature_npy)
        frame += 1
        # cv2.imshow("result", img)
        # video_writer.write(img)
        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break

        # model.show_result(img, result, score_thr=args.score_thr, wait_time=1, show=True)

    # video_writer.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    main()