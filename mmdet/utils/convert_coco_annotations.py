import argparse
import os
import os.path as osp
import json
import pycocotools.mask as mask_util
import cv2

def main():
    # input_bbox_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.bbox.json"
    # input_segm_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.segm.json"
    # output_path = "/ws/data/OpenLORIS/annotations/preprocess_val.json"
    # modified_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_val.json"

    # input_bbox_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_train.bbox.json"
    # input_segm_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_train.segm.json"
    # output_path = "/ws/data/OpenLORIS/annotations/preprocess_train.json"
    # modified_path = "/ws/data/OpenLORIS/annotations/mask_rcnn_r50_fpn_2x_train.json"
    # input_bbox_path = "/ws/data/OpenLORIS/results/test.bbox.json"
    # input_segm_path = "/ws/data/OpenLORIS/results/test.segm.json"
    # output_path = "/ws/data/OpenLORIS/annotations/home1-5.json"
    # modified_path = "/ws/data/OpenLORIS/annotations/home1-5_mod.json"
    mode = "val"
    input_bbox_path = f"/ws/data/OpenLORIS/annotations/yolox_s_8x8_300e_openloris_{mode}.json.bbox.json"
    input_segm_path = None
    output_path = f"/ws/data/OpenLORIS/annotations/preprocess_{mode}.json"
    modified_path = f"/ws/data/OpenLORIS/annotations/yolox_s_8x8_300e_openloris_{mode}.json"
    with open(input_bbox_path, 'r') as in_bb_file:
        in_bb_data = json.load(in_bb_file)
    if input_segm_path != None:
        with open(input_segm_path, 'r') as in_segm_file:
            in_segm_data = json.load(in_segm_file)
        for test in in_segm_data:
            mask = mask_util.decode(test['segmentation'])
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours[0].flatten().tolist()) # [x1, y1, x2, y2, ...]
            polygon_l.append(contours[0].flatten().tolist())  # [x1, y1, x2, y2, ...]

    with open(output_path, 'r') as out_file:
        out_data = json.load(out_file)

    ann_id = 0
    for out_img in out_data['images']:
        # print(img['id'], img['file_name'])
        for idx, in_bb in enumerate(in_bb_data):
            if out_img['id'] == in_bb['image_id']:
                bx_min, by_min, bx_max, by_max = min(in_bb['bbox'][0], in_bb['bbox'][2]), \
                                                 min(in_bb['bbox'][1], in_bb['bbox'][3]), \
                                                 max(in_bb['bbox'][0], in_bb['bbox'][2]), \
                                                 max(in_bb['bbox'][1], in_bb['bbox'][3])
                if input_segm_path != None:
                    polygons = []
                    mask = mask_util.decode(in_segm_data[idx]['segmentation'])
                    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if contour.size >= 6:
                            contour = contour.flatten().tolist()
                            polygons.append(contour)

                ann = dict(
                    id=ann_id,
                    image_id=out_img['id'],
                    category_id=in_bb['category_id'],
                    # segmentation=polygons,
                    # area = (bx_max - bx_min) * (by_max - by_min),
                    # bbox = [bx_min, by_min, bx_max - bx_min, by_max - by_min],
                    area=in_bb['bbox'][2] * in_bb['bbox'][3],
                    bbox=in_bb['bbox'],
                    iscrowd=0
                )
                ann_id += 1
                out_data['annotations'].append(ann)
                print(ann_id)


    with open(modified_path, 'w', encoding='utf-8') as make_file:
        json.dump(out_data, make_file) # indent = '\t'

# annotation = {
#     "id": int,
#     "image_id": int,
#     "category_id": int,
#     "segmentation": RLE or [polygon],
#     "area": float,
#     "bbox": [x,y,width,height],
#     "iscrowd": 0 or 1,
# }


if __name__ == '__main__':
    main()