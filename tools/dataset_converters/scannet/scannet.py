# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import print_function, absolute_import, division
import argparse
import glob
import os.path as osp
from collections import namedtuple

import mmcv
import numpy as np
import pycocotools.mask as maskUtils


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!


## nyu40id, nyu40class
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unannotated'          ,  0 ,      255 , 'void'            , 0       , True         , False        , (  0,  0,  0) ),
    Label(  'wall'                 ,  1 ,      0 , 'structure'            , 0       , True         , False         , (  0,  0,  0) ),
    Label(  'floor'                ,  2 ,      1 , 'structure'            , 0       , True         , False        , (  0,  0,  0) ),
    Label(  'cabinet'              ,  3 ,      2 , 'object'            , 0       , True         , False         , (  0,  0,  0) ),
    Label(  'bed'                  ,  4 ,      3 , 'object'            , 0       , True         , False         , (  0,  0,  0) ),
    Label(  'chair'                ,  5 ,      4 , 'object'            , 0       , True         , False        , (111, 74,  0) ),
    Label(  'sofa'                 ,  6 ,      5 , 'object'            , 0       , True         , False        , ( 81,  0, 81) ),
    Label(  'table'                ,  7 ,      6 , 'object'            , 1       , True         , False      , (128, 64,128) ),
    Label(  'door'                 ,  8 ,      7 , 'structure'            , 1       , True         , False     , (244, 35,232) ),
    Label(  'window'               ,  9 ,      8 , 'structure'            , 1       , True         , False         , (250,170,160) ),
    Label(  'bookshelf'            , 10 ,      9 , 'object'            , 1       , True         , False         , (230,150,140) ),
    Label(  'picture'              , 11 ,      10 , 'object'    , 2       , True         , False        , ( 70, 70, 70) ),
    Label(  'counter'              , 12 ,      11 , 'structure'    , 2       , True         , False        , (102,102,156) ),
    Label(  'blinds'               , 13 ,       12 , 'structure'    , 2       , True         , False        , (190,153,153) ),
    Label(  'desk'                 , 14 ,      13 , 'object'    , 2       , True         , False         , (180,165,180) ),
    Label(  'shelves'              , 15 ,      14 , 'object'    , 2       , True         , False         , (150,100,100) ),
    Label(  'curtain'              , 16 ,      15 , 'object'    , 2       , True         , False         , (150,120, 90) ),
    Label(  'dresser'              , 17 ,      16 , 'object'          , 3       , True         , False        , (153,153,153) ),
    Label(  'pillow'               , 18 ,      17 , 'object'          , 3       , True         , False         , (153,153,153) ),
    Label(  'mirror'               , 19 ,      18 , 'object'          , 3       , True         , False        , (250,170, 30) ),
    Label(  'floor mat'            , 20 ,      19 , 'object'          , 3       , True         , False        , (220,220,  0) ),
    Label(  'clothes'              , 21 ,      20 , 'object'          , 4       , True         , False        , (107,142, 35) ),
    Label(  'ceiling'              , 22 ,      21 , 'structure'          , 4       , True         , False        , (152,251,152) ),
    Label(  'books'                , 23 ,      22 , 'object'             , 5       , True         , False        , ( 70,130,180) ),
    Label(  'refrigerator'         , 24 ,      23 , 'object'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'television'           , 25 ,      24 , 'object'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'paper'                , 26 ,      25 , 'object'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'towel'                , 27 ,      26 , 'object'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'shower curtain'       , 28 ,      27 , 'object'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'box'                  , 29 ,      28 , 'object'         , 7       , True         , False         , (  0,  0, 90) ),
    Label(  'whiteboard'           , 30 ,      29 , 'object'         , 7       , True         , False         , (  0,  0,110) ),
    Label(  'person'               , 31 ,      30 , 'object'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'night stand'          , 32 ,      31 , 'object'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'toilet'               , 33 ,      32 , 'structure'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'sink'                 , 34 ,      33 , 'structure'         , 7       , True         , False        ,  (119, 11, 80)),
    Label(  'lamp'                 , 35 ,      34 , 'object'         , 7       , True         , False        ,  (119, 11, 150)),
    Label(  'bathtub'              , 36 ,      35 , 'structure'         , 7       , True         , False        ,  (119, 50, 11)),
    Label(  'bag'                  , 37 ,      36 , 'object'         , 7       , True         , False        ,  (119, 50, 50)),
    Label(  'otherstructure'       , 38 ,      255 , 'structure'         , 7       , True         , True        ,  (119, 50, 90)),
    Label(  'otherfurniture'       , 39 ,      255 , 'structure'         , 7       , True         , True        ,  (119, 50, 150)),
    Label(  'otherprop'            , 40 ,      255 , 'structure'         , 7       , True         , True        ,  (119, 50, 180)),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


def collect_files(img_dir, gt_dir):
    suffix = 'leftImg8bit.png'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '**/*.png')):
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_instanceIds.png'
        # Note that labelIds are not converted to trainId for seg map
        segm_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    # deubg
    inst_debug = mmcv.imread("/ws/data/scannet/debug/_000000_gtFine_instanceIds.png", 'unchanged')
    unique_inst_ids =np.unique(inst_debug)
    inst_img = mmcv.imread(inst_file, 'unchanged')
    unique_inst_ids = np.unique(inst_img)
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    # unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        label = id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        category_id = label.id
        iscrowd = int(inst_id < 1000)
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name, osp.basename(segm_file)))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    parser.add_argument('--ordered', action='store_true', help='ordered file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    set_name = dict(
        train='instancesonly_filtered_gtFine_train.json',
        val='instancesonly_filtered_gtFine_val.json',
        test='instancesonly_filtered_gtFine_test.json')

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It took {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split))
            if args.ordered:
                files = sorted(files)
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
