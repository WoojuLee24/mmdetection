import argparse
import os
import numpy as np
from PIL import Image
import pandas as pd
import time
import shutil

def make_gt(out_file=None,
            bboxes=None,
            segms=None,
            labels=None,
            save_npy=False,
            ):

    name = out_file[:-15]

    if save_npy:
        npy_name = out_file[72:78]  # stuttgart
        path_bbox = os.path.join('/ws/external', 'outputs', 'cityscapes', 'bbox')
        path_mask = os.path.join('/ws/external', 'outputs', 'cityscapes', 'mask')
        path_label = os.path.join('/ws/external', 'outputs', 'cityscapes', 'label')

        if not(os.path.exists(path_bbox)):
            os.mkdir(path_bbox)
        if not(os.path.exists(path_mask)):
            os.mkdir(path_mask)
        if not(os.path.exists(path_label)):
            os.mkdir(path_label)

        # save numpy output
        np.save(os.path.join(path_bbox, 'bbox_%s' % npy_name), bboxes)
        np.save(os.path.join(path_mask, 'mask_%s' % npy_name), segms)
        np.save(os.path.join(path_label, 'label_%s' % npy_name), labels)

    # make gtFine_instanceIds
    instance_id = np.zeros((1024, 2048), dtype=np.int32)
    # id_count edited
    id_count = {'person': 0, 'rider': 0, 'car': 0, 'truck': 0, 'bus': 0, 'train': 0, 'motorcycle': 0, 'bicycle': 0}
    for i, seg in enumerate(segms):
        temp = seg * 1
        seg_idx = np.where(temp == 1)
        id_ = 0
        if labels[i] == 0:
            id_count['person'] += 1
            id_ = 24000 + id_count['person']  # # 93 for png
        elif labels[i] == 1:
            id_count['rider'] += 1
            id_ = 25000 + id_count['rider']
        elif labels[i] == 2:
            id_count['car'] += 1
            id_ = 26000 + id_count['car']
        elif labels[i] == 3:
            id_count['truck'] += 1
            id_ = 27000 + id_count['truck']
        elif labels[i] == 4:
            id_count['bus'] += 1
            id_ = 28000 + id_count['bus']
        elif labels[i] == 5:
            id_count['train'] += 1
            id_ = 31000 + id_count['train']
        elif labels[i] == 6:
            id_count['motorcycle'] += 1
            id_ = 32000 + id_count['motorcycle']
        elif labels[i] == 7:
            id_count['bicycle'] += 1
            id_ = 33000 + id_count['bicycle']
        else:
            id_ = 0

        for j in range(len(seg_idx[0])):
            instance_id[seg_idx[0][j], seg_idx[1][j]] = id_

    instance_id = instance_id.astype(np.int32)
    # np_elem = np.unique(instance_id.flatten())
    lab = Image.fromarray(instance_id, 'I')
    instance_name = name + 'gtFine_instanceIds.png'
    color_name = name + 'gtFine_color.png'
    label_name = name + 'gtFine_labelIds.png'

    lab.save(instance_name)
    lab.save(color_name)
    lab.save(label_name)



def get_scannet_label_table(path):
    file = path + "scannetv2-labels.combined.tsv"
    table = pd.read_csv(file, delimiter='\t', keep_default_na=False)
    table = table.to_numpy()
    table = sorted(table, key=lambda table: table[0])

    ## raw label to nyu40id format label
    ## https://github.com/facebookresearch/votenet/blob/main/scannet/meta_data/scannetv2-labels.combined.tsv
    ## http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt

    nyu40id = {0: 0}
    nyu40class = {0: 'unannotated'}
    for i, row in enumerate(table):
        nyu40class[table[i][0]] = table[i][7]   # class name
        nyu40id[table[i][0]] = table[i][4]      # id
    return nyu40class, nyu40id

def parse_args():
    parser = argparse.ArgumentParser(description='ScanNet dataset to cityscapes format')
    parser.add_argument("--dataset", help="input dataset dir path", type=str, default="/ws/data/scannet/scans/")
    parser.add_argument("--scene", help="scene dir name", type=str, default="scene0000_00")
    parser.add_argument("--filtered", help="true: smoothing data, false: raw data", type=bool, default=True)
    parser.add_argument("--gtLabel", help="scannetv2-labels.combined.tsv path", type=str, default="/ws/data/scannet/")
    parser.add_argument("--outrgb", help="output rgb directory path", type=str, default="/ws/data/scannet/leftImg8bit/train/")
    parser.add_argument("--outdir", help="output directory path", type=str, default="/ws/data/scannet/gtFine/train/")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    arg_dataset = args.dataset
    arg_scene = args.scene
    arg_gt_label = args.gtLabel
    arg_rgb_frame = args.outrgb
    arg_outdir = args.outdir
    filtered = args.filtered

    path_data = os.path.join(arg_dataset, arg_scene)
    if not (os.path.exists(arg_outdir)): os.mkdir(arg_outdir)
    path_out = os.path.join(arg_outdir, arg_scene)
    if not (os.path.exists(path_out)):   os.mkdir(path_out)
    rgb_out = os.path.join(arg_rgb_frame, arg_scene)
    if not (os.path.exists(rgb_out)):   os.mkdir(rgb_out)

    nyu40class, nyu40id = get_scannet_label_table(arg_gt_label)

    ##### Show 40 class text.
    # print(nyu40class)

    if filtered:
        dir_label = args.scene + "_2d-label-filt/label-filt/"
        dir_instance = args.scene + "_2d-instance-filt/instance-filt/"
    else:
        dir_label = args.scene + "2d-label/label"
        dir_instance = args.scene + "2d-instance/instance"

    path_rgb = os.path.join(path_data, "color")
    path_label = os.path.join(path_data, dir_label)
    path_instance = os.path.join(path_data, dir_instance)

    num_data = len(os.listdir(path_label))
    for frame in range(num_data):

        start = time.time() # start time checker

        ##### make name & pass the existed file
        rgb_name = rgb_out + '/' + arg_scene + '_{:0>6}_leftImg8bit.png'.format(frame)
        instance_name = path_out + '/' + arg_scene + '_{:0>6}_gtFine_instanceIds.png'.format(frame)
        color_name = path_out + '/' + arg_scene + '_{:0>6}_gtFine_color.png'.format(frame)
        label_name = path_out + '/' + arg_scene + '_{:0>6}_gtFine_labelIds.png'.format(frame)

        if os.path.isfile(rgb_name):
            print("Pass {:0>6} rgb frame".format(frame))
        else:
            file_rgb = os.path.join(path_rgb, str(frame) + ".jpg")
            shutil.copy(file_rgb, rgb_name)

        if os.path.isfile(instance_name) and os.path.isfile(color_name) and os.path.isfile(label_name):
            print("Pass {:0>6} frame".format(frame))
            continue

        img_label = Image.open(path_label + '{}.png'.format(frame))       # mode=I
        img_instance = Image.open(path_instance + '{}.png'.format(frame)) # mode=L
        img_instance = img_instance.convert("I")
        np_label = np.array(img_label)                                    # 968 * 1296, dtype int32
        np_instance = np.array(img_instance)                              # 968 * 1296, dtype uint8
        np_instance.astype(np.int32)

        h, w = np_label.shape[0], np_label.shape[1]                       # height, width

        ##### convert format scanNet to cityscapes. label format is nyu40id
        # output = np.zeros((h, w), dtype=np.int32)
        # for i, row in enumerate(output):
        #     if not any(np_label[i]):
        #         continue
        #     for j, column in enumerate(row):
        #         output[i][j] = nyu40id[np_label[i][j]] * 1000 + np_instance[i][j]

        ## improve inference speed
        np_mapped_label = np.vectorize(nyu40id.get)(np_label)
        output = np_mapped_label * 1000 + np_instance

        ##### save gtFine
        output = output.astype(np.int32)
        img_output = Image.fromarray(output, 'I')

        img_output.save(instance_name)
        img_output.save(color_name)
        img_output.save(label_name)

        print("save {:0>6} frame".format(frame))
        end = time.time()
        print(f"{end - start:.5f} sec")

        ##### check image overflow for debugging ################################
        img_debug = Image.open(instance_name)
        pixels = np.array(img_debug)
        if 65535 in np.unique(pixels):
            print('Failed')
            print('origin pixel: ', np.unique(output))
            print('reopen pixel: ', np.unique(pixels))
            print('origin instance: ', np.unique(np_instance))


if __name__ == '__main__':
    main()