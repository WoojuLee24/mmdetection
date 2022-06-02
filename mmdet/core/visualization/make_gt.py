import os
import numpy as np
from PIL import Image

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

# def scan_make_gt():
#
#
#
# def par
#
# def main():
#
#
# if __name__ == '__main__':
#     main()