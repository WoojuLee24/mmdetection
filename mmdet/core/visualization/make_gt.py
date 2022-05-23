import os
import numpy as np
from PIL import Image

def make_gt(out_file=None,
             bboxes=None,
             segms=None,
             labels=None):
    path_bbox = os.path.join('/ws/external', 'outputs', 'cityscapes', 'bbox')
    path_mask = os.path.join('/ws/external', 'outputs', 'cityscapes', 'mask')
    path_label = os.path.join('/ws/external', 'outputs', 'cityscapes', 'label')

    if not(os.path.exists(path_bbox)):
        os.mkdir(path_bbox)
    if not(os.path.exists(path_mask)):
        os.mkdir(path_mask)
    if not(os.path.exists(path_label)):
        os.mkdir(path_label)

    name = out_file[72:78]
    # save numpy output
    np.save(os.path.join(path_bbox, 'bbox_%s' % name), bboxes)
    np.save(os.path.join(path_mask, 'mask_%s' % name), segms)
    np.save(os.path.join(path_label, 'label_%s' % name), labels)

    # make gtFine_instanceIds
    instance_id = np.zeros((1024, 2048), dtype=np.int32)
    for i, seg in enumerate(segms):
        temp = seg * 1
        seg_idx = np.where(temp == 1)
        if labels[i] == 0:
            id_ = 24001  # person
        elif labels[i] == 1:
            id_ = 25001  # rider
        elif labels[i] == 2:
            id_ = 26001  # car
        elif labels[i] == 3:
            id_ = 27001  # truck
        elif labels[i] == 4:
            id_ = 28001  # bus
        elif labels[i] == 5:
            id_ = 31001  # train
        elif labels[i] == 6:
            id_ = 32001  # motorcycle
        elif labels[i] == 7:
            id_ = 33001  # bicycle
        else:
            id_ = 0

        for j in range(len(seg_idx[0])):
            instance_id[seg_idx[0][j], seg_idx[1][j]] = id_

    instance_id = (instance_id).astype(np.int32)
    ins_out = Image.fromarray(instance_id, 'I')
    ins_name = out_file[:78] + '_gtFine_instanceIds.png'
    ins_out.save(ins_name)

    # test = Image.open(lab_name)
    # np_test = np.array(test)