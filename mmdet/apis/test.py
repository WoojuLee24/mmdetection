# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmdet.models.trackers.sort_tracker import Sort, associate_detections_to_trackers

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_test_fpn(model,
                        data_loader,
                        show=False,
                        out_dir=None,
                        show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    prev = dict()
    # mot_tracker = Sort(max_age=1,
    #                    min_hits=3,
    #                    iou_threshold=0.3)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            x1_tensor, x2_tensor, x3_tensor, x4_tensor, x5_tensor = model.module.fpn_features
            x1_tensor = x1_tensor.mean(axis=1, keepdims=True)
            x1_tensor_show = torch.cat([x1_tensor, x1_tensor, x1_tensor], dim=1)

            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                ## get results
                import numpy as np
                if isinstance(result[i], tuple):
                    bbox_result, segm_result = result[i]
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = result[i], None
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

                if show_score_thr > 0:
                    assert bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > show_score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                    if segms is not None:
                        segms = segms[inds, ...]

                mask = segms.astype(np.int32)

                x1_tensor = F.interpolate(x1_tensor, (h, w), mode='nearest')
                x1_tensor = torch.squeeze(x1_tensor, 1)
                x1_npy = x1_tensor.detach().cpu().numpy()

                pres = {"bboxes": bboxes, "mask": mask, "labels": labels, "x1_npy": x1_npy}

                # trackers = mot_tracker.update(bboxes)
                # matched = mot_tracker.matched
                if len(prev) > 0:
                    # get matched bboxes between prev and pres boxes
                    pres_prev_matched, unmatched_pres_dets, unmatched_prev_dets = \
                        associate_detections_to_trackers(pres['bboxes'], prev['bboxes'], iou_threshold=0.3)

                    pres_matched_mask = pres["mask"][pres_prev_matched[:, 0]]
                    pres_matched_feat = pres["x1_npy"] * pres_matched_mask

                    prev_matched_mask = prev["mask"][pres_prev_matched[:, 1]]
                    prev_matched_feat = prev["x1_npy"] * prev_matched_mask

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                        ### mask jpg save ###
                        import os
                        import matplotlib.pyplot as plt
                        folder, file = img_meta['ori_filename'].split("/")
                        folder_path = os.path.join(out_dir, folder)
                        if not os.path.exists(folder_path):
                            os.mkdir(folder_path)
                        for idx in range(len(pres_matched_feat)):
                            pres_jpg_file = file[:-4] + "_f{}_pres.jpg".format(idx)
                            pres_jpg_file_path = os.path.join(folder_path, pres_jpg_file)
                            prev_jpg_file = file[:-4] + "_f{}_prev.jpg".format(idx)
                            prev_jpg_file_path = os.path.join(folder_path, prev_jpg_file)
                            plt.imsave(prev_jpg_file_path, prev_matched_feat[idx])
                            plt.imsave(pres_jpg_file_path, pres_matched_feat[idx])
                    else:
                        out_file = None

                    ### save the mask prediction results ###
                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

                prev = pres

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
