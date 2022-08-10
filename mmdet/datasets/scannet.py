# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.utils import print_log

from .builder import DATASETS
from .coco import CocoDataset
from .cityscapes import CityscapesDataset
# from tools.dataset_converters.scannet.scannet import labels
from collections import namedtuple


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


@DATASETS.register_module()
class ScannetDataset(CityscapesDataset):

    # CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    #            'bicycle')
    CLASSES = tuple(label.name for label in labels if label.category=="object")

    def _filter_imgs(self, min_size=32):
        k = labels
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_in_cat
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
                bboxes_ignore, labels, masks, seg_map. \
                "masks" are already decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=img_info['segm_file'])

        return ann

    def results2txt(self, results, outfile_prefix):
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            list[str]: Result txt files which contains corresponding \
                instance segmentation images.
        """
        try:
            import cityscapesscripts.helpers.labels as CSLabels
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        result_files = []
        os.makedirs(outfile_prefix, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.data_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            # segm results
            if isinstance(segm_result, tuple):
                # Some detectors use different scores for bbox and mask,
                # like Mask Scoring R-CNN. Score of segm will be used instead
                # of bbox score.
                segms = mmcv.concat_list(segm_result[0])
                mask_score = segm_result[1]
            else:
                # use bbox score for mask score
                segms = mmcv.concat_list(segm_result)
                mask_score = [bbox[-1] for bbox in bboxes]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            prog_bar.update()
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = mask_score[i]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(outfile_prefix,
                                            basename + f'_{i}_{classes}.png')
                    mmcv.imwrite(mask, png_filename)
                    fout.write(f'{osp.basename(png_filename)} {class_id} '
                               f'{score}\n')
            result_files.append(pred_txt)

        return result_files

    def format_results(self, results, txtfile_prefix=None):
        """Format the results to txt (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving txt/png files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2txt(results, txtfile_prefix)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 outfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in Cityscapes/COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None): The prefix of output file. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with COCO protocol, it would be the
                prefix of output json file. For example, the metric is 'bbox'
                and 'segm', then json files would be "a/b/prefix.bbox.json" and
                "a/b/prefix.segm.json".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output txt/png files. The output files would be
                png images under folder "a/b/prefix/xxx/" and the file name of
                images would be written into a txt file
                "a/b/prefix/xxx_pred.txt", where "xxx" is the video name of
                cityscapes. If not specified, a temp file will be created.
                Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric or cityscapes mAP \
                and AP@50.
        """
        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]

        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, outfile_prefix, logger))
            metrics.remove('cityscapes')

        # left metrics are all coco metric
        if len(metrics) > 0:
            # create CocoDataset with CityscapesDataset annotation
            self_coco = CocoDataset(self.ann_file, self.pipeline.transforms,
                                    None, self.data_root, self.img_prefix,
                                    self.seg_prefix, self.proposal_file,
                                    self.test_mode, self.filter_empty_gt)
            # TODO: remove this in the future
            # reload annotations of correct class
            self_coco.CLASSES = self.CLASSES
            self_coco.data_infos = self_coco.load_annotations(self.ann_file)
            eval_results.update(
                self_coco.evaluate(results, metrics, logger, outfile_prefix,
                                   classwise, proposal_nums, iou_thrs))

        return eval_results

    def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of output txt file
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: Cityscapes evaluation results, contains 'mAP' \
                and 'AP@50'.
        """

        try:
            import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, txtfile_prefix)

        if tmp_dir is None:
            result_dir = osp.join(txtfile_prefix, 'results')
        else:
            result_dir = osp.join(tmp_dir.name, 'results')

        eval_results = OrderedDict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
        CSEval.args.predictionPath = os.path.abspath(result_dir)
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = os.path.join(result_dir,
                                                   'gtInstances.json')
        CSEval.args.groundTruthSearch = os.path.join(
            self.img_prefix.replace('leftImg8bit', 'gtFine'),
            '*/*_gtFine_instanceIds.png')

        groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
        assert len(groundTruthImgList), 'Cannot find ground truth images' \
            f' in {CSEval.args.groundTruthSearch}.'
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
