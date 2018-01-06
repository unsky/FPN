# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._feat_stride = [int(i) for i in layer_params['feat_stride'].split(',')]
        self._scales = cfg.FPNRSCALES
        self._ratios = cfg.FPNRATIOS
        #anchor_scales = layer_params.get('scales', (8, ))
        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 1000)

     
        s = 0
        for i in range(5):
            height, width = bottom[i].data.shape[-2:]
            s = s + height * width
            
            i_anchors = generate_anchors(base_size=self._feat_stride[i], ratios=self._ratios, scales=self._scales)
            A = i_anchors.shape[0]

        # labels
        top[0].reshape(1, 1, A * s)
        # bbox_targets
        top[1].reshape(1, A * 4, s)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, s)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, s)

    def forward(self, bottom, top):
        

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        h = []
        w = []
        for i in range(5):
            height, width = bottom[i].data.shape[-2:]
            h.append(height)
            w.append(width)
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[5].data
        # im_info
        im_info = bottom[6].data[0, :]

        all_anchors_list = []
        inds_inside_list = []
        total_anchors = 0
        
        feat_strides = self._feat_stride
        ratios = self._ratios

        scales = self._scales

        fpn_args = []
        fpn_anchors_fid = np.zeros(0).astype(int)
        fpn_anchors = np.zeros([0, 4])
        fpn_labels = np.zeros(0)
        fpn_inds_inside = []
        for feat_id in range(len(feat_strides)):
            # len(scales.shape) == 1 just for backward compatibility, will remove in the future
        
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios, scales=scales)

            num_anchors = base_anchors.shape[0]
            feat_height = h[feat_id]
            feat_width = w[feat_id]

            # 1. generate proposals from bbox deltas and shifted anchors
            shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
            shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = num_anchors
            K = shifts.shape[0]
            all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            all_anchors = all_anchors.reshape((K * A, 4))
            total_anchors = int(K * A)

            # only keep anchors inside the image
            inds_inside = np.where((all_anchors[:, 0] >= -self._allowed_border) &
                                (all_anchors[:, 1] >= -self._allowed_border) &
                                (all_anchors[:, 2] < im_info[1] + self._allowed_border) &
                                (all_anchors[:, 3] < im_info[0] + self._allowed_border))[0]

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            # label: 1 is positive, 0 is negative, -1 is dont care
            # for sigmoid classifier, ignore the 'background' class
            labels = np.empty((len(inds_inside),), dtype=np.float32)
            labels.fill(-1)

            fpn_anchors_fid = np.hstack((fpn_anchors_fid, len(inds_inside)))
            fpn_anchors = np.vstack((fpn_anchors, anchors))
            fpn_labels = np.hstack((fpn_labels, labels))
            fpn_inds_inside.append(inds_inside)
            fpn_args.append([feat_height, feat_width, A, total_anchors])

        if gt_boxes.size > 0:
            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(fpn_anchors.astype(np.float), gt_boxes.astype(np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(fpn_anchors)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            # fg label: for each gt, anchor with highest overlap
            fpn_labels[gt_argmax_overlaps] = 1
            # fg label: above threshold IoU
            fpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        else:
            fpn_labels[:] = 0
        # subsample positive labels if we have too many
        num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(fpn_labels >= 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            if DEBUG:
                disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
            fpn_labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCHSIZE == -1 else cfg.TRAIN.RPN_BATCHSIZE - np.sum(fpn_labels >= 1)
        bg_inds = np.where(fpn_labels == 0)[0]
        fpn_anchors_fid = np.hstack((0, fpn_anchors_fid.cumsum()))

        # if balance_scale_bg:
        #     num_bg_scale = num_bg / len(feat_strides)
        #     for feat_id in range(0, len(feat_strides)):
        #         bg_ind_scale = bg_inds[(bg_inds >= fpn_anchors_fid[feat_id]) & (bg_inds < fpn_anchors_fid[feat_id+1])]
        #         if len(bg_ind_scale) > num_bg_scale:
        #             disable_inds = npr.choice(bg_ind_scale, size=(len(bg_ind_scale) - num_bg_scale), replace=False)
        #             fpn_labels[disable_inds] = -1
        # else:
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            if DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            fpn_labels[disable_inds] = -1

        fpn_bbox_targets = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
        if gt_boxes.size > 0:
            fpn_bbox_targets[fpn_labels >= 1, :] = bbox_transform(fpn_anchors[fpn_labels >= 1, :], gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
            # fpn_bbox_targets[:] = bbox_transform(fpn_anchors, gt_boxes[argmax_overlaps, :4])
        # fpn_bbox_targets = (fpn_bbox_targets - np.array(cfg.TRAIN.BBOX_MEANS)) / np.array(cfg.TRAIN.BBOX_STDS)
        fpn_bbox_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

        fpn_bbox_weights[fpn_labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
        fpn_bbox_outside_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(fpn_labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(fpn_labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(fpn_labels == 0))
        
        fpn_bbox_outside_weights[fpn_labels == 1, :] = positive_weights
        fpn_bbox_outside_weights[fpn_labels == 0, :] = negative_weights

        label_list = []
        bbox_target_list = []
        bbox_weight_list = []
        bbox_outside_weight_list = []
        for feat_id in range(0, len(feat_strides)):
            feat_height, feat_width, A, total_anchors = fpn_args[feat_id]
            # map up to original set of anchors
            labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=-1)
            bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
            bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
            bbox_outside_weights = _unmap(fpn_bbox_outside_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)

            labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
            labels = labels.reshape((1, A * feat_height * feat_width))

            bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
            bbox_targets = bbox_targets.reshape((1, A * 4, -1))
            bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
            bbox_weights = bbox_weights.reshape((1, A * 4, -1))

            bbox_outside_weights = bbox_outside_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
            bbox_outside_weights = bbox_outside_weights.reshape((1, A * 4, -1))

            label_list.append(labels)
            bbox_target_list.append(bbox_targets)
            bbox_weight_list.append(bbox_weights)
            bbox_outside_weight_list.append(bbox_outside_weights)
            # label.update({'label_p' + str(feat_id + feat_id_start): labels,
            #               'bbox_target_p' + str(feat_id + feat_id_start): bbox_targets,
            #               'bbox_weight_p' + str(feat_id + feat_id_start): bbox_weights})

    
        labels = np.concatenate(label_list, axis=1)
        bbox_targets = np.concatenate(bbox_target_list, axis=2)
        bbox_inside_weights =  np.concatenate(bbox_weight_list, axis=2)
        bbox_outside_weights = np.concatenate(bbox_outside_weight_list, axis=2)

        # print bbox_targets.shape
        # print bbox_inside_weights.shape
        # print bbox_outside_weights.shape
        # print labels.shape
       
        
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets

        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights

        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights

        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights



    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """" unmap a subset inds of data into original data of size count """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
