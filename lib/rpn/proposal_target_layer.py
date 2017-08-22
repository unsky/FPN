# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)
        top[2].reshape(1, 5)
        # labels
        top[3].reshape(1, 1)
        # bbox_targets
        top[4].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[5].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[6].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        p2_rois = bottom[0].data
        p3_rois = bottom[1].data
        p4_rois = bottom[2].data
        p5_rois = bottom[3].data

        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[4].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)



        ################################################
        p2_rois = np.vstack(
            (p2_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        p3_rois = np.vstack(
            (p3_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )       
        p4_rois = np.vstack(
            (p4_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        p5_rois = np.vstack(
            (p5_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        #################################################
        assert np.all(p2_rois[:, 0] == 0), \
                'Only single item batches are supported'
        assert np.all(p3_rois[:, 0] == 0), \
                'Only single item batches are supported'
        assert np.all(p4_rois[:, 0] == 0), \
                'Only single item batches are supported'
        assert np.all(p5_rois[:, 0] == 0), \
                'Only single item batches are supported'
        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        ####################################################
        labels_p2, rois_p2, bbox_targets_p2, bbox_inside_weights_p2 = _sample_rois(
            p2_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        labels_p3, rois_p3, bbox_targets_p3, bbox_inside_weights_p3 = _sample_rois(
            p3_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        labels_p4, rois_p4, bbox_targets_p4, bbox_inside_weights_p4 = _sample_rois(
            p4_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        labels_p5, rois_p5, bbox_targets_p5, bbox_inside_weights_p5 = _sample_rois(
            p5_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        ####################################################

        
    


        ####################################################
        labels = []
        labels = labels_p2.tolist() + labels_p3.tolist() + labels_p4.tolist() + labels_p5.tolist()
        labels = np.array(labels)
        bbox_targets = np.vstack((bbox_targets_p2,bbox_targets_p3,bbox_targets_p4,bbox_targets_p5))

        bbox_inside_weights = np.vstack((bbox_inside_weights_p2,bbox_inside_weights_p3,bbox_inside_weights_p4,bbox_inside_weights_p5))
        outside_weight_p2 = np.array(bbox_inside_weights_p2 > 0).astype(np.float32)
        outside_weight_p3 = np.array(bbox_inside_weights_p3 > 0).astype(np.float32)
        outside_weight_p4 = np.array(bbox_inside_weights_p4 > 0).astype(np.float32)     
        outside_weight_p5 = np.array(bbox_inside_weights_p5 > 0).astype(np.float32)        
        bbox_outside_weights = np.vstack((outside_weight_p2,outside_weight_p3,outside_weight_p4, outside_weight_p5))
        ######################################################
        # sampled rois
        # print labels
        # print len(labels),len(bbox_targets),len(bbox_inside_weights),len(bbox_outside_weights)
        # print len(rois_p2),len(rois_p3),len(rois_p4),len(rois_p5)
        # print len(labels_p2),len(labels_p3),len(labels_p4),len(labels_p5)
        # print len(bbox_inside_weights_p2),len(bbox_inside_weights_p3),len(bbox_inside_weights_p4),len(bbox_inside_weights_p5)
        # print len(outside_weight_p2),len(outside_weight_p3),len(outside_weight_p4),len(outside_weight_p5)
        # print len(labels[np.where(labels > 0)]),len(labels_p2[np.where(labels_p2 > 0)]),len(labels_p3[np.where(labels_p3 > 0)]),len(labels_p4[np.where(labels_p4 > 0)]),len(labels_p5[np.where(labels_p5 > 0)])
   
        top[0].reshape(*rois_p2.shape)
        top[0].data[...] = rois_p2
    
        top[1].reshape(*rois_p3.shape)
        top[1].data[...] = rois_p3

        top[2].reshape(*rois_p4.shape)
        top[2].data[...] = rois_p4
        
        top[3].reshape(*rois_p5.shape)
        top[3].data[...] = rois_p5
        # classification labels
        top[4].reshape(*labels.shape)
        top[4].data[...] = labels

        # bbox_targets
        top[5].reshape(*bbox_targets.shape)
        top[5].data[...] = bbox_targets

        # bbox_inside_weights
        top[6].reshape(*bbox_inside_weights.shape)
        top[6].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[7].reshape(*bbox_outside_weights.shape)
        top[7].data[...] = bbox_outside_weights
      

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
