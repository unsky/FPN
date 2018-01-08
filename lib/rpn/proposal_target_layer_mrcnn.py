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
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import matplotlib  
matplotlib.use('Agg') 
DEBUG = False
def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
   # print im_array.shape
    import matplotlib  
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import savefig  
    import random
    a =  [103.06 ,115.9 ,123.15]
    a = np.array(a)
    im = transform_inverse(im_array,a)
    plt.imshow(im)
    for j in range(detections.shape[0]):
        # if class_names[j] == 0:
        #     continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        det =dets
        bbox = det[0:] 
        score = det[0]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        # plt.gca().text(bbox[0], bbox[1] - 2,
        #                    '{:s} {:.3f}'.format(str(class_names[j]), score),
        #                    bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
    name = np.mean(im)
    savefig ('vis/'+str(name)+'.png')
    plt.clf()
    plt.cla()

    plt. close(0)

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im




class ProposalMergeRcnnTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._batch_rois = cfg.TRAIN.BATCH_SIZE

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)
        top[2].reshape(1, 5)
        top[3].reshape(1, 5)
        # labels
        top[4].reshape(1, 1)
        # bbox_targets
        top[5].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[6].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[7].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        aaa = all_rois[:]
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        im = bottom[2].data
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        
        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        rois, labels, bbox_targets, bbox_weights ,layer_indexs = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes,sample_type='fpn', k0 = 4)
        vis =False
        if vis:
            ind = np.where(labels!=0)[0]
            im_shape = im.shape
            means = np.tile(
                     np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (21, 1)).ravel()
            stds = np.tile(
                    np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (21, 1)).ravel()
            bbox_targets = bbox_targets*stds +means
            
            pred_boxes = bbox_transform_inv(rois[:,1:], bbox_targets)
            pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            l =labels[ind]
            ro = rois[ind,1:]
            b = bbox_targets[ind,:]
            p = pred_boxes[ind,:]*bbox_weights[ind,:]
            r = []
            for i in range(p.shape[0]):
                r.append(p[i,l[i]*4:l[i]*4+4])
            r_ =  np.vstack(r)

      #  Optionally normalize targets by a precomputed mean and stdev

            vis_all_detection(im, aaa[:,1:], l, 1)

        
        labels_all = [] 
        bbox_targets_all = []
        bbox_weights_all = []
        rois_all =[]
        for i in range(4):
            index = (layer_indexs == (i + 2))
            num_index = sum(index)
            if num_index == 0:
                rois_ = np.zeros((1*4, 5), dtype=rois.dtype)
                labels_ = np.ones((1*4, ), dtype=labels.dtype)*-1
                bbox_targets_ = np.zeros((1*4, self._num_classes * 4), dtype=bbox_targets.dtype)
                bbox_weights_ = np.zeros((1*4, self._num_classes * 4), dtype=bbox_weights.dtype)
            else:
                rois_ = rois[index, :]
                labels_ = labels[index]
                bbox_weights_= bbox_weights[index, :]
                bbox_targets_ = bbox_targets[index, :]

            rois_all.append(rois_)
            labels_all.append(labels_)  
            bbox_targets_all.append(bbox_targets_)
            bbox_weights_all.append(bbox_weights_)


        rois_p2 = rois_all[0]
        rois_p3 = rois_all[1]
        rois_p4 = rois_all[2]
        rois_p5 = rois_all[3]    
        labels_all = np.concatenate(labels_all)
        bbox_targets_all = np.concatenate(bbox_targets_all,axis= 0)
        bbox_weights_all = np.concatenate(bbox_weights_all,axis= 0)
      #  print bbox_targets_all.shape,bbox_weights_all.shape, rois_p2.shape,rois_p3.shape,rois_p4.shape,rois_p5.shape,labels_all.shape

        top[0].reshape(*rois_p2.shape)
        top[0].data[...] = rois_p2
    
        top[1].reshape(*rois_p3.shape)
        top[1].data[...] = rois_p3

        top[2].reshape(*rois_p4.shape)
        top[2].data[...] = rois_p4
        
        top[3].reshape(*rois_p5.shape)
        top[3].data[...] = rois_p5
        
        # classification labels
        top[4].reshape(*labels_all.shape)
        top[4].data[...] = labels_all

        # bbox_targets
        top[5].reshape(*bbox_targets_all.shape)
        top[5].data[...] = bbox_targets_all

        # bbox_inside_weights
        top[6].reshape(*bbox_weights_all.shape)
        top[6].data[...] = bbox_weights_all

        # bbox_outside_weights
        top[7].reshape(*bbox_weights_all.shape)
        top[7].data[...] = np.array(bbox_weights_all > 0).astype(np.float32)
      

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

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes,sample_type='fpn', k0 = 4):
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

    if sample_type == 'fpn':
        #print 0
        w = (rois[:,3]-rois[:,1])
        h = (rois[:,4]-rois[:,2])
        s = w * h
        s[s<=0]=1e-6
        layer_index = np.floor(k0+np.log2(np.sqrt(s)/224))

        layer_index[layer_index<2]=2
        layer_index[layer_index>5]=5
        #print 1
        return rois, labels, bbox_targets, bbox_inside_weights, layer_index #rois:[512,5]   labels:[512,]
    else:
        return rois, labels, bbox_targets, bbox_inside_weights


