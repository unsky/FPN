# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from utils.cython_bbox import bbox_overlaps

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    stride = cfg.TRAIN.IMAGE_STRIDE
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        if stride == 0:
            im_scale_factors.append(im_scale)
            processed_ims.append(im)
        else:
            # pad to product of stride
            im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
            im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
            im_channel = im.shape[2]
            padded_im = np.zeros((im_height, im_width, im_channel))
            padded_im[:im.shape[0], :im.shape[1], :] = im
            im_scale_factors.append(im_scale)
            processed_ims.append(padded_im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

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

def vis_rois_detection(im_array, detections):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib  
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import savefig  
    import random
    a =  [103.06 ,115.9 ,123.15]
    a = np.array(a)
    im = transform_inverse(im_array,a)
    plt.imshow(im)
  #  print class_names.shape
    for j in range(detections.shape[0]):

        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
     
        det =dets

        bbox = det[0:] 
 
        rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
    plt.show()
    name = np.mean(im)
    savefig ('vis/'+str(name)+'.png')
    plt.clf()
    plt.cla()

    plt. close(0)
def bbox_vote(dets_NMS, dets_all, thresh=0.5):
    dets_voted = np.zeros_like(dets_NMS)   # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
			np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
			np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert(len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)        # Weighted bounding boxes
        dets_voted[i][4] = det[4]                         # Keep the original score

        # Weighted scores (if enabled)
        if cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])

    return dets_voted

def im_detect(net, im, boxes=None,num_classes=21):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)



    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] 
        index= np.where(np.sum(boxes,axis=1)!=0)[0]
        boxes = boxes[index,:]
     
# / im_scales[0]
    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        scores = scores[index]

      #  print scores[0:10]
    
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
    
        box_deltas = box_deltas[index,:]
     

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            means = np.tile(
                    np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1)).ravel()
            stds = np.tile(
                    np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1)).ravel()
      #  Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas * stds + means
        

      #  print boxes.shape,box_deltas.shape
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        s = (blobs['data'].astype(np.float32, copy=False).shape[2],blobs['data'].astype(np.float32, copy=False).shape[3],blobs['data'].astype(np.float32, copy=False).shape[1])
 
        pred_boxes = clip_boxes(pred_boxes, s)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    vis = False
    if vis:
        vis_rois_detection(blobs['data'].astype(np.float32, copy=False),pred_boxes/ im_scales[0])
  

    return scores, pred_boxes/ im_scales[0]

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
  
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals,imdb.num_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        vis = False
        if  vis:
            imj =im
            name = 'output/bads/'+ str(i) + '.jpg'
            for jj in xrange(1, imdb.num_classes):  
                indsj = np.where(scores[:, jj] > thresh)[0]
                cls_scoresj = scores[indsj, jj]
                cls_boxesj = boxes[indsj, jj*4:(jj+1)*4]
                cls_detsj = np.hstack((cls_boxesj, cls_scoresj[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                keep = nms(cls_detsj, cfg.TEST.NMS)
                cls_detsj = cls_detsj[keep, :]
                detsj = cls_detsj
                for ii in xrange(np.minimum(10, detsj.shape[0])):
                    bboxj = detsj[ii, :4]
                    scorej = detsj[ii, -1]
                    if bboxj != []:
                        x1 = bboxj[0]
                        y1 = bboxj[3]
                        x2 = bboxj[2]
                        y2 = bboxj[1]
                        # if x1 < 0:
                        #     x1=0
                        # if y1> imj.shape[1]:
                        #     y1=imj.shape[1]-1
                        # if x2 > imj.shape[0]:
                        #     x2 = imj.shape[0]-1
                        # if y2 < 0:
                        #     y2 = 0
                        if scorej > 0.1:
                            cv2.rectangle(imj, (x1, y1), (x2,y2),(0,255,0), 4)
                            text = str(jj) + ": " + str(scorej)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(imj, text, (x1, y1),font , 1, (0,0,255), 4)
            cv2.imwrite(name, imj)
            #aaa
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            
            dets_NMSed = cls_dets[keep, :]
            cls_dets = bbox_vote(dets_NMSed, cls_dets)
           # print cls_scores
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
