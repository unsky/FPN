# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import numpy.random as npr

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        
        self._feat_stride = [int(i) for i in layer_params['feat_stride'].split(',')]
        self._scales = 2 ** np.arange(3,4)
        self._ratios = [0.5,1,2]
        self._min_sizes = [4,8,16,32,64]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores_list = bottom[0].data
        
        bbox_deltas_list = bottom[1].data
        im_info = bottom[2].data[0, :]


        p2_shape =  bottom[3].data.shape
        p3_shape =  bottom[4].data.shape
        p4_shape =  bottom[5].data.shape
        p5_shape =  bottom[6].data.shape
        p6_shape =  bottom[7].data.shape
        feat_shape = []
        feat_shape.append(p2_shape)
        feat_shape.append(p3_shape)
        feat_shape.append(p4_shape)
        feat_shape.append(p5_shape)
        feat_shape.append(p6_shape)  
        
        num_feat = len(feat_shape)#[1,5,4]
        score_index_start=0
        bbox_index_start=0
        keep_proposal = []
        keep_scores = []
    #########################

        for i in range(num_feat):
            feat_stride = int(self._feat_stride[i])#4,8,16,32,64
            #print 'feat_stride:', feat_stride
            anchor = generate_anchors(base_size=feat_stride, ratios=self._ratios, scales=self._scales)
            num_anchors = anchor.shape[0]#3
            height = feat_shape[i][2]
            width = feat_shape[i][3]
            shift_x = np.arange(0, width) * feat_stride
            shift_y = np.arange(0, height) * feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
            A = num_anchors#3
            K = shifts.shape[0]#height*width
            anchors = anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))#3*height*widht,4
            scores = (scores_list[0,int(score_index_start):int(score_index_start+K*A*2)]).reshape((1,int(2*num_anchors),-1,int(width)))#1,2*3,h,w
            scores = scores[:,num_anchors:,:,:]#1,3,h,w
            bbox_deltas = (bbox_deltas_list[0,int(bbox_index_start):int(bbox_index_start+K*A*4)]).reshape((1,int(4*num_anchors),-1,int(width)))#1,4*3,h,w
            score_index_start += K*A*2
            bbox_index_start += K*A*4
            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))#[1,h,w,12]--->[1*h*w*3,4]
            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))#[1,h,w,3]--->[1*h*w*3,1]
            proposals =  bbox_transform_inv(anchors, bbox_deltas)#debug here, corresponding?
            proposals = clip_boxes(proposals, im_info[:2])
            keep = _filter_boxes(proposals, self._min_sizes[i] * im_info[2])
            keep_proposal.append(proposals[keep, :])
            keep_scores.append(scores[keep])
        proposals = keep_proposal[0]
        scores = keep_scores[0]
        for i in range(1,num_feat):
            proposals=np.vstack((proposals, keep_proposal[i]))
            scores=np.vstack((scores, keep_scores[i]))
        #print 'roi concate t_1 spends :{:.4f}s'.format(time.time()-t_1)
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        #t_2 = time.time()
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        #print 'roi concate t_2_1_1 spends :{:.4f}s'.format(time.time()-t_2)
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        #t_nms = time.time()
        det = np.hstack((proposals, scores)).astype(np.float32)
        keep = nms(det,nms_thresh)
        #print 'roi concate nms spends :{:.4f}s'.format(time.time()-t_nms)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            try:
                pad = npr.choice(keep, size=post_nms_topN - len(keep))
            except:
                proposals = np.zeros((post_nms_topN, 4), dtype=np.float32)
                proposals[:,2] = 16
                proposals[:,3] = 16
                batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
                blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
                top[0].reshape(*(blob.shape))
                top[0].data[...] = blob
                return
            keep = np.hstack((keep, pad))

        proposals = proposals[keep, :]
        scores = scores[keep]
        #print 'roi concate t_2 spends :{:.4f}s'.format(time.time()-t_2)
        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    

        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """ Remove all boxes with any side smaller than min_size """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor
