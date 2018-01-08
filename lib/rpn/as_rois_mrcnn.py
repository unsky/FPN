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

class As_rois_MergeRcnnLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._batch_rois = cfg.TEST.RPN_POST_NMS_TOP_N

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)
        top[2].reshape(1, 5)
        top[3].reshape(1, 5)
        top[4].reshape(1, 5)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        rois = bottom[0].data
        w = (rois[:,3]-rois[:,1])
        h = (rois[:,4]-rois[:,2])
        s = w * h
        k0 =4
        s[s<=0]=1e-6
        layer_indexs = np.floor(k0+np.log2(np.sqrt(s)/224))

        layer_indexs[layer_indexs<2]=2
        layer_indexs[layer_indexs>5]=5

        rois_all =[]

        for i in range(4):
            index = (layer_indexs == (i + 2))
            num_index = sum(index)
            if num_index == 0:
                rois_ = np.zeros((1*4, 5), dtype=rois.dtype)
            else:
                rois_ = rois[index, :]
            rois_all.append(rois_)


        rois = np.concatenate(rois_all,axis= 0)
        rois_p2 = rois_all[0]
        rois_p3 = rois_all[1]
        rois_p4 = rois_all[2]
        rois_p5 = rois_all[3]       

        top[0].reshape(*rois.shape)
        top[0].data[...] = rois
        
        top[1].reshape(*rois_p2.shape)
        top[1].data[...] = rois_p2
    
        top[2].reshape(*rois_p3.shape)
        top[2].data[...] = rois_p3

        top[3].reshape(*rois_p4.shape)
        top[3].data[...] = rois_p4
        
        top[4].reshape(*rois_p5.shape)
        top[4].data[...] = rois_p5
        
      

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



