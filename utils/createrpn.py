import tensorflow as tf
import os
from Lib.TensorBase.tensorbase.base import Layers
from Lib.loss_functions import rpn_cls_loss, rpn_bbox_loss
from Lib.faster_rcnn_config import cfg
from Lib.rpn_softmax import rpn_softmax
from Networks.anchor_target_layer import anchor_target_layer
from Networks.proposal_layer import proposal_layer
import numpy as np
from Lib.test_aux import test_net

class createrpn():
    def __init__(self, feature_map,gt_boxe,key):
        self.featuremap=feature_map
        self.gt_boxes = gt_boxe
        self.im_dims = [1024,2048]
        self._feat_stride = 32
        self.lr = 0.001
        self.step = 0
        self.de_steps=25000
        self.key = key
        self.eval_mode = True if (key == 'EVAL') else False
        self._network()
        #self._optimizer()

    def _network(self):
        """ Define the network outputs """
        # Initialize network dicts
        #self.cnn = {}
        self.rpn_net = {}
        self.roi_proposal_net = {}
        self.fast_rcnn_net = {}

        # Train network
        if self.eval_mode == False:
            with tf.variable_scope('model'):
                self._faster_rcnn(self.featuremap, self.gt_boxes, self.im_dims)
        else:
            with tf.variable_scope('model'):
                #assert tf.get_variable_scope().reuse is True
                self._faster_rcnn(self.featuremap, None, self.im_dims)

    def _faster_rcnn(self, featuremap, gt_boxes, im_dims):
        # VALID and TEST are both evaluation mode

        # Region Proposal Network (RPN)
        self.rpn_net[self.key]= rpn(featuremap, gt_boxes, im_dims, self._feat_stride, self.eval_mode)

        # RoI Proposals
        self.roi_proposal_net[self.key] = roi_proposal(self.rpn_net[self.key], gt_boxes, im_dims, self.eval_mode)


    def _optimizer(self):
        """ Define losses and initialize optimizer """
        with tf.variable_scope("losses"):
            # Losses (come from TRAIN networks)
            self.rpn_cls_loss = self.rpn_net['TRAIN'].get_rpn_cls_loss()
            self.rpn_bbox_loss = self.rpn_net['TRAIN'].get_rpn_bbox_loss()


            # Total Loss
            self.cost = tf.reduce_sum(
                self.rpn_cls_loss + self.rpn_bbox_loss )

        # Optimizer arguments
        decay_steps = cfg.TRAIN.LEARNING_RATE_DECAY_RATE * self.de_steps  # Number of Epochs x images/epoch
        learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.step,
                                                   decay_steps=decay_steps, decay_rate=cfg.TRAIN.LEARNING_RATE_DECAY,
                                                   staircase=True)
        # Optimizer: ADAM
        opt =tf.train.AdamOptimizer(learning_rate=learning_rate)
        output_vars =[v for v in tf.trainable_variables() if ('rpn'  in v.name ) ]
        self.optimizer = opt.minimize(self.cost, var_list=output_vars)

    def _read_names(self, names_file):
        ''' Read the names.txt file and return a list of all bags '''
        with open(names_file) as f:
            names = f.read().splitlines()
        return names

    def _run_train_iter(self, feed_dict,sess):
        """ Run training iteration"""
        _ = sess.run([ self.optimizer], feed_dict=feed_dict)

    def createbbox(self,fname):
        fname.tolist()
        filepath, tempfilename = os.path.split(fname[0])
        tempname=tempfilename[0:-4].decode('utf-8')
        annotation_file = '/home/qmy/label/txt/'+ tempname+'.txt'
        gt_bbox = np.loadtxt(annotation_file, ndmin=2)
        return gt_bbox


    def evaluate(self,sess,img, raw_output_up,preds_summary,test=True):
        """ Evaluate network on the validation set. """
        key = 'TEST' if test is True else 'VALID'

        tf_inputs = (img)
        tf_outputs = (self.roi_proposal_net['EVAL'].get_rois(),
                      self.fast_rcnn_net['EVAL'].get_bbox_refinement(),
                      raw_output_up,
                      preds_summary)

        class_metrics = test_net(self.flags['data_directory'], self.names, sess, tf_inputs, tf_outputs, key=key, thresh=0.5, vis=True)

class rpn:
    '''
    Region Proposal Network (RPN): From the convolutional feature maps
    (TensorBase Layers object) of the last layer, generate bounding boxes
    relative to anchor boxes and give an "objectness" score to each

    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, featureMaps, gt_boxes, im_dims, _feat_stride, eval_mode):
        self.featureMaps = featureMaps
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self._feat_stride = _feat_stride
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES
        self.eval_mode = eval_mode
        self._network()

    def _network(self):
        # There shouldn't be any gt_boxes if in evaluation mode
        if self.eval_mode is True:
            assert self.gt_boxes is None, \
                'Evaluation mode should not have ground truth boxes (or else what are you detecting for?)'

        _num_anchors = len(self.anchor_scales) * 1

        rpn_layers = Layers(self.featureMaps)

        with tf.variable_scope('rpn'):
            # Spatial windowing
            for i in range(len(cfg.RPN_OUTPUT_CHANNELS)):
                rpn_layers.conv2d(filter_size=cfg.RPN_FILTER_SIZES[i], output_channels=cfg.RPN_OUTPUT_CHANNELS[i])

            features = rpn_layers.get_output()

            with tf.variable_scope('cls'):
                # Box-classification layer (objectness)
                self.rpn_bbox_cls_layers = Layers(features)
                self.rpn_bbox_cls_layers.conv2d(filter_size=1, output_channels=_num_anchors * 2, activation_fn=None)

            with tf.variable_scope('target'):
                # Only calculate targets in train mode. No ground truth boxes in evaluation mode
                if self.eval_mode is False:
                    print(anchor_target_layer)
                    # Anchor Target Layer (anchors and deltas)
                    rpn_cls_score = self.rpn_bbox_cls_layers.get_output()
                    self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                        anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                                            _feat_stride=self._feat_stride, anchor_scales=self.anchor_scales)

            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                self.rpn_bbox_pred_layers = Layers(features)
                self.rpn_bbox_pred_layers.conv2d(filter_size=1, output_channels=_num_anchors * 4, activation_fn=None)

    # Get functions
    def get_rpn_cls_score(self):
        return self.rpn_bbox_cls_layers.get_output()

    def get_rpn_labels(self):
        assert self.eval_mode is False, 'No RPN labels without ground truth boxes'
        return self.rpn_labels

    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred_layers.get_output()

    def get_rpn_bbox_targets(self):
        assert self.eval_mode is False, 'No RPN bounding box targets without ground truth boxes'
        return self.rpn_bbox_targets

    def get_rpn_bbox_inside_weights(self):
        assert self.eval_mode is False, 'No RPN inside weights without ground truth boxes'
        return self.rpn_bbox_inside_weights

    def get_rpn_bbox_outside_weights(self):
        assert self.eval_mode is False, 'No RPN outside weights without ground truth boxes'
        return self.rpn_bbox_outside_weights

    # Loss functions
    def get_rpn_cls_loss(self):
        assert self.eval_mode is False, 'No RPN cls loss without ground truth boxes'
        rpn_cls_score = self.get_rpn_cls_score()
        rpn_labels = self.get_rpn_labels()
        return rpn_cls_loss(rpn_cls_score, rpn_labels)


    def get_rpn_bbox_loss(self):
        assert self.eval_mode is False, 'No RPN bbox loss without ground truth boxes'
        rpn_bbox_pred = self.get_rpn_bbox_pred()
        rpn_bbox_targets = self.get_rpn_bbox_targets()
        rpn_bbox_inside_weights = self.get_rpn_bbox_inside_weights()
        rpn_bbox_outside_weights = self.get_rpn_bbox_outside_weights()
        return rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)


class roi_proposal:
    '''
    Propose highest scoring boxes to the RCNN classifier

    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, rpn_net, gt_boxes, im_dims, eval_mode):
        self.rpn_net = rpn_net
        self.rpn_cls_score = rpn_net.get_rpn_cls_score()
        self.rpn_bbox_pred = rpn_net.get_rpn_bbox_pred()
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self.num_classes = cfg.NUM_CLASSES
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES
        self.eval_mode = eval_mode

        self._network()

    def _network(self):
        # There shouldn't be any gt_boxes if in evaluation mode
        if self.eval_mode is True:
            assert self.gt_boxes is None, \
                'Evaluation mode should not have ground truth boxes (or else what are you detecting for?)'

        with tf.variable_scope('roi_proposal'):
            # Convert scores to probabilities 转换得分到概率
            self.rpn_cls_prob = rpn_softmax(self.rpn_cls_score)

            # Determine best proposals
            key = 'TRAIN' if self.eval_mode is False else 'TEST'
            self.blobs = proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob, rpn_bbox_pred=self.rpn_bbox_pred,
                                        im_dims=self.im_dims, cfg_key=key, _feat_stride=self.rpn_net._feat_stride,
                                        anchor_scales=self.anchor_scales)


    def get_rois(self):
        return  self.blobs



