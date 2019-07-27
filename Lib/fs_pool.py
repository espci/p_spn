# -*- coding: utf-8 -*-
import tensorflow as tf

def fs_pool(featureMaps,rois,im_dims):
    '''
    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are 
    formatted as:
    (image_id, x1, y1, x2, y2)
    
    Note: Since mini-batches are sampled from a single image, image_id = 0s
    '''
    with tf.variable_scope('fs_pool'):
        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
        box_ind = tf.cast(rois[:,0],dtype=tf.int32)

        #  box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
        boxes = rois[:,1:]

        normalization = tf.cast([im_dims[1], im_dims[0], im_dims[1], im_dims[0]], dtype=tf.float32)

        boxes = tf.div(boxes,normalization)
        boxes = tf.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]],axis=1)  # y1, x1, y2, x2
        
        # pool output size
        crop_size = tf.constant([14,14])
        
        # pool
        pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, method="nearest",box_ind=box_ind, crop_size=crop_size)

        pooledFeatures = tf.equal(pooledFeatures,11)
        pooledFeatures = tf.cast(pooledFeatures,tf.float32)
        # Max pool to (1x1)
        mean = tf.nn.avg_pool(pooledFeatures, ksize=[1, 14, 14, 1], strides=[1, 14, 14, 1], padding='SAME')

        mean = tf.squeeze(mean)
    return mean