import argparse
import time
import matplotlib.pyplot as plt # plt 用于显示图片
import tensorflow as tf
import numpy as np
import cv2
from Lib.nms_wrapper import nms
from utils.image_reader import get_filename_list,cityscapeInputs,_extract_mean
from tqdm import trange
from utils.createrpn import createrpn
from utils.config import Config
from Lib.fs_pool import fs_pool
from model import  ICNet_BN

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
SNAPSHOT_DIR = ''
image_dir = '/home/qmy/label/train.txt'
# mapping different model
model_config = {'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")

    parser.add_argument("--model", type=str, default='trainval_bn',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        )
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['ade20k', 'cityscapes'],
                        )
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()

def main():
    args = get_arguments()  
    cfg = Config(dataset=args.dataset, is_training=False, filter_scale=args.filter_scale)
    model = model_config[args.model]
    image_filenames= get_filename_list(image_dir)
    image_batch = tf.placeholder(tf.float32, shape=[1, 1024, 2048, 3])
    label_batch = tf.placeholder(tf.uint8, shape=[1, 1024, 2048, 1])
    gt_boxes = tf.placeholder(tf.int32, shape=[None, 5])
    images,fnames = cityscapeInputs(image_filenames, 1)
    img = tf.squeeze(image_batch)
    img = _extract_mean(img, IMG_MEAN, swap_channel=True)
    img = tf.reshape(img,[1,1024,2048,3])

    net = model(img, cfg=cfg, mode='eval')

    feature_maps = net.conv5_3
    model = createrpn(feature_maps, gt_boxes, 'EVAL')
    boxes = tf.round(model.roi_proposal_net['EVAL'].get_rois() * 255 / 1024)
    mean = fs_pool(net.output, boxes, [255,511])


    pred = tf.image.resize_bilinear(net.logits, size=[1024, 2048], align_corners=True)
    pred = tf.argmax(pred, axis=3)
    pred = tf.expand_dims(pred, axis=3)
    preds_summary = tf.py_func(Config.decode_labels, [pred, 1, 19], tf.uint8)
    preds_summary = tf.squeeze(preds_summary)

    # mIoU
    pred_flatten = tf.reshape(net.output, [-1,])
    label_flatten = tf.reshape(label_batch, [-1,])

    mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes']+1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    
    net.create_session()
    #net.restore(cfg.model_paths[args.model])
    net.restore(SNAPSHOT_DIR)
    saver = tf.train.Saver(tf.global_variables())
    for step in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
        image_bat, fname= net.sess.run([images, fnames])
        ima = image_bat.reshape([1024,2048,3])
        tf_outputs = (boxes,
                      net.output,
                      preds_summary,
                      mean
                      )
        feed_dict = {image_batch: image_bat}
        rois, temp,pred, roi_mean = net.sess.run(tf_outputs, feed_dict)

        pred_fin = pred // 2 + ima // 3
        
        f2 = time.time()
        temp = temp.reshape(255,511)
        rois = rois[:, 1:5]
        keep = np.where(roi_mean > 0.45)[0]
        rois = rois[keep, :]
        roi_mean=np.expand_dims(roi_mean[keep],axis=1)
        roisnms = np.hstack((rois,roi_mean))

        keep = np.where(roisnms[:, 4] > 0)[0]
        roisnms = roisnms[keep, :]
        keep = roisnms[:, 4].ravel().argsort()  # [::-1]
        roisnms = roisnms[keep, :]

        keep = nms(roisnms, 0.3)
        roisnms = roisnms[keep, :]
        roisnms = roisnms.astype(np.int)
        sign=danpeo(roisnms, temp, pred_fin)
        f3 = time.time()

        roisnms = roisnms * 4
        drawbbox(sign,pred_fin,roisnms)
        pred_fin = pred_fin.astype('uint8')
        ima = ima.astype('uint8')


        plt.subplot(121)
        plt.imshow(pred_fin)
        plt.subplot(122)
        plt.imshow(ima)
        plt.show()


def danpeo(roisnms,temp,pred_fin):
    sign = []
    for i in range(len(roisnms)):
        stand = (roisnms[i][3] - roisnms[i][1]) / 1.2+roisnms[i][1]
        stand = stand.astype(np.int)
        left = roisnms[i][0]
        right = roisnms[i][2]
        danger(stand,roisnms,left,right,temp,pred_fin,sign)

    return sign
def danger(stand,roisnms,left,right,temp,pred_fin,sign):

    t =temp[stand,left:right]
    tt = np.extract(t == 11, t)
    if len(tt) != 0:
        stand+=1
        if stand < 255:
            danger(stand, roisnms, left, right, temp,pred_fin,sign)
        else:
            sign.append(0)

    else:
        tt = np.extract(t == 0, t)
        if len(tt) != 0:
            sign.append(1)
        else:
            sign.append(0)
def drawbbox(sign,pred_fin,roisnms):

    for i in range(len(sign)):
        if sign[i] != 0:
            cv2.rectangle(pred_fin, (roisnms[i][0], roisnms[i][1]), (roisnms[i][2], roisnms[i][3]), (255, 0, 0), 2)
        else:
            cv2.rectangle(pred_fin, (roisnms[i][0], roisnms[i][1]), (roisnms[i][2], roisnms[i][3]), (0, 255, 0), 2)

if __name__ == '__main__':
    main()
