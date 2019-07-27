import numpy as np
import os
from PIL import Image
label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle

class Config(object):
    # Setting dataset directory
    CITYSCAPES_DATA_DIR = '/media/qmy/C65CD9A25CD98D95/cityscapesScripts'
    ADE20K_DATA_DIR = './data/ADEChallengeData2016/'
      
    ADE20K_eval_list = os.path.join('./data/list/ade20k_val_list.txt')
    CITYSCAPES_eval_list = os.path.join('./data/list/cityscapes_val_list.txt')
    
    ADE20K_train_list = os.path.join('./data/list/ade20k_train_list.txt')
    CITYSCAPES_train_list = os.path.join('./data/list/cityscapes_train_list.txt')
    
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
    ADE20k_param = {'name': 'ade20k',
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'eval_size': [480, 480],
                'eval_steps': 2000,
                'eval_list': ADE20K_eval_list,
                'train_list': ADE20K_train_list,
                'data_dir': ADE20K_DATA_DIR}
    
    cityscapes_param = {'name': 'cityscapes',
                    'num_classes': 19,
                    'ignore_label': 255,
                    'eval_size': [1024, 2048],
                    'eval_steps': 25000,
                    'eval_list': CITYSCAPES_eval_list,
                    'train_list': CITYSCAPES_train_list,
                    'data_dir': CITYSCAPES_DATA_DIR}
    
    model_paths = {'train': './model/cityscapes/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/cityscapes/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': '/home/qmy/视频/ICNet-tensorflow-master/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': './model/ade20k/model.ckpt-27150'}
    
    ## If you want to train on your own dataset, try to set these parameters.
    others_param = {'name': 'YOUR_OWN_DATASET',
                    'num_classes': 0,
                    'ignore_label': 0,
                    'eval_size': [0, 0],
                    'eval_steps': 0,
                    'eval_list': '/PATH/TO/YOUR_EVAL_LIST',
                    'train_list': '/PATH/TO/YOUR_TRAIN_LIST',
                    'data_dir': '/PATH/TO/YOUR_DATA_DIR'}

    ## You can modify following lines to train different training configurations.
    INFER_SIZE = [1024, 2048, 3] 
    TRAINING_SIZE = [720, 720]
    TRAINING_STEPS = 60001
    
    N_WORKERS = 1
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    POWER = 0.9
    RANDOM_SEED = 1234
    WEIGHT_DECAY = 0.0001
    SNAPSHOT_DIR = './snapshots/'
    SAVE_NUM_IMAGES = 4
    SAVE_PRED_EVERY = 500
    
    # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    
    def __init__(self, dataset, is_training=False, filter_scale=1, random_scale=False, random_mirror=False):
        print('Setup configurations...')
        
        if dataset == 'ade20k':
            self.param = self.ADE20k_param
        elif dataset == 'cityscapes':
            self.param = self.cityscapes_param
        elif dataset == 'others':
            self.param = self.others_param

        self.dataset = dataset
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.is_training = is_training
        self.filter_scale = filter_scale
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))

        print("\n")

    def decode_labels(mask, num_images=1, num_classes=21):
        """Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
        """
        n, h, w, c = mask.shape
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs