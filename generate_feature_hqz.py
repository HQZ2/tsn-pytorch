import torch
import numpy as np
import torch.utils.data as data
from transforms import *
from PIL import Image
import os
import os.path
from opts import parser
import pickle
from models import TSN
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import h5py

ACTNET200V13_PKL_SMALL = '/mnt/workspace/pkls/small_actNet200-V1-3.pkl'
VIDEO_PATH = '/mnt/workspace/activitynet-frames/resized-activitynet-frames'
FRAMES_NUM_PKL = '/mnt/workspace/pkls/frames_num.pkl'


class DataSet(data.Dataset):
    def __init__(self, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True):

        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.load_groundtruth()

    def load_image(self, img_path):
        try:
            return [Image.open(img_path).convert('RGB')]
        except:
            print("Couldn't load image:{}".format(img_path))
            return None

    def load_groundtruth(self):
        with open(ACTNET200V13_PKL_SMALL, 'rb') as f:
            self.small_groundtruth = pickle.load(f)['database']
        with open(FRAMES_NUM_PKL, 'rb') as f:
            self.frames_num = pickle.load(f)

    def nextvideo(self):
        for key in self.small_groundtruth.keys():
            key2 = 'v_{}.mp4'.format(key)
            try:
                num = self.frames_num[key2]
            except:
                continue
            video_path = os.path.join(VIDEO_PATH, key2)
            idx = 8
            imgs = []
            while idx < num:
                seg_imgs = self.load_image(os.path.join(video_path, self.image_tmpl.format(idx)))
                if seg_imgs is None:
                    continue
                tran_img = self.transform(seg_imgs)
                imgs.append(tran_img.numpy())
                # images[key2] = tran_img
                idx += 16

            imgs = np.concatenate(imgs)
            imgs = np.reshape(imgs,(-1,3,224,224))
            yield key, imgs

    # def nextbatch(self):
    #     return self.get_test_indices()


class ExtructFeatrue(object):

    def __init__(self):
        saved = torch.load('/mnt/workspace/model/activitynet_kinetics400_resnet152_rgb_model/activitynet_resnet152_rgb_model_best_054.pth.tar')
        self.model = TSN(201,3,'RGB','resnet152',1)
        self.train_augmentation = self.model.get_augmentation()
        self.input_mean = self.model.input_mean
        self.input_std = self.model.input_std
        self.softmax = nn.Softmax(dim=-1).cuda()
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(saved['state_dict'])

        self.base_model = nn.DataParallel(self.model.module.base_model).cuda()
        self.new_fc     = nn.DataParallel(self.model.module.new_fc).cuda()

        self.model.eval()
        self.base_model.eval()
        self.new_fc.eval()

    def loadFeatrue(self,x):

        midfeature = self.base_model(x)
        classfeature = self.softmax(self.new_fc(midfeature))
        return midfeature,classfeature


if __name__ == '__main__':

    EF = ExtructFeatrue()
    # input = Variable(torch.randn(8, 3, 224, 224), volatile=True)
    # a, b = EF.loadFeatrue(input)
    # import IPython;IPython.embed()

    # args = parser.parse_args()
    normalize = GroupNormalize(EF.input_mean, EF.input_std)
    import easydict
    args = easydict.EasyDict()
    args.modality = 'RGB'
    args.arch ='resnet152'

    dataset = DataSet(modality=args.modality,
            image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB",
                                                               "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
            transform=torchvision.transforms.Compose([
                EF.train_augmentation,
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                normalize,
            ]))
    gen = dataset.nextvideo()

    f = h5py.File('/mnt/workspace/feature/small_RGB_feature.h5', 'w')

    for idx,item in enumerate(gen):
        vid = item[0]
        imgs = item[1]
        n = imgs.shape[0]
        BATCH = 128
        ticks = np.arange(0,n-0.0001,BATCH)

        mids = []
        clas  = []

        for fr in ticks:

            to = min(fr+BATCH,n)
            fr = int(fr)
            to = int(to)
            #print(fr,to,ticks)

            input_var = torch.autograd.Variable(torch.from_numpy(imgs[fr:to,...]), volatile=True).cuda()

            mid, cla = EF.loadFeatrue(input_var)
            mid = mid.cpu().data.numpy()
            cla = cla.cpu().data.numpy()

            mids.append(mid)
            clas.append(cla)

        mids = np.concatenate(mids)
        clas = np.concatenate(clas)

        f['{}/2048'.format(vid)] = mids
        f['{}/201'.format(vid)]  = clas

        print(idx,vid)

    f.close()


