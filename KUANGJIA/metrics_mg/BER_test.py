import torch.nn.functional as F
import os
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image

image_root = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-08-01-09-06(mirrorrgbd-hdfnet)/predicts_MIRROR_hdfnet/'
gt_root = '/media/user/shuju/zh/RGBD-Mirror/test/GT_416/'

########################  tu pian du qu
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.binary_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.transform(gt).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, name


    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

###############################################################################################################
####################################################### BER ###################################################
test_loader = test_dataset(image_root, gt_root, 416)

avg_ber, img_num = 0.0, 0.0
total_bers = np.zeros((2,), dtype=np.float)
total_bers_count = np.zeros((2,), dtype=np.float)
with torch.no_grad():
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        predict = image.cuda(0)
        gt = gt.cuda(0)
        pred = (predict >=0.5)
        # print('predict', predict, 'pred', pred)
        label = (gt >=0.5)
        # print('gt', gt, 'label', label)
        N_p = torch.sum(label) + 1e-20
        N_n = torch.sum(torch.logical_not(label)) + 1e-20
        TP = torch.sum(pred & label)
        TN = torch.sum(torch.logical_not(pred) & torch.logical_not(label))
        ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))
        if ber == ber:
            avg_ber += ber
            img_num += 1.0
avg_ber /= img_num
ber_last = avg_ber.item() * 100
print('test_ber', ber_last)
###############################################################################################################
########################################################### IOU ###############################################
test_loader = test_dataset(image_root, gt_root, 416)

avg_iou, img_num = 0.0, 0.0
with torch.no_grad():
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        predict = image.cuda(0)
        gt = gt.cuda(0)
        pred = (predict >=0.5)
        label = (gt >=0.5)

        iou = torch.sum((pred & label)) / torch.sum((pred | label))

        if iou == iou:  # for nan
            avg_iou += iou
            img_num += 1.0
avg_iou /= img_num
avg_iou_last = avg_iou.item()
print('test_iou', avg_iou_last)
##############################################################################################################
###################################################### MAE ###################################################
test_loader = test_dataset(image_root, gt_root, 416)
avr_mae = 0.0

with torch.no_grad():
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        predict = image.cuda(0)
        gt = gt.float().cuda(0)

        maeval = torch.sum(torch.abs(gt - predict)) / (416.0*416.0)
        avr_mae += maeval

avr_mae /= test_loader.size
avr_mae_last = avr_mae.item()
print('test_mae', avr_mae_last)
#############################################################################################################
####################################################### Fmeasure ############################################
