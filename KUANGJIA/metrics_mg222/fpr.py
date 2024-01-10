import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


def Eval_FPR(self):
    avg_fpr, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            if np.sum(pred.cpu().numpy()) > 1:
                avg_fpr += 1

            img_num += 1

        avg_fpr /= img_num
        return avg_fpr



if __name__ == "__main__":
    # get img file in a list
    label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withoutglass/GT'
    pre_path = '/home/noone/桌面/对比模型/HAFNet/133_2/predicts_133withoutglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withoutglass/GT_224'
    # pre_path = '/media/noone/KINGSTON/对比模型/rfnet/GLASS/best/predicts_272_withoutglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withoutglass/GT'
    # pre_path = '/home/noone/桌面/result_test/ACNet/yu ce tu/2022-05-21-09-59(glassrgbt-acnet)/predicts_480640_withoutglass'
    # pre_path = '/home/noone/桌面/models/RGBD-Mirror/test/PDNet_实验结果'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/mask_single'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    avg_fpr, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            pred = trans(pred).cuda()
            gt = trans(gt).cuda()

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            if np.sum(pred.cpu().numpy()) > 1:
                avg_fpr += 1

            img_num += 1

    avg_fpr /= img_num
    print(avg_fpr)