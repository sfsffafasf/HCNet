import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


def Eval_mae(self):
    # print('eval[MAE]:{} dataset with {} method.'.format(
    #     self.dataset, self.method))
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)

            # pred = (pred >= 0.5)
            # gt = (gt >= 0.5)

            pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))

            # pred = 1 - pred
            # gt = 1 - gt

            mea = torch.abs(pred - gt).mean()
            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
        avg_mae /= img_num
        return avg_mae.item()



if __name__ == "__main__":
    # get img file in a list
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    # pre_path = '/home/noone/桌面/对比模型/HAFNet/133_2/predicts_133withglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withoutglass/GT_224'
    # pre_path = '/media/noone/KINGSTON/对比模型/rfnet/GLASS/best/predicts_272_withoutglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withoutglass/GT'
    # pre_path = '/home/noone/桌面/对比模型/HDFNet/结果251/predicts'
    # pre_path = '/home/noone/桌面/models/RGBD-Mirror/test/PDNet_实验结果'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/mask_single'
    pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-01-11-25(glassrgbt_merged-new_year_convnext_128_5)/predicts'
    label_path = '/media/user/shuju/zh/RGBT-GLASS-MERGED/test/GT'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/GT_416'
    # pre_path = '/home/noone/桌面/sp_vgg_new/run/2022-06-21-18-50(mirrorrgbd-new_year_convnext_128_5)/predicts_MIRROR'
    # pre_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/SalMap/predicts_395_224/MIRROR_1024'
    # label_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/Dataset/PDNET/MIRROR_1024/GT_1024'
    img_list = os.listdir(pre_path)
    loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    total_bers = np.zeros((2,), dtype=float)
    total_bers_count = np.zeros((2,), dtype=float)
    avg_mae, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            # pred = trans(pred).cuda()
            # gt = trans(gt).cuda()
            pred = trans(pred)
            gt = trans(gt)

            if name == '2.png' :
                print(name, pred.shape, '\n', pred)

            if name == '3.png' :
                print(name, pred.shape, '\n', pred)

            pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))

            # pred = (pred >= 0.5).float()
            # gt = (gt >= 0.5).float()

            mea = torch.abs(pred - gt).mean()

            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
    avg_mae /= img_num
    print(avg_mae.item())