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
    # label_path = '/media/user/shuju/zh/RGBT-GLASS/test_withoutglass/GT_224'
    # pre_path = '/media/user/shuju/zh/HAINet/run/2022-06-03-22-05(glassrgbt-HAI_models)/predicts_177without'
    # label_path = '/media/user/shuju/zh/RGBD-Mirror/test/GT_224_binary'
    # pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/toolbox/models/DFMNet/run/2022-07-02-15-57(mirrorrgbd-DFMNet)/predicts_MIRROR'
    label_path = '/media/user/shuju/zh/RGBD-Mirror/test/GT_416'
    pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-05-21-41(mirrorrgbd-new_year_convnext_128_5)/predicts_MIRROR'
    # pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-01-11-25(glassrgbt_merged-new_year_convnext_128_5)/predicts'
    # label_path = '/media/user/shuju/zh/RGBT-GLASS-MERGED/test/GT'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    total_bers = np.zeros((2,), dtype=float)
    total_bers_count = np.zeros((2,), dtype=float)
    avg_mae, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            pred = trans(pred).cuda(2)
            gt = trans(gt).cuda(2)
            if name == '2.png':
                print(pred, pred.shape)

            pred = pred.squeeze()

            pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            gt = torch.where(gt >= 0.5, torch.ones_like(gt), torch.zeros_like(gt))

            mea = torch.abs(pred - gt).mean()

            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
    avg_mae /= img_num
    print(avg_mae.item())