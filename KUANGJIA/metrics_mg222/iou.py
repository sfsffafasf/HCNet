import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


def Eval_IOU(self):
    # print('eval[IOU]:{} dataset with {} method.'.format(
    #     self.dataset, self.method))
    avg_iou, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:

            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:

                pred = trans(pred)
                gt = trans(gt)

            # pred = 1 - pred
            # gt = 1 - gt

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

            if iou == iou:  # for Nan
                avg_iou += iou
                img_num += 1.0
        avg_iou /= img_num
        return avg_iou.item()








if __name__ == "__main__":
    # get img file in a list
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    # pre_path = '/home/noone/桌面/对比模型/HAFNet/133_2/predicts_133withglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    # pre_path = '/home/noone/桌面/result_test/ACNet/yu ce tu/2022-05-21-09-59(glassrgbt-acnet)/predicts_480640_withglass'
    # pre_path = '/home/noone/桌面/models/RGBD-Mirror/test/PDNet_实验结果'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/mask_single'
    pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-01-11-25(glassrgbt_merged-new_year_convnext_128_5)/predicts'
    label_path = '/media/user/shuju/zh/RGBT-GLASS-MERGED/test/GT'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/GT_416'
    # pre_path = '/home/noone/桌面/sp_vgg_new/run/2022-06-21-18-50(mirrorrgbd-new_year_convnext_128_5)/predicts_MIRROR'
    # pre_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/SalMap/predicts_395_224/MIRROR_1024'
    # label_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/Dataset/PDNET/MIRROR_1024/GT_1024'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    avg_iou, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            # print("111", pred, gt)
            pred = trans(pred)
            gt = trans(gt)
            # print("222", pred, gt)

            if name == '2.png' :
                print(name, pred.shape, '\n', pred)

            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            # print("333", pred, gt)

            iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

            # total_bers = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    #         if total_bers == total_bers:  # for Nan
    #             total_bers += total_bers
    #             total_bers_count += 1.0
    #
    # ber = 1.0 * total_bers / total_bers_count
    # mBer = np.nanmean(ber)
    #     #
    # print(mBer*100 )

            # ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

            if iou == iou:  # for Nan
                avg_iou += iou
                img_num += 1.0


    avg_iou /= img_num
    print(avg_iou.item())
