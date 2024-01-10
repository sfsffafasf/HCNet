import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2
from toolbox import get_dataset
from torch.utils.data import DataLoader
from PIL import Image

if __name__ == "__main__":
    # get img file in a list
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT_480'
    # pre_path = '/home/noone/桌面/对比模型/HAFNet/133_2/predicts_133withglass'
    # pre_path = '/home/noone/桌面/对比模型/SPNet/TU PIAN/predicts_285withglass'
    # label_path = '/media/noone/KINGSTON/对比模型/rfnet/GLASS/best/predicts_272withglass'
    # pre_path = '/home/noone/桌面/对比模型/S2MA/S2MA/predicts_270+1_withglass'
    # pre_path = '/home/noone/桌面/models/RGBD-Mirror/test/PDNet_实验结果'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/mask_single'
    pre_path = '/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-07-01-11-25(glassrgbt_merged-new_year_convnext_128_5)/predicts'
    label_path = '/media/user/shuju/zh/RGBT-GLASS-MERGED/test/GT'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/GT_416'
    # pre_path = '/home/noone/桌面/sp_vgg_new/run/2022-06-29-16-51(mirrorrgbd-new_year_convnext_128_5)/predicts_MIRROR'
    # trainset, testset = get_dataset(cfg)
    # test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
    #                          pin_memory=True).
    # label = Image.open(os.path.join('/home/noone/桌面/models/RGBD-Mirror', 'test', 'mask_single', 'GT_224_binary' + '.png')).convert('L')
    # pre_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/SalMap/predicts_395_224/MIRROR_1024'
    # label_path = '/home/noone/sunfan/RGBDBenchmark-EvaluationTools/Dataset/PDNET/MIRROR_1024/GT_1024'


    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    total_bers = np.zeros((2,), dtype=float)
    total_bers_count = np.zeros((2,), dtype=float)
    avg_ber, img_num = 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            # gt = Image.open(os.path.join('/home/noone/桌面/models/RGBD-Mirror', 'test', 'mask_single', 'GT_224_binary' + '.png')).convert('L')
            pred = trans(pred)
            gt = trans(gt)
            # print(gt, gt.shape)
            if name == '2.png':
                print(name, pred.shape, '\n', pred)



            pred = (pred >= 0.5)
            gt = (gt >= 0.5)

            N_p = torch.sum(gt) + 1e-20
            N_n = torch.sum(torch.logical_not(gt)) + 1e-20  # should we add this？

            TP = torch.sum(pred & gt)
            TN = torch.sum(torch.logical_not(pred) & torch.logical_not(gt))

            # total_bers = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    #         if total_bers == total_bers:  # for Nan
    #             total_bers += total_bers
    #             total_bers_count += 1.0
    #
    # ber = 1.0 * total_bers / total_bers_count
    # mBer = np.nanmean(ber)
    #     #
    # print(mBer*100 )

            ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

            if ber == ber:  # for Nan
                avg_ber += ber
                img_num += 1.0

    avg_ber /= img_num
    print(avg_ber.item() * 100)