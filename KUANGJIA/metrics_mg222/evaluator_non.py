import os
import time

import numpy as np
import torch
from torchvision import transforms


class Eval_thread():
    def __init__(self, loader, method, output_dir, cuda):
        self.loader = loader
        # self.method = method.replace("_results", '')
        self.method = method
        self.dataset = "rgbt"
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result_non.txt')

    def run(self):
        Res = {}

        iou = self.Eval_IOU()
        Res['IOU'] = iou

        mae = self.Eval_mae()
        Res['MAE'] = mae

        fpr = self.Eval_FPR()
        Res['FPR'] = fpr

        with open(self.logfile, 'a', encoding='UTF-8') as target:
            target.write(
                self.method + "," + str(round(Res['MAE'], 3)) + "," + str(round(Res['IOU'] * 100, 2)) + "," + str(
                    round(Res['FPR'], 2)) + '\n')

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

                # pred = 1 - pred
                # gt = 1 - gt

                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def Eval_IOU(self):
        # print('eval[IOU]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
        avg_iou, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:

                if self.cuda:
                    print(pred, gt)
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)

                pred = 1 - pred
                gt = 1 - gt

                pred = (pred >= 0.5)
                gt = (gt >= 0.5)

                iou = torch.sum((pred & gt)) / torch.sum((pred | gt))

                if iou == iou:  # for Nan
                    avg_iou += iou
                    img_num += 1.0
            avg_iou /= img_num
            return avg_iou.item()

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
    import json
    from toolbox import get_dataset

    pre_path = '/home/noone/桌面/models/predict_MCANet/predicts_23_withglass'
    label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    img_list = zip(os.listdir(pre_path), os.listdir(label_path))
    # img_list = os.listdir(pre_path), os.listdir(label_path)

    test = Eval_thread(loader=img_list, method="AAA" , output_dir="/home/noone/桌面/models/predict_MCANet", cuda=True)
    test.run()