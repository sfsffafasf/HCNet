import os
import time

import numpy as np
import torch
from torchvision import transforms
import cv2


class Eval_thread():
    def __init__(self, loader, method, output_dir, cuda):
        self.loader = loader
        self.method = method.replace("_results", '')
        self.dataset = "rgbt"
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        Res = {}

        ber = self.Eval_BER()
        Res['BER'] = ber

        iou = self.Eval_IOU()
        Res['IOU'] = iou

        mae = self.Eval_mae()
        Res['MAE'] = mae

        Fm, prec, recall = self.Eval_fmeasure()
        max_f = Fm.max().item()
        mean_f = Fm.mean().item()
        # prec = prec.cpu().numpy()
        # recall = recall.cpu().numpy()
        # avg_p = self.Eval_AP(prec, recall)  # AP
        # Fm = Fm.cpu().numpy()
        Res['MaxFm'] = max_f
        # Res['MeanFm'] = mean_f

        with open(self.logfile, 'a', encoding='UTF-8') as target:
            target.write(
                self.method + "," + str(round(Res['IOU'] * 100, 2)) + "," + str(round(Res['MAE'], 3)) + "," + str(
                    round(Res['BER'], 3)) + "," + str(
                    round(Res['MaxFm'], 3)) + '\n')

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

    def Eval_fmeasure(self):
        # print('eval[FMeasure]:{} dataset with {} method.'.format(
        #     self.dataset, self.method))
        beta2 = 0.3
        avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:

                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()

                    # pred = 1 - pred
                    # gt = 1 - gt
                    if torch.min(gt) != torch.max(gt):
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                           torch.min(pred) + 1e-20)
                    else:
                        pred = pred / torch.max(pred)

                else:
                    pred = trans(pred)
                    gt = trans(gt)

                    # pred = 1 - pred
                    # gt = 1 - gt

                    if torch.min(gt) != torch.max(gt):
                        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                           torch.min(pred) + 1e-20)
                    else:
                        pred = pred / torch.max(pred)

                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            Fm = avg_f / img_num
            avg_p = avg_p / img_num
            avg_r = avg_r / img_num
            return Fm, avg_p, avg_r

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                    1e-20)
        return prec, recall

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
    label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT_480'
    pre_path = '/home/noone/桌面/对比模型/HAFNet/133_2/predicts_133withglass'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT_224'
    # pre_path = '/media/noone/KINGSTON/对比模型/rfnet/GLASS/best/predicts_272withglass'
    # pre_path = '/home/noone/桌面/models/RGBD-Mirror/test/PDNet_实验结果_binary_any'
    # label_path = '/home/noone/桌面/models/RGBD-Mirror/test/mask_single'
    # label_path = '/home/noone/桌面/models/RGBT-GLASS/test_withglass/GT'
    # pre_path = '/home/noone/桌面/result_test/ACNet/yu ce tu/2022-05-21-09-59(glassrgbt-acnet)/predicts_480640_withglass'
    img_list = os.listdir(pre_path)
    # loader = []
    trans = transforms.Compose([transforms.ToTensor()])
    beta2 = 0.3
    avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0
    for i,name in enumerate(img_list):
        # if name.endswith('.png'):
            pred = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            gt = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            pred = trans(pred).cuda()
            gt = trans(gt).cuda()

            if torch.min(gt) != torch.max(gt):
                pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                   torch.min(pred) + 1e-20)
            else:
                pred = pred / torch.max(pred)

            prec, recall = torch.zeros(255).cuda(), torch.zeros(255).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, 255).cuda()
            for i in range(255):
                y_temp = (pred >= thlist[i]).float()
                tp = (y_temp * gt).sum()
                prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (gt.sum() +
                                                                        1e-20)

            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            avg_f += f_score
            avg_p += prec
            avg_r += recall
            img_num += 1.0
    Fm = avg_f / img_num
    # avg_p = avg_p / img_num
    max_f = Fm.max().item()
    print(max_f)


