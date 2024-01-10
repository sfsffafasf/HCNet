import os
import os.path as osp
import time
from tqdm import tqdm
from PIL import Image
# import imageio
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_dataset
from toolbox import get_model
from toolbox import averageMeter, runningScore
# from toolbox import class_to_RGB, load_ckpt, save_ckpt
# from torchvision.utils import save_image
# from skimage import img_as_ubyte
# import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import cv2

# from toolbox.datasets.irseg import IRSeg
# from toolbox.datasets.glassrgbt import GlassRGBT
from toolbox.datasets.mirrorrgbd import MirrorRGBD
from toolbox.msg import runMsg


def evaluate(logdir, save_predict=False, options=['train', 'val', 'test', 'test_day', 'test_night', "test_withglass", "test_withoutglass"], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device("cuda:0")

    loaders = []
    for opt in options:
        # dataset = IRSeg(cfg, mode=opt)
        # dataset = PST900(cfg, mode=opt)
        # dataset = GlassRGBT(cfg, mode=opt)
        dataset = MirrorRGBD(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap

    model = get_model(cfg).to(device)

    # model.load_state_dict(torch.load(os.path.join(logdir, prefix+'model.pth'), map_location={'cuda:1': 'cuda:0'}))
    # save_ckpt('/home/dtrimina/Desktop/lxy/Segmentation_final/run', model)

    # model = load_ckpt(logdir, model, prefix=prefix)
    model.load_state_dict(torch.load(os.path.join(logdir, '72model.pth'), map_location={'cuda:0': 'cuda:0'}))
    # model.load_state_dict(torch.load(os.path.join(logdir, 'model.pth')))

    to_pil = transforms.ToPILImage()

    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])best_testIOU_model.pth
    running_metrics_val = runMsg()
    time_meter = averageMeter()

    save_path = os.path.join(logdir, '72model.pth')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
                time_start = time.time()
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)[1]
                    # for i in range(len(predict)):
                    #     print(predict[i].shape)
                    # print(sample['label_path'], predict.dtype, "predict=", predict, '\n', 'sig_pre = ', torch.sigmoid(predict))
                # if sample['label_path'][0] == '2.png':
                #     print(predict[:1, :, :, :], predict[:1, :, :, :].shape, '\n',
                #           torch.sigmoid(predict[:1, :, :, :]), predict[:1, :, :, :].shape, sample['label_path'][0])

                # predict = torch.sigmoid(predict)
                # print(predict.shape)
                # predict = predict.cpu().numpy()
                # # print(predict)
                # # predict = torch.squeeze(predict, 0)
                # # print(predict.shape)
                predict = torch.sigmoid(predict)
                running_metrics_val.update(label.cpu().float(), predict.cpu().float())
                predict = predict.squeeze()

                #predict = predict.squeeze()  #224 224
                # if sample['label_path'][0] == '2.png':
                #     print(predict, predict.shape)


                #mth1
                # if sample['label_path'][0] == '2.png':
                #     print(predict.shape)
                # predict = np.array(to_pil(predict.data.squeeze(0).cpu()))
                # Image.fromarray(predict).convert('L').save(os.path.join(save_path, sample['label_path'][0]))
                #     print((np.array(predict)/255.0).shape)
                # predict.save(os.path.join(save_path, sample['label_path'][0]))

                #mth2
                cv2.imwrite(os.path.join(save_path, sample['label_path'][0]), predict.cpu().numpy()*255)
                # torch.save(predict, os.path.join(save_path, sample['label_path'][0]) + '.pt')
                # print(save_path, sample['label_path'][0])
                # if sample['label_path'][0] == '2.png':
                #     imageio.imsave(save_path + sample['label_path'][0], img_as_ubyte(predict))

                # predict = transforms.ToPILImage(predict)
                # predict_copy = Image.Image.copy(predict)
                # save_path = osp.join(save_path + sample['label_path'][0])
                # Image.Image.save(predict_copy, fp = save_path + sample['label_path'][0])

                # predict *= 255
                # predict = Image.fromarray(predict)
                # predict = predict.convert('L')
                # if sample['label_path'][0] == '2.png':
                #     print(predict, predict.size)
                # predict.save(save_path + sample['label_path'][0])

                # save_image(predict, save_path + sample['label_path'][0])
                # print(sample['label_path'][0])
                # plt.imsave(save_path + sample['label_path'][0], arr = predict,cmap='gray')

                # predict = transforms.ToPILImage(predict)
                # Image.fromarray(predict).convert('L').save(os.path.join(save_path , sample['label_path'][0] + '.png'))

                # Image.fromarray(prediction1).convert('L').save(os.path.join(results_path, exp_name, img_name + '.png'))
                # predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                # label = label.cpu().numpy()
                # print(label.shape, predict.shape)
                # running_metrics_val.update(label, predict)

                time_meter.update(time.time() - time_start, n=image.size(0))

                # if save_predict:
                #     predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                #     predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)  # 如果数据集没有给定cmap,使用默认cmap
                #     predict = Image.fromarray(predict)
                #     predict.save(os.path.join(save_path, sample['label_path'][0]))

        # metrics = running_metrics_val.get_scores()
        # print('overall metrics .....')
        # for k, v in metrics[0].items():
        #     print(k, f'{v:.8f}')

        # print('iou for each class .....')
        # for k, v in metrics[1].items():
        #     print(k, f'{v:.8f}')
        # print('acc for each class .....')
        # for k, v in metrics[2].items():
        #     print(k, f'{v:.8f}')
        metrics = running_metrics_val.get_scores()
        # print(metrics)

        print('overall metrics .....')
        iou = metrics["iou: "].item() * 100
        ber = metrics["ber: "].item() * 100
        mae = metrics["mae: "].item()
        # F_measure = metrics["F_measure: "].item()
        print('iou:', iou, 'ber:', ber, 'mae:', mae, 'F_measure')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, default="/home/li/桌面/kuangjia/run/2023-11-20-19-23(mirrorrgbd-n11gongsgaisedeb1bugong)")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")

    args = parser.parse_args()

    # prefix option ['', 'best_val_', 'best_test_]
    # options=['test', 'test_day', 'test_night']
    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_withglass'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_withoutglass'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['val'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_day'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_night'], prefix='')
    # msc_evaluate(args.logdir, save_predict=args.s)
