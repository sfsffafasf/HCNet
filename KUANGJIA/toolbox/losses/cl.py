import torch.nn.functional as F
import torch
def SCL(y_s,y):
    # Normalise representations
    #print(sum(y_s[0][0][0]))
    y_s = F.interpolate(y_s, size=(13, 13), mode='bilinear')
    z_s_norm = F.normalize(y_s, dim=1)
    b,c,h,w=z_s_norm.shape
    z_s_norm=z_s_norm.reshape(b,c,h*w).permute(0,2,1)
    z_s_norm_t=z_s_norm.permute(0,2,1)
    c_ss = torch.bmm(z_s_norm, z_s_norm_t)
    y_d = F.interpolate(y,size=(13,13),mode='bilinear')
    b, c, h, w = y_d.shape
    y_d = y_d.reshape(b, c, h * w).permute(0, 2, 1)
    y_d_t = y_d.permute(0, 2, 1)
    yy = torch.bmm(y_d, y_d_t)
    loss=0.0
    loss += torch.log2(c_ss.pow(2).sum()) / (h*h*w*w)
    loss -= torch.log2((c_ss * yy).pow(2).sum()) / (h*h*w*w)
    return loss

# if __name__ == '__main__':
#     y_s=torch.randn(2,512,13,13)
#     y = torch.randn(2, 512, 416,416)
#     y=torch.ones_like(y)
#     #print(y_s,y)
#     print(SCL(y_s,y))

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def generate_positive_samples(image_path, num_samples):
#     # 读取原始图像
#     image = cv2.imread(image_path)
#
#     # 图像增强参数设定
#     scale_range = (0.8, 1.2)  # 缩放范围
#     rotation_angle_range = (-15, 15)  # 旋转角度范围
#     translation_range = (-20, 20)  # 平移范围
#
#     positive_samples = [image]  # 将原始图像作为第一个正样本
#
#     for _ in range(num_samples - 1):
#         # 创建一个与原始图像尺寸相同的仿射变换矩阵
#         transform_matrix = np.eye(2, 3, dtype=np.float32)
#
#         # 随机生成缩放比例
#         scale = np.random.uniform(*scale_range)
#         transform_matrix *= scale
#
#         # 随机生成旋转角度
#         angle = np.random.uniform(*rotation_angle_range)
#         rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
#         transform_matrix[:, :2] = np.dot(rotation_matrix[:, :2], transform_matrix[:, :2])
#
#         # 随机生成平移距离
#         translation_x = np.random.uniform(*translation_range)
#         translation_y = np.random.uniform(*translation_range)
#         transform_matrix[:, 2] += [translation_x, translation_y]
#
#         # 进行仿射变换
#         transformed_image = cv2.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0]))
#
#         positive_samples.append(transformed_image)
#
#     return positive_samples
#
#
def SCL1(y_s,y,t):

    dist_pos = F.pairwise_distance(anchor_features, positive_features)
    dist_neg = F.pairwise_distance(anchor_features, negative_features)
    margin = 0.2

    triplet_loss = torch.mean(torch.max(dist_pos - dist_neg + margin, torch.zeros_like(dist_pos)))

    return loss


import numpy as np

def calculate_iou(box1, box2):
    # 计算两个边界框的IoU
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    iou = intersection / union
    return iou

def calculate_giou(box1, box2):
    # 计算两个边界框的GIoU
    iou = calculate_iou(box1, box2)

    xmin = min(box1[0], box2[0])
    ymin = min(box1[1], box2[1])
    xmax = max(box1[2], box2[2])
    ymax = max(box1[3], box2[3])

    area_c = (xmax - xmin) * (ymax - ymin)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    union = area_box1 + area_box2 - intersection

    giou = iou - (area_c - union) / area_c
    return giou

def balanced_similarity(scores, labels):
    # 平衡正负样本的相似度
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    min_negative = np.min(negative_scores)
    max_positive = np.max(positive_scores)

    balanced_positive_scores = positive_scores / max_positive
    balanced_negative_scores = negative_scores / min_negative

    balanced_scores = np.concatenate((balanced_positive_scores, balanced_negative_scores))
    return balanced_scores

# 示例使用
boxes = [
    [10, 10, 50, 50],  # 真实目标框1
    [30, 30, 70, 70],  # 真实目标框2
    [60, 60, 80, 80]   # 真实目标框3
]
anchors = [
    [20, 20, 40, 40],  # 锚框1
    [50, 50, 70, 70],  # 锚框2
    [80, 80, 100, 100] # 锚框3
]
labels = [1, 1, 0]      # 真实目标框的标签 (1表示正样本，0表示负样本)

iou_threshold = 0.5     # IoU阈值

scores = []
for anchor in anchors:
    for box in boxes:
        iou = calculate_iou(anchor, box)
        if iou > iou_threshold:
            giou = calculate_giou(anchor, box)
            scores.append(giou)

scores = np.array(scores)
balanced_scores = balanced_similarity(scores, labels)
