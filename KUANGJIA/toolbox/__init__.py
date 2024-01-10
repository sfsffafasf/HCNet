from .metrics import averageMeter, runningScore
from .log import get_logger
import torch.nn as nn
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .losses.loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LovaszSoftmax, LDAMLoss, MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged', "mirrorrgbd_KD", 'mirrorrgbd_xtx']

    if cfg['dataset'] == 'camvid':
        from .datasets.camvid import Camvid
        return Camvid(cfg, mode='trainval'), Camvid(cfg, mode='test')

    if cfg['dataset'] == 'nyuv2':
        from .datasets.nyuv2 import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')


    if cfg['dataset'] == 'glassrgbt':
        from toolbox.datasets.glassrgbt import GlassRGBT
        return GlassRGBT(cfg, mode='train'), GlassRGBT(cfg, mode='test_withglass'), GlassRGBT(cfg, mode='test_withoutglass')

    if cfg['dataset'] == 'glassrgbt_merged':
        from toolbox.datasets.glassrgbt import GlassRGBT
        return GlassRGBT(cfg, mode='train'), GlassRGBT(cfg, mode='test')

    if cfg['dataset'] == 'mirrorrgbd':
        from toolbox.datasets.mirrorrgbd import MirrorRGBD
        return MirrorRGBD(cfg, mode='train'), MirrorRGBD(cfg, mode='test')

    if cfg['dataset'] == 'mirrorrgbd_KD':
        from toolbox.datasets.mirrorrgbd_KD import MirrorRGBD
        return MirrorRGBD(cfg, mode='train'), MirrorRGBD(cfg, mode='test')

    if cfg['dataset'] == 'mirrorrgbd_xtx':
        from toolbox.datasets.mirrorrgbd_xtx import MirrorRGBD
        return MirrorRGBD(cfg, mode='train'), MirrorRGBD(cfg, mode='test')


def get_model(cfg):

    ############# bbbmodel ################

    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgai':
        from toolbox.models.liSPmodelbdcnb4chuangacgai import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgaib4':
        from toolbox.models.liSPmodelbdcnb4chuangacgaib4quanB import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgaib0':
        from toolbox.models.liSPmodelbdcnb4chuangacgaib0 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'shendukefengli':
        from toolbox.models.shendukefengli import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgaiSEDA':
        from toolbox.models.liSPmodelbdcnb4chuangacgaiSEDA import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgaiSEDA1':
        from toolbox.models.liSPmodelbdcnb4chuangacgaiSEDA1 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb4chuangacgaib41050':
        from toolbox.models.liSPmodelbdcnb4chuangacgaib41050 import LiSPNetx22
        return LiSPNetx22()

    if cfg['model_name'] == 'l4':
        from toolbox.models.l4 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l1':
        from toolbox.models.l1 import LiSPNetx22
        return LiSPNetx22()

    if cfg['model_name'] == 'l4pt2t.py':
        from toolbox.models.l4pt2t import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'liSPmodelbdcnb3':
        from toolbox.models.l2model.liSPmodelbdcnb3 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l2ml4':
        from toolbox.models.l2model.l4 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l2ml41':
        from toolbox.models.l2model.l41 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l2ml4M':
        from toolbox.models.l2model.l4M import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'b41051':
        from toolbox.models.b41051 import LiSPNetx22
        return LiSPNetx22()


    if cfg['model_name'] == 'l4Mqupianjuanjiquanzhongn':
        from toolbox.models.l2model.l4Mqupianjuanjiquanzhongn import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l4Mqupianjuanjiquanzhongn1':
        from toolbox.models.l2model.l4Mqupianjuanjiquanzhongn1 import LiSPNetx22
        return LiSPNetx22()
    if cfg['model_name'] == 'l4Mqupianjuanjiquanzhongn12':
        from toolbox.models.l4Mqupianjuanjiquanzhongn12 import LiSPNetx22
        return LiSPNetx22()

    if cfg['model_name'] == 'l4Mqupianjuanjiquanzhongn13':
        from toolbox.models.tujuanjikuai.l4Mqupianjuanjiquanzhongn13 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n1':
        from toolbox.models.l2model.n1wujie import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11':
        from toolbox.models.l2model.n11 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n1wujie':
        from toolbox.models.l2model.n1wujie import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11s':
        from toolbox.models.l2model.n11s import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11gongs':
        from toolbox.models.n11gongs import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11gongsseda':
        from toolbox.models.n11gongsseda import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11gongsgaisede':
        from toolbox.models.n11gongsgaisede import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11b5':
        from toolbox.models.n11b5 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11b1':
        from toolbox.models.n11b1 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n11gongsgaisedeb1gong':
        from toolbox.models.n11gongsgaisedeb1gong import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'n1wujiebase':
        from toolbox.models.l2model.n1wujiebase import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'b41051rong':
        from toolbox.models.b41051rong import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'b41051dong':
        from toolbox.models.b41051dong import LiSPNetx22

        return LiSPNetx22()

    if cfg['model_name'] == 'JJ_BM_AM_DM':
        from toolbox.models.model2.JJ_BM_AM_DM import JJNet

        return JJNet()
    if cfg['model_name'] == 'n11gongsgaisedeb1bugong':
        from toolbox.models.n11gongsgaisedeb1bugong import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'zuihao4':
        from toolbox.models.zuihao4 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'zuihaoB5':
        from toolbox.models.zuihao4B5 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'Limodel21':
        from toolbox.models.Limodel21 import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'Limodel21COPYBBSgcm':
        from toolbox.models.Limodel21COPYBBSgcm import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'limodelcopybbsgcmwujia':
        from toolbox.models.limodelcopybbsgcmwujia import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'limodelcopybbsgcmwujiasanxiang':
        from toolbox.models.limodelcopybbsgcmwujiasanxiang import LiSPNetx22

        return LiSPNetx22()
    if cfg['model_name'] == 'Limodel21COPYBBSgcm4':
        from toolbox.models.Limodel21COPYBBSgcm4 import LiSPNetx22

        return LiSPNetx22()

