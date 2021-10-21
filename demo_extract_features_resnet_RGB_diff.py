import os
import skvideo.io
from PIL import Image
import random
import numpy as np
import argparse
import timeit
import pickle
import torch
import torch.nn as nn
from scipy import stats
from init_models.resnet import generate_2d_resnet
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
from math import ceil

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DatasetValidation_RGBDiff(Dataset):
    r"""
    A Validation Dataset for a folder of videos

    args:
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
    """

    def __init__(self, dataset='LIVE', basep='/mnt/disk/yongxu_liu/datasets', nsnippet=20):

        self.basep = basep
        self.dataset = dataset

        self.nsnippet = nsnippet
        self.n_data, self.info = self.get_test_info(dataset)

    def get_test_batch(self, file, width, height, frms):

        if min(width, height) > 760:
            scale = ceil(min(width, height) / 256) - 1
        else:
            scale = ceil(min(width, height) / 256)

        scaleW = width // scale
        scaleH = height // scale
        transform = tf.Compose([
            # tf.Resize([scaleH, scaleW]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ts = 2 if frms < 480 else 4
        # N = self.nfrm
        T = self.nsnippet
        offset = 10

        stride = (frms - offset * 2 - ts) / (T - 1)
        pos = np.asarray(range(T)) * stride + offset - 1
        pos = pos.astype('int')

        vid = np.zeros((T, 3, scaleH, scaleW), dtype=np.float32)
        vdd = np.zeros((T, 3, scaleH, scaleW), dtype=np.float32)

        buffer = skvideo.io.vread(file + '.yuv', height=height, width=width, inputdict={'-pix_fmt': 'yuv420p'})

        for i in range(T):
            thispos = pos[i]

            im1 = buffer[thispos].astype('float32')
            im2 = buffer[thispos + ts].astype('float32')

            im1 = self.AveragePooling(im1, scale) / 255.
            im2 = self.AveragePooling(im2, scale) / 255.
            im1 = transform(im1)
            im2 = transform(im2)

            vid[i] = np.copy(im1)
            vdd[i] = np.copy(im2 - im1)
        return vid, vdd

    def get_test_info(self, dataset):
        info_file = '{}/{}/{}_list_for_VQA.txt'.format(self.basep, dataset, dataset)
        ref_list, dis_list, d2r_list, score_list, width_list, height_list, fps_list, frms_list = \
            [], [], [], [], [], [], [], []
        with open(info_file, 'r') as f:
            for line in f:
                scn_idx, dis_idx, ref, dis, score, width, height, frm = line.split()
                scn_idx = int(scn_idx)
                dis_idx = int(dis_idx)
                width = int(width)
                height = int(height)
                frames = int(frm)

                dis_list.append('{}/{}/{}'.format(self.basep, dataset, dis[:-4]))
                ref_list.append('{}/{}/{}'.format(self.basep, dataset, ref[:-4]))
                d2r_list.append(scn_idx)
                score_list.append(float(score))
                width_list.append(width)
                height_list.append(height)
                frms_list.append(frames)

        ref_list = np.asarray(ref_list)
        dis_list = np.asarray(dis_list)
        d2r_list = np.array(d2r_list, dtype='int')
        score_list = np.array(score_list, dtype='float32')
        width_list = np.array(width_list, dtype='int')
        height_list = np.array(height_list, dtype='int')
        frms_list = np.array(frms_list, dtype='int')

        score_list = 1 + 8.5 * (score_list - score_list.min()) / (score_list.max() - score_list.min())

        return len(score_list), {'dis': dis_list,
                                 'mos': score_list,
                                 'frms': frms_list,
                                 'width': width_list,
                                 'height': height_list}

    def AveragePooling(self, img, kernel=2):

        # result = block_reduce(img, block_size=(kernel, kernel, 1), func=np.mean)
        H, W, C = img.shape
        Hc = H // kernel
        Wc = W // kernel
        H = Hc * kernel
        W = Wc * kernel

        # crop if possible
        img = img[:H, :W]

        # integral
        data = np.reshape(img, [H * Wc, kernel, C])
        data = np.sum(data, axis=1)

        data = np.reshape(np.transpose(np.reshape(data, [H, Wc, C]), [1, 0, 2]), [Hc * Wc, kernel, C])
        data = np.sum(data, axis=1)

        result = np.transpose(np.reshape(data, [Wc, Hc, C]), [1, 0, 2]) / (kernel * kernel)
        return result

    def __getitem__(self, index):
        file = self.info['dis'][index]
        frms = self.info['frms'][index]

        width = self.info['width'][index]
        height = self.info['height'][index]
        dis, dif = self.get_test_batch(file, width, height, frms)
        mos = self.info['mos'][index]
        return dis, dif, mos

    def __len__(self):
        return self.n_data


class HEKE_BVQA_resnet(nn.Module):
    def __init__(self, depth=50, group=1, hidden=32):
        """Multi-Knowledge Ensemble Learning for VQA.
        - MS-SSIM
        - GMSD
        - ST-GMSD
        - ST-RRED

        Args:
            depth (int): resnet depth, default to 50
            group (int): group convolution for each method
            hidden (int): hidden size for FC layer, default to 32
        """
        super(HEKE_BVQA_resnet, self).__init__()

        group = group * 4
        self.group = group
        self.spatial = generate_2d_resnet(model_depth=depth, pretrained=False)  # output [2048 x M x N]
        self.temporal = generate_2d_resnet(model_depth=34, pretrained=False)    # output [512 x M x N]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # feature distribution is varied among contents. How to solve?
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.group_conv_1 = nn.Conv2d(in_channels=5120, out_channels=1024,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1)
        self.group_conv_2 = nn.Conv2d(in_channels=1024, out_channels=hidden*group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_3 = nn.Conv2d(in_channels=hidden*group, out_channels=group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.act = nn.ELU()

        # init weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                #                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, dif):
        frgb1, frgb2, frgb3, frgb4 = self.spatial(rgb)
        fdif1, fdif2, fdif3, fdif4 = self.temporal(dif)

        rgb = frgb4
        dif = fdif4

        rgb1 = self.maxpool(rgb)
        rgb2 = self.avgpool(rgb)
        dif1 = self.maxpool(dif)
        dif2 = self.avgpool(dif)

        x = torch.cat((rgb1, rgb2, dif1, dif2), 1)  # [N, 5120, 1]
        x = self.group_conv_1(x)
        x = self.act(x)
        f5 = x.flatten(1)
        x = self.group_conv_2(x)
        x = self.act(x)
        x = self.group_conv_3(x)

        # -------------------------------------------
        f1_ave = self.avgpool(frgb1).flatten(1)
        f1_max = self.maxpool(frgb1).flatten(1)
        f1_std = torch.std(frgb1, dim=(2, 3), keepdim=False)

        f2_ave = self.avgpool(frgb2).flatten(1)
        f2_max = self.maxpool(frgb2).flatten(1)
        f2_std = torch.std(frgb2, dim=(2, 3), keepdim=False)

        f3_ave = self.avgpool(frgb3).flatten(1)
        f3_max = self.maxpool(frgb3).flatten(1)
        f3_std = torch.std(frgb3, dim=(2, 3), keepdim=False)

        f4_ave = self.avgpool(frgb4).flatten(1)
        f4_max = self.maxpool(frgb4).flatten(1)
        f4_std = torch.std(frgb4, dim=(2, 3), keepdim=False)

        frgb = (torch.cat([f1_ave, f1_max, f1_std], dim=1).cpu().numpy(),
                torch.cat([f2_ave, f2_max, f2_std], dim=1).cpu().numpy(),
                torch.cat([f3_ave, f3_max, f3_std], dim=1).cpu().numpy(),
                torch.cat([f4_ave, f4_max, f4_std], dim=1).cpu().numpy())

        # ---------------------------------------------
        f1_ave = self.avgpool(fdif1).flatten(1)
        f1_max = self.maxpool(fdif1).flatten(1)
        f1_std = torch.std(fdif1, dim=(2, 3), keepdim=False)

        f2_ave = self.avgpool(fdif2).flatten(1)
        f2_max = self.maxpool(fdif2).flatten(1)
        f2_std = torch.std(fdif2, dim=(2, 3), keepdim=False)

        f3_ave = self.avgpool(fdif3).flatten(1)
        f3_max = self.maxpool(fdif3).flatten(1)
        f3_std = torch.std(fdif3, dim=(2, 3), keepdim=False)

        f4_ave = self.avgpool(fdif4).flatten(1)
        f4_max = self.maxpool(fdif4).flatten(1)
        f4_std = torch.std(fdif4, dim=(2, 3), keepdim=False)

        fdif = (torch.cat([f1_ave, f1_max, f1_std], dim=1).cpu().numpy(),
                torch.cat([f2_ave, f2_max, f2_std], dim=1).cpu().numpy(),
                torch.cat([f3_ave, f3_max, f3_std], dim=1).cpu().numpy(),
                torch.cat([f4_ave, f4_max, f4_std], dim=1).cpu().numpy())

        return x.flatten(1), (frgb, fdif)


def fn_fix_seed(seed=12318):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('--' * 12)
    print('seed: %d' % seed)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ': ', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ': ', num_param)
            total_param += num_param
    return total_param


def run_blindly(args):
    # if args.seed > 0:
    #     fn_fix_seed(args.seed)

        # [DEBUG MODE]
    try:
        os.environ['IPYTHONENABLE']

        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        args.load = './pretrained/resnet_model_HEKE4.pkl'
    except KeyError:
        print('in Release mode')

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    database = args.database

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdataset = DatasetValidation_RGBDiff(dataset=database, basep='/mnt/disk/yongxu_liu/datasets')
    test_loader = DataLoaderX(testdataset, batch_size=1, shuffle=False)

    model = HEKE_BVQA_resnet()
    state_dict = torch.load(args.load)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1 and args.multi_gpu is True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))

    else:
        model = model.to(device)
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))
    print('number of trainable parameters = ', count_parameters(model))

    print(database)
    print(args.model)

    # ---------------------------------
    model.eval()
    feat = []
    pred_gmsd, pred_stgmsd, pred_strred, pred_msssim, y_label = [], [], [], [], []
    with torch.no_grad():
        for i, (dis, dif, mos) in enumerate(test_loader):
            print('{} '.format(i), end='', flush=True)
            batch = dis.shape[0]
            ns = dis.shape[1]
            sh2 = dis.shape[2:]

            dis = dis.view(torch.Size([batch * ns]) + sh2).to(device)
            dif = dif.view(torch.Size([batch * ns]) + sh2).to(device)
            y_label.append(mos.cpu().numpy())

            outputs, ff = model(dis, dif)
            gmsd, stgmsd, strred, msssim = torch.mean(outputs, 0).cpu().numpy()

            pred_gmsd.append(gmsd)
            pred_stgmsd.append(stgmsd)
            pred_strred.append(strred)
            pred_msssim.append(msssim)

            feat.append(ff)

    y_label = np.asarray(y_label).reshape(-1)
    pred_gmsd = np.asarray(pred_gmsd).reshape(-1)
    pred_stgmsd = np.asarray(pred_stgmsd).reshape(-1)
    pred_strred = np.asarray(pred_strred).reshape(-1)
    pred_msssim = np.asarray(pred_msssim).reshape(-1)

    SROCC_gmsd = stats.spearmanr(pred_gmsd, y_label)[0]
    SROCC_stgmsd = stats.spearmanr(pred_stgmsd, y_label)[0]
    SROCC_strred = stats.spearmanr(pred_strred, y_label)[0]
    SROCC_msssim = stats.spearmanr(pred_msssim, y_label)[0]

    print('---' * 7)
    print('{} = {}'.format('DMOS', y_label))
    print('{} = {}'.format('gmsd', pred_gmsd))
    print('{} = {}'.format('stgmsd', pred_stgmsd))
    print('{} = {}'.format('strred', pred_strred))
    print('{} = {}'.format('msssim', pred_msssim))
    print('---' * 7)
    print('')

    print(' ========= {} ========='.format(args.database))
    print('')
    print('  gmsd: {:.4f}'.format(SROCC_gmsd))
    print('stgmsd: {:.4f}'.format(SROCC_stgmsd))
    print('strred: {:.4f}'.format(SROCC_strred))
    print('msssim: {:.4f}'.format(SROCC_msssim))
    print('-----' * 5)

    with open('{p}/{db}.feat_{pf}.pkl'.format(p=args.save, db=args.database, pf=args.prefix), 'wb') as f:
        pickle.dump({'features': feat, 'mos': y_label}, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Blindly assessing and feature extraction with HEKE-4 (resnet)')
    parser.add_argument('--model_depth', default=50, type=int,
                        help='model: ResNet{10|18|34|50}. Default to 50')  # we use ResNet50 in the work
    parser.add_argument('--model', default='resnet_HEKE_4', type=str,
                        help='which model name to run and save')
    parser.add_argument('--load', default='./pretrained/resnet_model_HEKE4.pkl', type=str,
                        help='full name of model to load')
    parser.add_argument('--save', default='./data/resnet_HEKE_4',
                        help='path to save features')
    parser.add_argument('--prefix', default='',
                        help='prefix for saving')

    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')

    args = parser.parse_args()
    print(args)

    run_blindly(args=args)

# ==========================================================================================


