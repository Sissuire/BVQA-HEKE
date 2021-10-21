import os
import random
import numpy as np
import argparse
import timeit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from network import HEKE_BVQA_resnet as myModel
from dataset import DatasetTraining_RGBDiff, DatasetValidation_RGBDiff

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def loss_criterion(criterion, pred, gmsd, stgmsd, strred, msssim, epsilon):

    loss1 = criterion(pred[:, 0], gmsd.view(-1)) + \
            criterion(pred[:, 1], stgmsd.view(-1)) + \
            criterion(pred[:, 2], strred.view(-1)) + \
            criterion(pred[:, 3], msssim.view(-1))

    # loss2 = torch.mean(torch.std(pred_gmsd, 1)) + \
    #         torch.mean(torch.std(pred_stgmsd, 1)) + \
    #         torch.mean(torch.std(pred_strred, 1)) + \
    #         torch.mean(torch.std(pred_msssim, 1))

    loss2 = torch.sum(torch.std(pred, 1))
    loss = loss1 - epsilon * loss2
    return loss
    # return loss1


def savemodel(model, savepath):
    torch.save(model, savepath)


def train_model(model, device, optimizer, scheduler, train_loader, test_loaders, args, init_epoch=0):

    num_epochs = args.epoch
    group = args.group
    epsilon = args.epsilon
    steps = args.steps
    savep = args.save

    criterion = nn.SmoothL1Loss()

    step = 0
    step_loss = 0.
    epoch_loss = 0.

    best_epoch_loss = 1e7

    for epoch in range(init_epoch, init_epoch+num_epochs):
        start_time = timeit.default_timer()
        model.train()
        for dis, dif, msssim, gmsd, stgmsd, strred, vmaf in train_loader:
            # batch = dis.shape[0]
            # ns = dis.shape[1]
            # sh2 = dis.shape[2:]
            dis = dis.to(device)
            dif = dif.to(device)
            # dis = dis.view(torch.Size([batch * ns]) + sh2).to(device)
            msssim = msssim.view(-1, 1).to(device).float()
            gmsd = gmsd.view(-1, 1).to(device).float()
            stgmsd = stgmsd.view(-1, 1).to(device).float()
            strred = strred.view(-1, 1).to(device).float()
            vmaf = vmaf.view(-1, 1).to(device).float()

            optimizer.zero_grad()

            pred = model(dis, dif)
            # pred_msssim, pred_gmsd, pred_stgmsd, pred_strred, pred_vmaf = model(dis)

            loss = loss_criterion(criterion,
                                  pred, gmsd, stgmsd, strred, msssim,
                                  epsilon=epsilon)

            loss.backward()
            optimizer.step()

            step += 1
            step_loss += loss.cpu().float()
            epoch_loss += loss.cpu().float()

            if step % steps == 0:
                print('Train - epoch: {epoch:03d}, step: {step:05d}*{steps:d}, step_loss: {step_loss:.4f}.'.format(
                    epoch=epoch, step=step//steps, steps=steps, step_loss=step_loss))
                step_loss = 0.

        scheduler.step(epoch_loss)

        stop_time = timeit.default_timer()
        minutes, seconds = divmod(stop_time - start_time, 60)
        print('--' * 8)
        print('Train - epoch: {epoch:03d}, epoch_loss: {epoch_loss:.4f}, elapsed: {minute:03.0f}:{second:02.0f}'.format(
            epoch=epoch, epoch_loss=epoch_loss, minute=minutes, second=seconds))
        print('--' * 8)

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            if args.multi_gpu:
                state = {'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'best_epoch_loss': best_epoch_loss}
            else:
                state = {'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'best_epoch_loss': best_epoch_loss}
            torch.save(state, '{}/model_best_epoch_loss.pkl'.format(savep))

        epoch_loss = 0.

        # save model
        # if epoch > 50 and epoch % save_freq == 0:
        if args.multi_gpu:
            state = {'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
        else:
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
        torch.save(state, '{savep}/model_at_{epoch:03d}_epoch.pkl'.format(savep=savep, epoch=epoch))

        # test model

        model.eval()
        print('+++' * 8)
        for dataset in test_loaders.keys():
            pred_msssim_l, pred_gmsd_l, pred_stgmsd_l, pred_strred_l, pred_msssim_l = [], [], [], [], []
            gt_l = []
            for dis, dif, mos in test_loaders[dataset]:
                batch = dis.shape[0]
                ns = dis.shape[1]
                sh2 = dis.shape[2:]

                dis = dis.view(torch.Size([batch * ns]) + sh2).to(device)
                dif = dif.view(torch.Size([batch * ns]) + sh2).to(device)
                gt_l.extend(mos.view(-1).float())

                with torch.set_grad_enabled(False):
                    # pred_msssim, pred_gmsd, pred_stgmsd, pred_strred, pred_vmaf = model(dis)

                    pred = model(dis, dif)
                    pred_gmsd = torch.mean(pred[:, :group], 1)
                    pred_stgmsd = torch.mean(pred[:, group:group*2], 1)
                    pred_strred = torch.mean(pred[:, group*2:group*3], 1)
                    pred_msssim = torch.mean(pred[:, group*3:], 1)

                    # pred_msssim = torch.mean(pred_msssim.view((batch, ns)), dim=1)
                    pred_gmsd = torch.mean(pred_gmsd.view((batch, ns)), dim=1)
                    pred_stgmsd = torch.mean(pred_stgmsd.view((batch, ns)), dim=1)
                    pred_strred = torch.mean(pred_strred.view((batch, ns)), dim=1)
                    pred_msssim = torch.mean(pred_msssim.view((batch, ns)), dim=1)

                    # pred_msssim_l.extend(pred_msssim.view(-1).cpu().float())
                    pred_gmsd_l.extend(pred_gmsd.view(-1).cpu().float())
                    pred_stgmsd_l.extend(pred_stgmsd.view(-1).cpu().float())
                    pred_strred_l.extend(pred_strred.view(-1).cpu().float())
                    pred_msssim_l.extend(pred_msssim.view(-1).cpu().float())

            # pred_msssim_l = np.asarray(pred_msssim_l, dtype=np.float32)
            pred_gmsd_l = np.asarray(pred_gmsd_l, dtype=np.float32)
            pred_stgmsd_l = np.asarray(pred_stgmsd_l, dtype=np.float32)
            pred_strred_l = np.asarray(pred_strred_l, dtype=np.float32)
            pred_msssim_l = np.asarray(pred_msssim_l, dtype=np.float32)
            gt_l = np.asarray(gt_l, dtype=np.float32)

            # msssim_SROCC = abs(spearmanr(pred_msssim_l, gt_l)[0])
            gmsd_SROCC = abs(spearmanr(pred_gmsd_l, gt_l)[0])
            stgmsd_SROCC = abs(spearmanr(pred_stgmsd_l, gt_l)[0])
            strred_SROCC = abs(spearmanr(pred_strred_l, gt_l)[0])
            msssim_SROCC = abs(spearmanr(pred_msssim_l, gt_l)[0])

            print('Validation - epoch:{epoch:03d}, [{db}|SROCC]-gmsd: {gmsd:.4f}, '
                  'stgmsd: {stgmsd:.4f}, strred: {strred:.4f}, msssim: {msssim: .4f}.'.format(
                epoch=epoch, db=dataset, gmsd=gmsd_SROCC,
                stgmsd=stgmsd_SROCC, strred=strred_SROCC, msssim=msssim_SROCC))
        print('+++' * 8)


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


def run(opt):

    seed = opt.seed  # fix seed
    if seed > 0:
        fn_fix_seed(seed=seed)

    # save_checkpoint = opt.save_model
    # load_checkpoint = opt.load_model

    LEARNING_RATE = opt.lr
    L2_REGULARIZATION = opt.weight_decay
    NUM_EPOCHS = opt.epoch
    MULTI_GPU_MODE = opt.multi_gpu

    train_batch = opt.train_batch
    test_batch = opt.test_batch

    # save_freq = opt.save_freq
    depth = opt.model_depth
    init_epoch = opt.init_epoch

    # [DEBUG MODE]
    try:
        os.environ['IPYTHONENABLE']

        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '9'
        MULTI_GPU_MODE = True
        opt.dynamic = True
        # opt.speedup = True
        init_epoch = 20

        # opt.resume = True
        # opt.load = './save_baseline_v2_1/model_at_025_epoch.pkl'
    except KeyError:
        print('No debug')

    # depth = 10
    # isDynamic = False

    testset_LIVE = DatasetValidation_RGBDiff(dataset='LIVE',
                                             basep='/data/sissuire/datasets/images4database_downscale',
                                             height=216, width=384)
    testset_CSIQ = DatasetValidation_RGBDiff(dataset='CSIQ',
                                             basep='/data/sissuire/datasets/images4database_downscale',
                                             height=240, width=416)
    testset_IVPL = DatasetValidation_RGBDiff(dataset='IVPL_480',
                                             basep='/data/sissuire/datasets/images4database_downscale',
                                             height=272, width=480)
    test_loaders = {
        'LIVE': DataLoaderX(testset_LIVE, batch_size=1, shuffle=False, num_workers=2, drop_last=False),
        'CSIQ': DataLoaderX(testset_CSIQ, batch_size=1, shuffle=False, num_workers=2, drop_last=False),
        'IVPL': DataLoaderX(testset_IVPL, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    }

    basep = '/home/sissuire/datasets/Video_Set/dis_png'
    info = 'dataset.clean_info.rgbdiff.lthpc.pkl'
    trainset = DatasetTraining_RGBDiff(info=info, basep=basep, isDynamic=True,
                                       height=int(216), width=int(384))
    train_loader = DataLoaderX(trainset, batch_size=train_batch,
                               shuffle=True, num_workers=2, drop_last=True, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # opt.resume = True
    # init_epoch = 11
    # opt.load = './save_cls_v1/model_best_epoch_loss.pkl'

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE is True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(myModel(depth=depth, group=opt.group).to(device), device_ids=device_ids)
        # if opt.resume:
        #     # load from classification model
        #     cls_model = torch.load(opt.load)
        #     model.module.resnet = cls_model.module.resnet
        #     print('load wieghts from pretrained model')
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))

    else:
        model = myModel(depth=depth, group=opt.group).to(device)
        # if opt.resume:
        #     cls_model = torch.load(opt.load)
        #     model.resnet = cls_model.resnet
        #     print('load wieghts from pretrained model')
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))
    print('number of trainable parameters = ', count_parameters(model))

    # criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=3, cooldown=1, min_lr=8e-5)

    savep = opt.save
    if not os.path.exists(savep):
        os.mkdir(savep)

    if opt.resume and os.path.exists(opt.load):
        # state = {'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, '{savep}/model_at_{epoch:03d}_epoch.pkl'.format(savep=savep, epoch=epoch))

        checkpoint = torch.load(opt.load)
        print("loading checkpoint from EPOCH {}".format(checkpoint['epoch']))

        save_state_dict = checkpoint['state_dict']

        if torch.cuda.device_count() > 1 and MULTI_GPU_MODE is True:

            model_state = model.module.state_dict()
            model_state_list = model_state.keys()
            for key in save_state_dict.keys():
                if key in model_state_list:
                    model_state[key] = save_state_dict[key]
                else:
                    print('key [{}] is not in the model'.format(key))
            model.module.load_state_dict(model_state)
        else:
            model_state = model.state_dict()
            model_state_list = model_state.keys()
            for key in save_state_dict.keys():
                if key in model_state_list:
                    model_state[key] = save_state_dict[key]
                else:
                    print('key [{}] is not in the model'.format(key))
            model.load_state_dict(model_state)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # init_epoch = init_epoch if init_epoch > 0 else checkpoint['epoch'] + 1
    print('init_epoch: {:03d}'.format(init_epoch))
    train_model(model, device, optimizer, scheduler, train_loader, test_loaders, opt, init_epoch=init_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Pretraining for HEKE-BVQA on WELL-Set with resnet')
    parser.add_argument('--model_depth', default=50, type=int,
                        help='model: ResNet{10|18|34|50}. Default to 50')  # we use ResNet18 in the work
    parser.add_argument('--resume', action='store_true', help='enable resume model to train')
    parser.add_argument('--init_epoch', default=0, type=int,
                        help='set initial epoch number (only work when `resume` is set)')
    parser.add_argument('--load', default=None, type=str,
                        help='full name of model to load (only work when `resume` is set')
    parser.add_argument('--save', default='save_model', type=str, help='full name of model to save')

    parser.add_argument('--group', default=1, type=int, help='ensemble groups for each method (default: 1)')
    parser.add_argument('--epsilon', default=0, type=float,
                        help='epsilon for NCL loss ([1e-3, 1e-2], default: 0)')

    parser.add_argument('--epoch', default=100, type=int, help=' training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--train_batch', default=2, type=int, help='train batch size')
    parser.add_argument('--test_batch', default=1, type=int, help='test batch size')
    parser.add_argument('--steps', default=1000, type=int, help='steps to print')

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization (default: 1e-4)')
    parser.add_argument('--seed', default=-1, type=int, help='seed for reproducibility (-1 means randomness)')

    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')
    parser.add_argument('--dynamic', action='store_true', help='enable dynamic training data')

    return parser.parse_args()


if __name__ == '__main__':

    opt = parse_args()
    print(opt)
    run(opt)



