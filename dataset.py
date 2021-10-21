import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
from torchvision import transforms as tf


class VideoDataset(Dataset):
    r"""
    A Dataset for a folder of videos

    args:
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
    """

    def __init__(self, data, data_path='/mnt/diskplus/datasets/images4database_downscale', mode='train'):

        self.ref = data['ref']
        self.dis = data['dis']
        self.mos = data['mos']
        # self.framerate = data['fps']
        self.frms = data['frms']
        self.frame_height = data['height']
        self.frame_width = data['width']
        self.n_data = len(data['mos'])
        self.mode = mode
        if mode == 'train':
            self.n_data *= 10

        if 'data' in data.keys():
            self.data = data['data']
        else:
            self.data = None

    def get_batch(self, index):

        file = self.dis[index][:-4]
        nfrms = self.frms[index]
        ts = 2 if nfrms < 480 else 4

        N = 10  # sample 10 snippets for a 10-second video
        T = 8   # each snippet contains 8 frames

        seq_stride = (nfrms - 20 - T * ts) / (N - 1)
        seq = np.asarray(np.asarray(range(N)) * seq_stride + 15).astype('int')

        vid = None
        for i in range(N):
            for j in range(T):
                pos = seq[i] + j * ts
                img = imread('{file}_{no:03d}.png'.format(file=file, no=pos))

                img = self.transform(img)
                if vid is None:
                    ch, h, w = img.shape
                    vid = np.zeros((N, ch, T, h, w), dtype=np.float32)
                vid[i, :, j, :, :] = img
        return vid

    def transform(self, img):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        img = np.asarray(img).astype('float32')
        img /= 255.
        h, w, D = img.shape

        img = np.transpose(img, (2, 0, 1))

        for i in range(D):
            img[i] -= mean_vals[i]
            img[i] /= std_vals[i]

        return img

    def __getitem__(self, index):
        vi = index // 10
        si = index - vi * 10
        if self.mode == 'train':
            if self.data is None:
                vid = self.get_batch(vi)
            else:
                vid = self.data[vi]

            vid = vid[si]
            mos = self.mos[vi]
        else:
            if self.data is None:
                vid = self.get_batch(index)
            else:
                vid = self.data[index]

            mos = self.mos[index]

        return (vid, mos)

    # def get_train_batch(self, index):
    #     if index % 2 == 0:
    #         is_flip = True
    #     else:
    #         is_flip = False
    #
    #     index = index // 2
    #
    #     i_vid = index // self.n_frm
    #     i_frm = index - i_vid * self.n_frm
    #
    #     frms = self.frms[i_vid]
    #     step = (frms - 8) // (self.n_frm - 1)
    #     i_frm = frms - 4 - (self.n_frm - i_frm - 1) * step
    #
    #     ref = self.ref[i_vid]
    #     dis = self.dis[i_vid]
    #     label = float(self.label[i_vid])
    #     framerate = int(self.framerate[i_vid])
    #     frame_height = int(self.frame_height[i_vid])
    #     frame_width = int(self.frame_width[i_vid])
    #
    #     if framerate <= 30:
    #         stride_t = 1
    #     elif framerate <= 60:
    #         stride_t = 2
    #     else:
    #         raise ValueError('Unsupported fps')
    #
    #     ref = self.load_yuv(ref, frame_height, frame_width, stride_t, start=i_frm, is_flip=is_flip)
    #     dis = self.load_yuv(dis, frame_height, frame_width, stride_t, start=i_frm, is_flip=is_flip)
    #
    #     # if ref.endswith(('.YUV', '.yuv')):
    #     #     ref = self.load_yuv(ref, frame_height, frame_width, stride_t)
    #     # elif ref.endswith(('.mp4')):
    #     #     ref = self.load_encode(ref, frame_height, frame_width, stride_t)
    #     # else:
    #     #     raise ValueError('Unsupported video format')
    #
    #     # if dis.endswith(('.YUV', '.yuv')):
    #     #     # dis = self.load_yuv(dis, frame_height, frame_width, stride_t)
    #     #     dis = skvideo.io.vread(dis, frame_height, frame_width,
    #     #                                   inputdict={'-pix_fmt': 'yuvj420p'})
    #     #
    #     # elif dis.endswith(('.mp4')):
    #     #     # dis = self.load_encode(dis, frame_height, frame_width, stride_t)
    #     #     dis = skvideo.io.vread(dis)
    #     # else:
    #     #     raise ValueError('Unsupported video format')
    #
    #     ##  frm
    #     x = dis[1] / 255
    #
    #     ##  dif
    #     y = np.stack(((dis[1]-dis[0])/255, (dis[1]-dis[2])/255))
    #
    #     ##  gm
    #     # gx = np.abs(convolve(dis, dx, mode='valid'))
    #     # gy = np.abs(convolve(dis, dy, mode='valid'))
    #     # gt = np.abs(convolve(dis, dt, mode='valid'))
    #     # z = np.sqrt(gx*gx + gy*gy + gt*gt)
    #     # z = np.pad(z, ((0, 0), (1, 1), (1, 1)), 'edge')
    #
    #
    #     with torch.no_grad():
    #         gxc = self.convx(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #         gyc = self.convy(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #         gtc = self.convt(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #         zc = torch.sqrt(gxc*gxc + gyc*gyc + gtc*gtc)
    #         zc = F.pad(zc, (1, 1, 1, 1, 0, 0), 'replicate')
    #
    #     ## stGMSD
    #     # gx = np.abs(convolve(ref, dx, mode='valid'))
    #     # gy = np.abs(convolve(ref, dy, mode='valid'))
    #     # gt = np.abs(convolve(ref, dt, mode='valid'))
    #     # z0 = np.sqrt(gx*gx + gy*gy + gt*gt)
    #     # z0 = np.pad(z0, ((0, 0), (1, 1), (1, 1)), 'edge')
    #
    #         gxc = self.convx(torch.from_numpy(ref[np.newaxis, np.newaxis, :]))
    #         gyc = self.convy(torch.from_numpy(ref[np.newaxis, np.newaxis, :]))
    #         gtc = self.convt(torch.from_numpy(ref[np.newaxis, np.newaxis, :]))
    #         zc0 = torch.sqrt(gxc*gxc + gyc*gyc + gtc*gtc)
    #         zc0 = F.pad(zc0, (1, 1, 1, 1, 0, 0), 'replicate')
    #
    #         simi = (2 * zc * zc0 + 255) / (zc**2 + zc0**2 + 255)
    #     # simi = convolve2d(simi[0], np.ones((4, 4)).astype('float32') / 16, mode='valid')
    #     # simi = simi[np.newaxis, ::4, ::4]
    #         weak_label = torch.std(simi)
    #
    #     # offset_v = (frame_height - self.size_y) % self.stride_y
    #     # offset_t = int(offset_v / 4 * 2)
    #     # offset_b = offset_v - offset_t
    #     # offset_h = (frame_width - self.size_x) % self.stride_x
    #     # offset_l = int(offset_h / 4 * 2)
    #     # offset_r = offset_h - offset_l
    #
    #     # ref = ref[:, :, offset_t:frame_height-offset_b, offset_l:frame_width-offset_r]
    #     # dis = dis[:, :, offset_t:frame_height-offset_b, offset_l:frame_width-offset_r]
    #
    #     # spatial_crop = CropSegment(self.size_x, self.size_y, self.stride_x, self.stride_y)
    #     # ref = spatial_crop(ref)
    #     # dis = spatial_crop(dis)
    #
    #     # ref = torch.from_numpy(np.asarray(ref))
    #     # dis = torch.from_numpy(np.asarray(dis))
    #
    #     # label = torch.from_numpy(np.asarray(label))
    #     # feat = self.feat[index]
    #
    #     return x[np.newaxis, :], y, zc[0][0], weak_label
    # #
    # def get_test_batch(self, index):
    #     i_vid = index
    #
    #     frms = self.frms[i_vid]
    #     step = (frms - 8) // (self.n_frm - 1)
    #
    #     ref_file = self.ref[i_vid]
    #     dis_file = self.dis[i_vid]
    #     label = float(self.label[i_vid])
    #     framerate = int(self.framerate[i_vid])
    #     frame_height = int(self.frame_height[i_vid])
    #     frame_width = int(self.frame_width[i_vid])
    #
    #     if framerate <= 30:
    #         stride_t = 1
    #     elif framerate <= 60:
    #         stride_t = 2
    #     else:
    #         raise ValueError('Unsupported fps')
    #
    #     x0, y0, z0 = [], [], []
    #     for i_frm in range(self.n_frm):
    #         this_frm = frms - 4 - (self.n_frm - i_frm - 1) * step
    #         dis = self.load_yuv(dis_file, frame_height, frame_width, stride_t, start=this_frm)
    #
    #         # frm
    #         x = dis[1] / 255
    #         # dif
    #         y = np.stack(((dis[1] - dis[0]) / 255, (dis[1] - dis[2]) / 255))
    #         # gm
    #         with torch.no_grad():
    #             gxc = self.convx(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #             gyc = self.convy(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #             gtc = self.convt(torch.from_numpy(dis[np.newaxis, np.newaxis, :]))
    #             zc = torch.sqrt(gxc * gxc + gyc * gyc + gtc * gtc)
    #             zc = F.pad(zc, (1, 1, 1, 1, 0, 0), 'replicate')
    #
    #         x0.append(x[np.newaxis, :])
    #         y0.append(y)
    #         z0.append(zc[0][0])
    #
    #     x0 = torch.from_numpy(np.stack(x0))
    #     y0 = torch.from_numpy(np.stack(y0))
    #     z0 = torch.from_numpy(np.stack(z0))
    #
    #     label = torch.from_numpy(np.asarray(label).astype('float32'))
    #     # feat = self.feat[index]
    #
    #     # return x[np.newaxis, :], y, zc[0][0], label
    #     return x0, y0, z0, label
    #
    #
    # def load_yuv(self, file_path, frame_height, frame_width, stride_t, start=0, is_flip=False):
    #     r"""
    #     Load frames on-demand from raw video, currently supports only yuv420p
    #
    #     args:
    #         file_path (str): path to yuv file
    #         frame_height
    #         frame_width
    #         stride_t (int): sample the 1st frame from every stride_t frames
    #         start (int): index of the 1st sampled frame
    #     return:
    #         ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)
    #     """
    #
    #     bytes_per_frame = int(frame_height * frame_width * 1.5)
    #     # frame_count = os.path.getsize(file_path) / bytes_per_frame
    #
    #     ret = []
    #     # count = 0
    #     assert start > 1
    #
    #     scale = frame_height // 256 + 1
    #     ope = np.ones((scale, scale)).astype('float32') / (scale ** 2)
    #     with open(file_path, 'rb') as f:
    #         for frm in [start-stride_t, start, start+stride_t]:
    #             offset = frm * bytes_per_frame
    #             f.seek(offset, 0)
    #             frame = f.read(frame_height * frame_width)
    #             frame = np.frombuffer(frame, "uint8")
    #             frame = frame.astype('float32')
    #             frame = frame.reshape(frame_height, frame_width)
    #
    #             if scale > 1:
    #                 frame = convolve2d(frame, ope, 'valid')
    #                 frame = frame[::scale, ::scale]
    #
    #             if is_flip:
    #                 frame = np.fliplr(frame)
    #
    #             ret.append(frame)
    #             # count += 1
    #     ret = np.stack(ret)
    #     # ret = np.concatenate(ret, axis=1)
    #     # ret = torch.from_numpy(np.asarray(ret))
    #
    #     return ret
    #
    #
    # def load_yuv0(self, file_path, frame_height, frame_width, stride_t, start=0):
    #     r"""
    #     Load frames on-demand from raw video, currently supports only yuv420p
    #
    #     args:
    #         file_path (str): path to yuv file
    #         frame_height
    #         frame_width
    #         stride_t (int): sample the 1st frame from every stride_t frames
    #         start (int): index of the 1st sampled frame
    #     return:
    #         ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)
    #     """
    #
    #     bytes_per_frame = int(frame_height * frame_width * 1.5)
    #     frame_count = os.path.getsize(file_path) / bytes_per_frame
    #
    #     ret = []
    #     count = 0
    #
    #     with open(file_path, 'rb') as f:
    #         while count < frame_count:
    #             if count % stride_t == 0:
    #                 offset = count * bytes_per_frame
    #                 f.seek(offset, 0)
    #                 frame = f.read(frame_height * frame_width)
    #                 frame = np.frombuffer(frame, "uint8")
    #                 frame = frame.astype('float32') / 255.
    #                 frame = frame.reshape(1, 1, frame_height, frame_width)
    #                 ret.append(frame)
    #             count += 1
    #
    #     ret = np.concatenate(ret, axis=1)
    #     ret = torch.from_numpy(np.asarray(ret))
    #
    #     return ret
    #
    # def load_encode(self, file_path, frame_height, frame_width, stride_t, start=0):
    #     r"""
    #     Load frames on-demand from encode bitstream
    #
    #     args:
    #         file_path (str): path to yuv file
    #         frame_height
    #         frame_width
    #         stride_t (int): sample the 1st frame from every stride_t frames
    #         start (int): index of the 1st sampled frame
    #     return:
    #         ret (array): contains sampled frames. dim = (C, D, H, W)
    #     """
    #
    #     enc_path = file_path
    #     enc_name = re.split('/', enc_path)[-1]
    #
    #     yuv_name = enc_name.replace('.mp4', '.yuv')
    #     yuv_path = os.path.join('/dockerdata/tmp/', yuv_name)
    #     cmd = "ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}".format(src=enc_path, dst=yuv_path)
    #     subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    #     ret = self.load_yuv(yuv_path, frame_height, frame_width, stride_t, start=0)
    #
    #     return ret

    def __len__(self):
        # if self.mode == 'train':
        #     return self.n_data * self.n_frm * 2
        # elif self.mode == 'test':
        #     return self.n_data
        return self.n_data


class DatasetTraining(Dataset):
    r"""
    A Training Dataset for a folder of videos

    args:
        info (str): pickle file containinng the information of the dataset (dis_name, label, ...)
        basep (str): PATH to the dataset
    """

    def __init__(self, info, basep='/mnt/diskplus/datasets/images4database_downscale',
                 isDynamic=True,
                 height=216, width=384):

        self.basep = basep

        self.nfrm = 8
        with open(info, 'rb') as f:
            self.info = pickle.load(f)

        self.isDynamic = isDynamic
        self.n_data = len(self.info)
        if not isDynamic:
            self.fixList = np.random.randint(low=0, high=4, size=self.n_data)

        self.fixH, self.fixW = height, width

        self.transform = tf.Compose([
            tf.Resize([height, width]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_batch(self, file, pos):
        N = self.nfrm
        ts = 2

        vid = np.zeros((3, N, self.fixH, self.fixW), dtype=np.float32)
        for i in range(N):
            img = Image.open('{file}.{no:04d}.png'.format(file=file, no=pos))
            if img.height > img.width:
                img = img.transpose(Image.TRANSPOSE)
            img = self.transform(img)
            vid[:, i, :, :] = np.copy(img)

            pos += ts
        return vid

    def suppress(self, x, scale=0.1):
        """
        suppress MOS less than 0.5 or larger than 10.
        The ideal MOS range should be [1, 9.5]
        :param x: predicted score
        :return: transformed score
        """

        if x < 0.5:
            y = 0.5 - scale * (0.5 - x)
        elif x > 10:
            y = 10 + scale * (x - 10)
        else:
            y = x
        return y

    def __getitem__(self, index):
        info = self.info[index]
        if self.isDynamic:
            segi = np.random.randint(low=0, high=4)
        else:
            segi = self.fixList[index]

        dis = self.get_batch(file=info['file'], pos=info['seg'][segi])

        #          | [       min        ,         max       ,       mean        ]
        # msssim   | [0.2584740194470969, 0.9999975358135895, 0.9691534338080047]
        # gmsd     | [0.0000100433512947, 0.3269521051066762, 0.0406634450288763]
        # stgmsd   | [0.0000058711396248, 0.3591311676958205, 0.0318362180366851]
        # strred   | [0.0001390788958243, 8571.664268529374,  39.65565327736158 ]
        # vmaf     | [0.0,                100.0,              79.84541340532772 ]
        #          |  normalizing to [0.1, 9] is expected

        msssim = info['msssim'][segi]
        gmsd = info['gmsd'][segi]
        stgmsd = info['stgmsd'][segi]
        strred = info['strred'][segi]
        vmaf = info['vmaf'][segi]

        # msssim = 0.5 + 8.5 * (1 - (msssim - 0.258) / (1 - 0.258))
        # gmsd = (gmsd - 0) / (0.33 - 0) * 8.5 + 0.5
        # stgmsd = (stgmsd - 0) / (0.36 - 0) * 8.5 + 0.5
        # strred = (strred - 0) / (1000 - 0) * 8.5 + 0.5
        # vmaf = (vmaf - 0) / (100 - 0) * 8.5 + 0.5

        # supress
        msssim = self.suppress(msssim)
        gmsd = self.suppress(gmsd)
        stgmsd = self.suppress(stgmsd)
        strred = self.suppress(strred)
        vmaf = self.suppress(vmaf)
        return dis, \
               msssim.astype('float32'), \
               gmsd.astype('float32'), \
               stgmsd.astype('float32'), \
               strred.astype('float32'), \
               vmaf.astype('float32')

    def __len__(self):
        return self.n_data


class DatasetTraining_RGBDiff(Dataset):
    r"""
    A Training Dataset for a folder of videos

    args:
        info (str): pickle file containinng the information of the dataset (dis_name, label, ...)
        basep (str): PATH to the dataset
    """

    def __init__(self, info, basep='/mnt/diskplus/datasets/images4database_downscale',
                 isDynamic=True,
                 height=216, width=384):

        self.basep = basep

        self.nfrm = 8
        with open(info, 'rb') as f:
            self.info = pickle.load(f)

        self.isDynamic = isDynamic
        self.n_data = len(self.info)
        if not isDynamic:
            self.fixList = np.random.randint(low=0, high=8, size=self.n_data)

        self.fixH, self.fixW = height, width

        self.transform = tf.Compose([
            tf.Resize([height, width]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_batch(self, file, pos):
        ts = 2
        im1 = Image.open('{file}.{no:04d}.png'.format(file=file, no=pos))
        im2 = Image.open('{file}.{no:04d}.png'.format(file=file, no=pos+ts))
        if im1.height > im1.width:
            im1 = im1.transpose(Image.TRANSPOSE)
            im2 = im2.transpose(Image.TRANSPOSE)

        im1 = self.transform(im1)
        im2 = self.transform(im2)

        return im1, im2 - im1

    def suppress(self, x, scale=0.1):
        """
        suppress MOS less than 0.5 or larger than 10.
        The ideal MOS range should be [1, 9.5]
        :param x: predicted score
        :return: transformed score
        """

        if x < 0.5:
            y = 0.5 - scale * (0.5 - x)
        elif x > 10:
            y = 10 + scale * (x - 10)
        else:
            y = x
        return y

    def __getitem__(self, index):
        info = self.info[index]
        if self.isDynamic:
            segi = np.random.randint(low=0, high=8)
        else:
            segi = self.fixList[index]

        dis, dif = self.get_batch(file=info['file'], pos=info['seg'][segi])

        #          | [       min        ,         max       ,       mean        ]
        # msssim   | [0.2584740194470969, 0.9999975358135895, 0.9691534338080047]
        # gmsd     | [0.0000100433512947, 0.3269521051066762, 0.0406634450288763]
        # stgmsd   | [0.0000058711396248, 0.3591311676958205, 0.0318362180366851]
        # strred   | [0.0001390788958243, 8571.664268529374,  39.65565327736158 ]
        # vmaf     | [0.0,                100.0,              79.84541340532772 ]
        #          |  normalizing to [0.1, 9] is expected

        msssim = info['msssim'][segi]
        gmsd = info['gmsd'][segi]
        stgmsd = info['stgmsd'][segi]
        strred = info['strred'][segi]
        vmaf = info['vmaf'][segi]

        # msssim = 0.5 + 8.5 * (1 - (msssim - 0.258) / (1 - 0.258))
        # gmsd = (gmsd - 0) / (0.33 - 0) * 8.5 + 0.5
        # stgmsd = (stgmsd - 0) / (0.36 - 0) * 8.5 + 0.5
        # strred = (strred - 0) / (1000 - 0) * 8.5 + 0.5
        # vmaf = (vmaf - 0) / (100 - 0) * 8.5 + 0.5

        # supress
        msssim = self.suppress(msssim)
        gmsd = self.suppress(gmsd)
        stgmsd = self.suppress(stgmsd)
        strred = self.suppress(strred)
        vmaf = self.suppress(vmaf)
        return dis, dif,\
               msssim.astype('float32'), \
               gmsd.astype('float32'), \
               stgmsd.astype('float32'), \
               strred.astype('float32'), \
               vmaf.astype('float32')

    def __len__(self):
        return self.n_data


class DatasetValidation(Dataset):
    r"""
    A Validation Dataset for a folder of videos

    args:
        info (str): pickle file containinng the information of the dataset (dis_name, label, ...)
        basep (str): PATH to the dataset
    """

    def __init__(self, dataset='LIVE', basep='/mnt/diskplus/datasets/images4database_downscale',
                 height=216, width=384, nsnippet=12):

        self.basep = basep
        self.dataset = dataset

        # pre-load the test data for fast evaluation without data preparing
        self.nfrm = 8
        self.nsnippet = nsnippet
        self.n_data, self.info = self.get_test_info(dataset)
        # self.data = self.load_test_data()
        # self.n_data = len(self.data)

        self.fixH, self.fixW = height, width
        self.transform = tf.Compose([
            tf.Resize([height, width]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_batch(self, file, frms):
        ts = 2 if frms < 480 else 4
        N = self.nfrm
        T = self.nsnippet
        offset = 10

        stride = (frms - offset * 2 - ts * N) / (T - 1)
        pos = np.asarray(range(T)) * stride + offset
        pos = pos.astype('int')

        vid = np.zeros((T, 3, N, self.fixH, self.fixW), dtype=np.float32)
        for i in range(T):
            thispos = pos[i]
            for j in range(N):
                no_id = thispos + ts * j
                img = Image.open('{file}_{no:03d}.png'.format(file=file, no=no_id))
                if img.height > img.width:
                    img = img.transpose(Image.TRANSPOSE)
                img = self.transform(img)

                vid[i, :, j] = np.copy(img)

        return vid

    def get_test_info(self, dataset):
        info_file = '{}/{}/{}_list_for_VQA.txt'.format(self.basep, dataset, dataset)
        # info_file = '/data/sissuire/datasets/images4database_downscale/{}/{}_list_for_VQA.txt'.format(dataset, dataset)
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
                                 'frms': frms_list}

    def __getitem__(self, index):
        file = self.info['dis'][index]
        frms = self.info['frms'][index]
        dis = self.get_test_batch(file, frms)
        mos = self.info['mos'][index]
        # data = np.stack([ref, dis], axis=0)
        # label = np.asarray([[1, 0], [0, 1]]).astype('float32')
        # return data, label
        return dis, mos

    def __len__(self):
        return self.n_data


class DatasetValidation_RGBDiff(Dataset):
    r"""
    A Validation Dataset for a folder of videos

    args:
        info (str): pickle file containinng the information of the dataset (dis_name, label, ...)
        basep (str): PATH to the dataset
    """

    def __init__(self, dataset='LIVE', basep='/mnt/diskplus/datasets/images4database_downscale',
                 height=216, width=384, nsnippet=20):

        self.basep = basep
        self.dataset = dataset

        # pre-load the test data for fast evaluation without data preparing
        # self.nfrm = 1
        self.nsnippet = nsnippet
        self.n_data, self.info = self.get_test_info(dataset)
        # self.data = self.load_test_data()
        # self.n_data = len(self.data)

        self.fixH, self.fixW = height, width
        self.transform = tf.Compose([
            tf.Resize([height, width]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_batch(self, file, frms):
        ts = 2 if frms < 480 else 4
        # N = self.nfrm
        T = self.nsnippet
        offset = 10

        stride = (frms - offset * 2 - ts) / (T - 1)
        pos = np.asarray(range(T)) * stride + offset
        pos = pos.astype('int')

        vid = np.zeros((T, 3, self.fixH, self.fixW), dtype=np.float32)
        vdd = np.zeros((T, 3, self.fixH, self.fixW), dtype=np.float32)
        for i in range(T):
            thispos = pos[i]

            im1 = Image.open('{file}_{no:03d}.png'.format(file=file, no=thispos))
            im2 = Image.open('{file}_{no:03d}.png'.format(file=file, no=thispos + ts))
            if im1.height > im1.width:
                im1 = im1.transpose(Image.TRANSPOSE)
                im2 = im2.transpose(Image.TRANSPOSE)
            im1 = self.transform(im1)
            im2 = self.transform(im2)

            vid[i] = np.copy(im1)
            vdd[i] = np.copy(im2 - im1)
        return vid, vdd

    def get_test_info(self, dataset):
        info_file = '{}/{}/{}_list_for_VQA.txt'.format(self.basep, dataset, dataset)
        # info_file = '/data/sissuire/datasets/images4database_downscale/{}/{}_list_for_VQA.txt'.format(dataset, dataset)
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
                                 'frms': frms_list}

    def __getitem__(self, index):
        file = self.info['dis'][index]
        frms = self.info['frms'][index]
        dis, dif = self.get_test_batch(file, frms)
        mos = self.info['mos'][index]
        # data = np.stack([ref, dis], axis=0)
        # label = np.asarray([[1, 0], [0, 1]]).astype('float32')
        # return data, label
        return dis, dif, mos

    def __len__(self):
        return self.n_data


if __name__ == '__main__':

    # ===============================================================================
    test = ['LIVE', 'CSIQ', 'IVPL']
    testset_LIVE = DatasetValidation(dataset='LIVE')
    testset_CSIQ = DatasetValidation(dataset='CSIQ')
    testset_IVPL = DatasetValidation(dataset='IVPL')
    test_loader_LIVE = DataLoader(testset_LIVE, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    test_loader_CSIQ = DataLoader(testset_CSIQ, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    test_loader_IVPL = DataLoader(testset_IVPL, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    for dis, mos in test_loader_LIVE:
        print(dis.shape)
        print(mos)
        break
    # -------------------------------------------------------------------------------

    # ===============================================================================
    basep = '/mnt/diskplus/datasets/WELL-Set/dis_png'
    trainset = DatasetTraining(info='wellset.info.pkl', basep=basep, preload=False, isDynamic=True)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    for dis, msssim, gmsd, stgmsd, strred, vmaf in train_loader:

        print(dis.shape)
        break

    # database = ['LIVE', 'CSIQ', 'IVPL']
    # database = 'LIVE'
    # db_path = '/mnt/diskplus/datasets/images4database_downscale/{}'.format(database)
    # info_file = '/mnt/disk/yongxu_liu/datasets/{}/{}_list_for_VQA.txt'.format(database, database)

    # ref_list, dis_list, d2r_list, score_list, width_list, height_list, fps_list, frms_list = \
    #     [], [], [], [], [], [], [], []
    # base = '/mnt/disk/yongxu_liu/datasets/{}/'.format(database)
    # with open(info_file, 'r') as f:
    #     for line in f:
    #         scn_idx, dis_idx, ref, dis, score, width, height, frm = line.split()
    #         scn_idx = int(scn_idx)
    #         dis_idx = int(dis_idx)
    #         width = int(width)
    #         height = int(height)
    #         frames = int(frm)
    #
    #         dis_list.append(dis)
    #         ref_list.append(ref)
    #         width_list.append(width)
    #         height_list.append(height)
    #
    #
    # def downscale_img(img, block_size=(2, 2, 1)):
    #     img = skimage.measure.block_reduce(img, block_size=block_size, func=np.mean)
    #     return img
    #
    # for iv in range(len(dis_list)):
    #     file = dis_list[iv]
    #     # file_id = file.split('/')[-1][:-4]
    #     data = skvideo.io.vread(base + file, width=width_list[iv], height=height_list[iv],
    #                             inputdict={"-pix_fmt": "yuv420p"})
    #     frms = data.shape[0]
    #
    #     for ii in range(frms):
    #         img = downscale_img(data[ii], block_size=(5, 5, 1)).astype('uint8')
    #
    #         output = '{s_path}/{s_id}_{s_no:03d}.png'.format(s_path=db_path, s_id=file[:-4], s_no=ii + 1)
    #         imageio.imwrite(output, img)
    #
    # for iv in [8, 20, 35, 50, 60, 75, 88, 100, 110, 125]:
    #     file = ref_list[iv]
    #     # file_id = file.split('/')[-1][:-4]
    #     data = skvideo.io.vread(base + file, width=width_list[iv], height=height_list[iv],
    #                             inputdict={"-pix_fmt": "yuv420p"})
    #     frms = data.shape[0]
    #
    #     for ii in range(frms):
    #         img = downscale_img(data[ii], block_size=(5, 5, 1)).astype('uint8')
    #
    #         output = '{s_path}/{s_id}_{s_no:03d}.png'.format(s_path=db_path, s_id=file[:-4], s_no=ii + 1)
    #         imageio.imwrite(output, img)

    # preset = PreDataset(db_path=db_path, info_file=info_file, database=database, preload=True)
    # trainset = VideoDataset(preset.data, mode='train')
    # train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
    #
    # for vid, mos in train_loader:
    #
    #     print(vid.shape)
    #
    # print('done.')
