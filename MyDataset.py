import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import random
from scipy import ndimage
from scipy.io import loadmat
import cv2


def Tensor(img):  # from numpy to tensor
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.
    return img


def Array(img):
    img = img[0, 0].detach().numpy()  # from tensor to numpy
    # img = img.transpose(1, 2, 0) # [c,m,n]->[m,n,c]
    return img


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)
    # elif mode == 8:
    #     gamma = np.random.rand(1)
    #     image = image ** gamma
    #     return image
    # elif mode == 9:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     (h, s, v) = cv2.split(image)
    #     hV = cv2.equalizeHist(v)
    #     result = cv2.merge((h, s, hV))
    #     image = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    #     return image
    # elif mode == 10:
    #     (b, g, r) = cv2.split(image)
    #     bH = cv2.equalizeHist(b)
    #     gH = cv2.equalizeHist(g)
    #     rH = cv2.equalizeHist(r)
    #     result = cv2.merge((bH, gH, rH))
    #     return result


class MyDataset(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        need_names  (str): 'yes' for outputting image names, 'no' else
    """

    def __init__(self, rand_mode, patch_size=160, folder='/path/to/folder/'):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset, self).__init__()
        # self.train_low_folder, self.train_high_folder, self.train_snr_folder = folder
        self.train_low_folder, self.train_high_folder = folder
        # print(self.train_low_folder, self.train_high_cartoon_folder, self.train_low_cartoon_folder, self.train_high_folder)
        self.train_file_names = glob.glob(self.train_low_folder + '*.*')
        self.train_file_list = [os.path.join(self.train_low_folder, i) for i in self.train_file_names]
        self.num = 0
        self.patch_size = patch_size
        self.rand_mode = rand_mode

    # def __getitem__(self, index):
    #     """
    #     Loads and transforms an image.
    #     Parameters
    #     ----------
    #         index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
    #                      If the image has just one channel the function will convert it to an RGB format by
    #                      repeating the channel.
    #     Returns
    #     -------
    #                       (str): optional, image name without the extension
    #         (torch.FloatTensor): image before transformation, size c*h*w
    #         (torch.FloatTensor): image after transformation, size c*h*w
    #     """
    #     # .jpg or .png file
    #     low_img = Image.open(self.train_file_list[index])
    #     # low_img = cv2.imread(self.train_file_list[index])
    #     # low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2HSV)
    #     low_img = np.asarray(low_img)
    #     # low_img = (low_img - np.min(low_img)) / (np.max(low_img) - np.min(low_img))
    #     name = os.path.basename(self.train_file_names[index])[:-4].split('\\')[-1]
    #
    #     high_path = self.train_high_folder + name + '.png'
    #     high_img = Image.open(high_path)
    #     # high_img = cv2.imread(high_path)
    #     # high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2HSV)
    #     high_img = np.asarray(high_img)
    #     # high_img = (high_img - np.min(high_img)) / (np.max(high_img) - np.min(high_img))
    #
    #     # snr_path = self.train_snr_folder + name + '.png'
    #     # snr_img = Image.open(snr_path)
    #     # snr_img = np.asarray(snr_img)
    #
    #     h, w, _ = low_img.shape
    #
    #     x = random.randint(0, h - self.patch_size)
    #     y = random.randint(0, w - self.patch_size)
    #     input_low = Tensor(data_augmentation(
    #         low_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
    #     input_high = Tensor(data_augmentation(
    #         high_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
    #     # input_snr = Tensor(data_augmentation(
    #     #     snr_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
    #     # return name, input_low, input_high, input_snr
    #     return name, input_low, input_high
    def __getitem__(self, index):
        # --------------- 原代码（问题代码）---------------
        # low_img = Image.open(self.train_file_list[index])
        # low_img = np.asarray(low_img)
        # high_img = Image.open(high_path)
        # high_img = np.asarray(high_img)

        # --------------- 修改后代码（正确加载）---------------
        # 1. 用cv2加载BGR格式图像（匹配数据集保存格式）
        name = os.path.basename(self.train_file_names[index])[:-4].split('\\')[-1]
        low_img = cv2.imread(self.train_file_list[index])
        high_path = self.train_high_folder + name + '.png'
        if low_img is None:
            raise FileNotFoundError(f"读取低光图失败：{self.train_file_list[index]}")
        high_img = cv2.imread(high_path)
        if high_img is None:
            raise FileNotFoundError(f"读取正常光图失败：{high_path}")

        # 2. 将BGR转为RGB（匹配模型对RGB通道的期望）
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # 新增：确保像素值在0-255（避免读取异常图像导致颜色偏移）
        low_img = np.clip(low_img, 0, 255).astype(np.uint8)
        high_img = np.clip(high_img, 0, 255).astype(np.uint8)

        # 在读取图像后添加归一化
        low_img = np.asarray(low_img) / 255.0  # 假设图像是0-255的uint8格式
        high_img = np.asarray(high_img) / 255.0

        # 2. Tensor转换时，确保归一化正确（除以255后在0-1范围）
        def Tensor(img):
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0  # 明确用255.0避免整数除法
            return img

        # 后续裁剪、数据增强、Tensor转换逻辑不变...
        h, w, _ = low_img.shape
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        input_low = Tensor(data_augmentation(
            low_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
        input_high = Tensor(data_augmentation(
            high_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
        return name, input_low, input_high

    def __len__(self):
        return len(self.train_file_list)


class LOLDataset(Dataset):
    """基础LOL数据集（测试用）"""

    def __init__(self, low_dir, normal_dir, transform):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(low_dir) if f.endswith((".png", ".jpg"))]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_img = Image.open(os.path.join(self.low_dir, filename)).convert("RGB")
        normal_img = Image.open(os.path.join(self.normal_dir, filename)).convert("RGB")
        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)
        return low_img, normal_img, filename

    def __len__(self):
        return len(self.filenames)


# img = np.asarray(Image.open('/home/www/myRetinex/data/LOL/eval15/low/23.png'))
# he_img = data_augmentation(img, mode=10)
# he_img = cv2.cvtColor(he_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('./he_img.png', he_img)


class MyDataset_v2(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        need_names  (str): 'yes' for outputting image names, 'no' else
    """

    def __init__(self, rand_mode, patch_size=160, folder='/path/to/folder/'):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset_v2, self).__init__()
        self.train_low_folder, self.train_high_folder = folder
        # print(self.train_low_folder, self.train_high_cartoon_folder, self.train_low_cartoon_folder, self.train_high_folder)
        self.train_file_names = glob.glob(self.train_low_folder + '*.*')
        self.train_file_list = [os.path.join(self.train_low_folder, i) for i in self.train_file_names]
        self.num = 0
        self.patch_size = patch_size
        self.rand_mode = rand_mode

    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
                         If the image has just one channel the function will convert it to an RGB format by
                         repeating the channel.
        Returns
        -------
                          (str): optional, image name without the extension
            (torch.FloatTensor): image before transformation, size c*h*w
            (torch.FloatTensor): image after transformation, size c*h*w
        """
        # .jpg or .png file
        low_img = Image.open(self.train_file_list[index])
        # low_img = cv2.imread(self.train_file_list[index])
        # low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2HSV)
        low_img = np.asarray(low_img)
        # low_img = (low_img - np.min(low_img)) / (np.max(low_img) - np.min(low_img))
        name = os.path.basename(self.train_file_names[index])[:-4].split('\\')[-1]
        gt_name = name.split('w')[-1]

        high_path = self.train_high_folder + 'normal' + gt_name + '.png'
        high_img = Image.open(high_path)
        # high_img = cv2.imread(high_path)
        # high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2HSV)
        high_img = np.asarray(high_img)
        # high_img = (high_img - np.min(high_img)) / (np.max(high_img) - np.min(high_img))

        h, w, _ = low_img.shape
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        input_low = Tensor(data_augmentation(
            low_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
        input_high = Tensor(data_augmentation(
            high_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))

        return name, input_low, input_high

    def __len__(self):
        return len(self.train_file_list)


if __name__ == '__main__':
    train_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
                    "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"]
    batch_size = 10
    train_Data = []
    for patch_id in range(10):
        rand_mode = np.random.randint(0, 7)
        train_data = MyDataset(rand_mode, patch_size=128, folder=train_folder)
        train_Data.extend(train_data)
        # train_Data.extend(MyDataset(rand_mode=8, folder=train_folder))

    print('[*] Number of training data: %d' % len(train_Data))
    numBatch = len(train_Data) // int(batch_size)
    dataloader = DataLoader(dataset=train_Data, batch_size=batch_size, shuffle=True, num_workers=0,
                            drop_last=True)
    # filename, name, input_low, input_high = MyDataset_lolblur(0, patch_size=128, folder=train_folder)
    # print(filename, name, input_low, input_high)
#     train_low_folder, train_high_folder = train_folder
#     path_list = []
#     filenames = os.listdir(train_low_folder)
#     train_file_list = []
#     for filename in filenames:
#         low_folder = glob.glob(train_low_folder + filename + '/*.*')
#         for pathmat in low_folder:
#             train_file_list.append(pathmat)
#     # for filename in filenames:
#     #     low_folder = glob.glob(train_low_folder + filename + '/*.*')
#     name = os.path.dirname(train_file_list[0])
#     # print(train_high_folder + name.split('/')[-1] + '/' + os.path.basename(train_file_list[0]))
#     # train_file_names = glob.glob(train_low_folder + '*.*')
#     # # name = os.path.basename(train_file_names[0])
#     # print(train_file_names)
