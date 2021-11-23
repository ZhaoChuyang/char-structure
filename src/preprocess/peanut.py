import PIL
from PIL import Image, ImageFile
import numpy as np
from copy import copy
from collections import Counter
import math
from torchvision import transforms
import torch
from src.preprocess.util import DrawGaussian
from torch.utils.data.dataset import Dataset
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# https://github.com/NiteshBharadwaj/part-affinity
ImageFile.LOAD_TRUNCATED_IMAGES = True
img_path = 'data/1_1.png'
_num_kpts = 87
_num_connections = 61

# def get_heatmap(img, annotations, all_strokes, sigma):
#     """
#     generate heatmap for input character image
#     :param img: ndarray
#     :param annotations:
#         dict of all stroke annotations.
#         each annotation is a numpy array of size(N, M, 2),
#         N represents the number of strokes,
#         M represents the number of keypoints in this stroke type.
#     :param sigma:
#     :return:
#     """
#     heatmap_all = dict()
#     for stroke in all_strokes:
#         annotation = annotations[stroke]
#         num_strokes = annotation.shape[0]
#         num_keypoints = annotation.shape[1]
#         out_map = np.zeros(num_keypoints+1, img.shape[0], img.shape[1])
#         # 例如: 横包含两个点，num_strokes表示字中含有的横的数量，num_keypoints表示这个笔画含有的点的数量，e.g. 横含有两个点
#         for stroke_id in range(num_strokes):
#             for keypoint_id in annotation[stroke_id]:
#                 keypoint = annotation[stroke_id][keypoint_id]
#                 out_map[keypoint_id] = np.maximum(out_map[keypoint_id], DrawGaussian(out_map[keypoint_id], keypoint, sigma=sigma))
#         out_map[num_strokes] = 1 - np.sum(out_map[0:num_strokes], axis=0)  # Last heatmap is background
#         if stroke not in heatmap_all:
#             heatmap_all[stroke] = copy(out_map)
#     return heatmap_all

class Stroke:
    def __init__(self, stroke_info):
        self.stroke_info = stroke_info
        num_strokes_base = 0
        num_connection_base = 0
        self.kpt_name_to_id = {}
        self.kpt_id_to_name = {}
        self.connection_name_to_id = {}
        self.connection_id_to_name = {}

        for stroke in stroke_info:
            stroke['base'] = num_strokes_base + stroke['strokeOrderLength']
            for id in range(stroke['strokeOrderLength']):
                kpt_name = '{0}_{1}'.format(stroke['id'], id)
                kpt_id = num_strokes_base + id
                self.kpt_name_to_id[kpt_name] = kpt_id
                self.kpt_id_to_name[kpt_id] = kpt_name
            num_strokes_base += stroke['strokeOrderLength']

        for stroke in stroke_info:
            for id in range(stroke['strokeOrderLength']-1):
                connection_name = '{0}_{1}'.format(stroke['id'], id)
                connection_id = num_connection_base + id
                self.connection_name_to_id[connection_name] = connection_id
                self.connection_id_to_name[connection_id] = connection_name
            num_connection_base += stroke['strokeOrderLength']-1

    def get_kpt_id(self, kpt_name):
        return self.kpt_name_to_id[kpt_name]

    def get_stroke_kpt(self, kpt_id):
        kpt_name = self.kpt_id_to_name[kpt_id]
        return kpt_name

    def get_num_connections(self):
        sum_connections = 0
        for stroke in self.stroke_info:
            num_connections = stroke['strokeOrderLength'] - 1
            sum_connections += num_connections
        return sum_connections

    def get_connection_id(self, connection_name):
        connection_id = self.connection_name_to_id[connection_name]
        return connection_id

    def get_connection_name(self, connection_id):
        connection_name = self.connection_id_to_name[connection_id]
        return connection_name


class PeanutClsDataset(Dataset):
    def __init__(self, images_info, imgdir):
        super().__init__()
        self.images_info = images_info
        self.imgdir = imgdir
        self.transforms = self._get_transforms()
        self.char_map = {}
        self.char_counter = 0
        for record in images_info:
            char_id = str(record['cId'])
            if char_id not in self.char_map:
                self.char_map[char_id] = self.char_counter
                self.char_counter += 1


    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        record = self.images_info[idx]
        filename = record['fileName']
        char_id = str(record['cId'])
        # if char_id not in self.char_map:
        #     self.char_map[char_id] = self.char_counter
        #     self.char_counter += 1
        char_id = self.char_map[char_id]
        strokes = record['strokes']
        stroke_map = np.zeros(90)
        for i in strokes:
            stroke_map[i] = 1
        try:
            image = Image.open('%s/%s' % (self.imgdir, filename))
        except:
            return None
        image_ten = self.transforms(image)
        return image_ten, torch.tensor(char_id, dtype=torch.long), filename

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.85, 0.85, 0.85], [0.09, 0.09, 0.09]),
        ])



class ReinforceDataset(Dataset):
    def __init__(self, annotations, images_info, imgdir, stroke_tool, std_skeleton, stride=4, sigma=1, paf_width=1, mode='train', mask=False):
        super().__init__()
        self.threshold = 0.7
        self._stride = stride
        self._sigma = sigma
        self.annotations = annotations
        self.stroke_tool = stroke_tool
        self.paf_width = paf_width
        self.images_info = {}
        self.imgdir = imgdir
        self.mode = mode
        for image_info in images_info:
            self.images_info[str(image_info['fileName'])] = image_info
        self.mask = mask
        self.mask_threshold = 150
        self.std_skeleton = std_skeleton
        skeleton_dict = {}
        for record in std_skeleton:
            char_id = record['char_id']
            skeleton_dict[char_id] = record
        self.skeleton_dict = skeleton_dict

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        record = self.annotations[idx]
        # current_image_id = str(record['currentImageId'])
        # dataset_id = str(record['dataSetId'])
        # image_name = '%s-%s.png' % (dataset_id, current_image_id)
        image_name = record['fileName']
        char_id = record['charId']
        stroke_annotations = record['result']
        # image_info = self.images_info[image_name]

        # skip bad images
        if record['skip']:
            return None

        try:
            sample = {
                'image': Image.open('%s/%s' % (self.imgdir, image_name)).convert('RGB'),
                'image_name': image_name,
                'annotation': stroke_annotations,
            }
        except:
            return None
        mask = self._get_mask(sample)
        heatmap = self._get_heatmap(sample)
        paf_map = self._get_paf_map(sample)
        image_ten, heatmap_ten, vecmap_ten, mask_ten = self.transforms(self.mode, sample['image'], heatmap, paf_map, mask)
        data_dict = {
            'img': image_ten,
            'maskmap': mask_ten,
            'vecmap': vecmap_ten,
            'heatmap': heatmap_ten,
            'filename': image_name,
            'char_id': char_id,
            'std_paf': self.skeleton_dict[char_id]['vecmap'],
            'std_heatmap': self.skeleton_dict[char_id]['heatmap']
        }
        return data_dict


    def transforms(self, mode, image, heatmap, vecmap, mask):
        to_tensor = transforms.ToTensor()

        # convert heatmap, vecmap, mask to tensor
        heatmap = to_tensor(heatmap)
        vecmap = to_tensor(vecmap)
        mask = to_tensor(mask)

        # Apply resize
        resize_image = transforms.Resize(size=(256, 256))
        mask_h, mask_w = 256 // self._stride, 256 // self._stride
        resize_mask = transforms.Resize(size=(mask_h, mask_w))
        image = resize_image(image)
        heatmap = resize_mask(heatmap)
        vecmap = resize_mask(vecmap)
        mask = resize_mask(mask)

        if mode in ['train']:
            # random resized crop
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=[0.8, 1.0], ratio=[0.8, 1.2])
                image = TF.resized_crop(image, i, j, h, w, size=[256, 256])
                mask = TF.resized_crop(mask, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])
                vecmap = TF.resized_crop(vecmap, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])
                heatmap = TF.resized_crop(heatmap, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])

            # random rotation
            if random.random() > 0.5:
                degree = transforms.RandomRotation.get_params(degrees=[0., 20.])
                image = TF.rotate(image, degree, fill=255)
                vecmap = TF.rotate(vecmap, degree)
                heatmap = TF.rotate(heatmap, degree)
                mask = TF.rotate(mask, degree)

        # convert image to 3 channels tensor
        image = to_tensor(image)
        to_gray_scale = transforms.Grayscale(num_output_channels=3)
        image = to_gray_scale(image)
        image = image > self.threshold
        image = image.float()

        # normalize = transforms.Normalize([0.85,0.85,0.85], [0.09,0.09,0.09])
        # image = normalize(image)

        # heatmap = heatmap.half()
        # vecmap = vecmap.half()
        # mask = mask.half()
        heatmap = heatmap.float()
        vecmap = vecmap.float()
        mask = mask.float()

        return image, heatmap, vecmap, mask

    def _get_mask(self, sample):
        h, w = sample['image'].size
        h //= self._stride
        w //= self._stride
        mask = sample['image']
        to_gray = transforms.Grayscale(num_output_channels=1)
        mask = transforms.Resize((h, w))(mask)

        mask = to_gray(mask)
        mask = np.asarray(mask)

        mask = mask < self.mask_threshold
        mask = mask.astype('float')
        # not use mask, return all ones
        if self.mask is False:
            mask = np.ones((h, w))
        return mask

    def _get_heatmap(self, sample):
        '''
        :param sample:
        :return: size is (H x W x C)
        '''
        num_keypoints = _num_kpts
        image = sample['image']
        height, width = image.size
        # last dimension for background
        keypoint_maps = np.zeros(shape=(num_keypoints + 1, height // self._stride, width // self._stride), dtype=np.float32)

        for stroke in sample['annotation']:
            for id, kpt in enumerate(stroke['record']):
                kpt_name = '%s_%s' % (stroke['id'], id)
                kpt_id = self.stroke_tool.get_kpt_id(kpt_name)
                self._draw_gaussian(keypoint_maps[kpt_id], kpt[0], kpt[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        keypoint_maps = np.transpose(keypoint_maps, (1, 2, 0))

        return keypoint_maps

    def _draw_gaussian(self, keypoint_map, x, y, stride, sigma=1, scale=100):
        n_sigma = 10
        map_h, map_w = keypoint_map.shape
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        # top left coordinate
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)
        # bottom right coordinate
        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        # shift = stride / 2
        shift = stride / 2
        moves = [(0, 0), (0, shift), (0, -shift), (shift, 0), (-shift, 0), (shift, shift), (shift, -shift), (-shift, shift), (-shift, -shift)]
        # shift = 0
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                # d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                #      (map_y * stride + shift - y) * (map_y * stride + shift - y)
                # print(map_x * stride + shift, map_y * stride + shift, d2)
                dist = lambda xx, yy: (xx - x)**2 + (yy - y)**2
                d2 = min([dist(map_x * stride + moves[i][0], map_y * stride + moves[i][1]) for i in range(len(moves))])

                exponent = d2 / 2 / scale / scale

                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma=1):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _get_paf_map(self, sample):
        """
        :param sample: dict. {"image": PIL.Image, "image_id": index of this image}
        :return: Numpy Array. Shape: (image_height // stride, image_width // stride, num_connections*2)
        """
        num_connections = _num_connections
        height, width = sample['image'].size
        paf_maps = np.zeros(shape=(num_connections, 2, height // self._stride, width // self._stride), dtype=np.float32)

        for stroke in sample['annotation']:
            for id in range(len(stroke['record'])-1):
                connection_name = '%s_%s' % (stroke['id'], id)
                connection_id = self.stroke_tool.get_connection_id(connection_name)
                kpt_a = stroke['record'][id]
                kpt_b = stroke['record'][id+1]
                self._set_paf(paf_maps[connection_id],
                              sample['image'],
                              kpt_a[0], kpt_a[1], kpt_b[0], kpt_b[1],
                              self._stride, self.paf_width)

        paf_maps = paf_maps.reshape((-1, height // self._stride, width // self._stride))
        paf_maps = np.transpose(paf_maps, (1, 2, 0))
        return paf_maps


    def _set_paf(self, paf_map, image, x_a, y_a, x_b, y_b, stride, width, scale=5):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - width, 0))
        x_max = int(min(max(x_a, x_b) + width, w_map))
        y_min = int(max(min(y_a, y_b) - width, 0))
        y_max = int(min(max(y_a, y_b) + width, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba * scale
        y_ba /= norm_ba * scale
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= width:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba

class OpenPoseDataset(Dataset):
    def __init__(self, annotations, images_info, imgdir, stroke_tool, stride=4, sigma=1, paf_width=1, mode='train', mask=False, boost=False):
        super().__init__()
        self.threshold = 0.7
        self._stride = stride
        self._sigma = sigma
        self.annotations = annotations
        self.stroke_tool = stroke_tool
        self.paf_width = paf_width
        self.images_info = {}
        self.imgdir = imgdir
        self.mode = mode
        for image_info in images_info:
            self.images_info[str(image_info['fileName'])] = image_info
        self.mask = mask
        self.mask_threshold = 150
        self.boost = boost

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        record = self.annotations[idx]
        # current_image_id = str(record['currentImageId'])
        # dataset_id = str(record['dataSetId'])
        # image_name = '%s-%s.png' % (dataset_id, current_image_id)
        image_name = record['fileName']
        char_id = record['charId']
        stroke_annotations = record['result']
        # image_info = self.images_info[image_name]

        # skip bad images
        if record['skip']:
            return None

        try:
            sample = {
                'image': Image.open('%s/%s' % (self.imgdir, image_name)).convert('RGB'),
                'image_name': image_name,
                'annotation': stroke_annotations,
            }
        except:
            return None
        mask = self._get_mask(sample)
        heatmap = self._get_heatmap(sample)
        paf_map = self._get_paf_map(sample)
        image_ten, heatmap_ten, vecmap_ten, mask_ten = self.transforms(self.mode, sample['image'], heatmap, paf_map, mask)
        data_dict = {
            'img': image_ten,
            'maskmap': mask_ten,
            'vecmap': vecmap_ten,
            'heatmap': heatmap_ten,
            'filename': image_name,
            'char_id': char_id,
        }
        return data_dict


    def transforms(self, mode, image, heatmap, vecmap, mask):
        to_tensor = transforms.ToTensor()

        # convert heatmap, vecmap, mask to tensor
        heatmap = to_tensor(heatmap)
        vecmap = to_tensor(vecmap)
        mask = to_tensor(mask)

        # Apply resize
        resize_image = transforms.Resize(size=(256, 256))
        mask_h, mask_w = 256 // self._stride, 256 // self._stride
        resize_mask = transforms.Resize(size=(mask_h, mask_w))
        image = resize_image(image)
        heatmap = resize_mask(heatmap)
        vecmap = resize_mask(vecmap)
        mask = resize_mask(mask)

        if mode in ['train']:
            # random resized crop
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=[0.8, 1.0], ratio=[0.8, 1.2])
                image = TF.resized_crop(image, i, j, h, w, size=[256, 256])
                mask = TF.resized_crop(mask, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])
                vecmap = TF.resized_crop(vecmap, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])
                heatmap = TF.resized_crop(heatmap, i//self._stride, j//self._stride, h//self._stride, w//self._stride, size=[mask_h, mask_w])

            # random rotation
            if random.random() > 0.5:
                degree = transforms.RandomRotation.get_params(degrees=[0., 20.])
                image = TF.rotate(image, degree, fill=255)
                vecmap = TF.rotate(vecmap, degree)
                heatmap = TF.rotate(heatmap, degree)
                mask = TF.rotate(mask, degree)

        # convert image to 3 channels tensor
        image = to_tensor(image)
        to_gray_scale = transforms.Grayscale(num_output_channels=3)
        image = to_gray_scale(image)
        image = image > self.threshold
        image = image.float()

        # normalize = transforms.Normalize([0.85,0.85,0.85], [0.09,0.09,0.09])
        # image = normalize(image)

        # heatmap = heatmap.half()
        # vecmap = vecmap.half()
        # mask = mask.half()
        heatmap = heatmap.float()
        vecmap = vecmap.float()
        mask = mask.float()

        return image, heatmap, vecmap, mask

    def _get_mask(self, sample):
        h, w = sample['image'].size
        h //= self._stride
        w //= self._stride
        mask = sample['image']
        to_gray = transforms.Grayscale(num_output_channels=1)
        mask = transforms.Resize((h, w))(mask)

        mask = to_gray(mask)
        mask = np.asarray(mask)

        mask = mask < self.mask_threshold
        mask = mask.astype('float')
        # not use mask, return all ones
        if self.mask is False:
            mask = np.ones((h, w))
        return mask

    def _get_heatmap(self, sample):
        '''
        :param sample:
        :return: size is (H x W x C)
        '''
        num_keypoints = _num_kpts
        image = sample['image']
        height, width = image.size
        # last dimension for background
        keypoint_maps = np.zeros(shape=(num_keypoints + 1, height // self._stride, width // self._stride), dtype=np.float32)

        for stroke in sample['annotation']:
            for id, kpt in enumerate(stroke['record']):
                kpt_name = '%s_%s' % (stroke['id'], id)
                kpt_id = self.stroke_tool.get_kpt_id(kpt_name)
                self._draw_gaussian(keypoint_maps[kpt_id], kpt[0], kpt[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        keypoint_maps = np.transpose(keypoint_maps, (1, 2, 0))

        return keypoint_maps

    def _draw_gaussian(self, keypoint_map, x, y, stride, sigma=1, scale=100):
        n_sigma = 10
        map_h, map_w = keypoint_map.shape
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        # top left coordinate
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)
        # bottom right coordinate
        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        # shift = stride / 2
        shift = stride / 2
        moves = [(0, 0), (0, shift), (0, -shift), (shift, 0), (-shift, 0), (shift, shift), (shift, -shift), (-shift, shift), (-shift, -shift)]
        # shift = 0
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                # d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                #      (map_y * stride + shift - y) * (map_y * stride + shift - y)
                # print(map_x * stride + shift, map_y * stride + shift, d2)
                dist = lambda xx, yy: (xx - x)**2 + (yy - y)**2
                d2 = min([dist(map_x * stride + moves[i][0], map_y * stride + moves[i][1]) for i in range(len(moves))])

                exponent = d2 / 2 / scale / scale

                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma=1):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _get_paf_map(self, sample):
        """
        :param sample: dict. {"image": PIL.Image, "image_id": index of this image}
        :return: Numpy Array. Shape: (image_height // stride, image_width // stride, num_connections*2)
        """
        num_connections = _num_connections
        height, width = sample['image'].size
        paf_maps = np.zeros(shape=(num_connections, 2, height // self._stride, width // self._stride), dtype=np.float32)

        for stroke in sample['annotation']:
            for id in range(len(stroke['record'])-1):
                connection_name = '%s_%s' % (stroke['id'], id)
                connection_id = self.stroke_tool.get_connection_id(connection_name)
                kpt_a = stroke['record'][id]
                kpt_b = stroke['record'][id+1]
                self._set_paf(paf_maps[connection_id],
                              sample['image'],
                              kpt_a[0], kpt_a[1], kpt_b[0], kpt_b[1],
                              self._stride, self.paf_width)

        paf_maps = paf_maps.reshape((-1, height // self._stride, width // self._stride))
        paf_maps = np.transpose(paf_maps, (1, 2, 0))
        return paf_maps


    def _set_paf(self, paf_map, image, x_a, y_a, x_b, y_b, stride, width, scale=5):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - width, 0))
        x_max = int(min(max(x_a, x_b) + width, w_map))
        y_min = int(max(min(y_a, y_b) - width, 0))
        y_max = int(min(max(y_a, y_b) + width, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba * scale
        y_ba /= norm_ba * scale
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= width:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba



def get_heatmap(img, annotations, stroke_transformer):
    heatmap = np.zeros((_num_kpts + 1, img.size[0], img.size[1]))
    for stroke in annotations:
        # print(stroke['name'])
        for id, kpt in enumerate(stroke['record']):
            kpt_name = '%s_%s' % (stroke['id'], id)
            kpt_id = stroke_transformer.get_kpt_id(kpt_name)
            heatmap[kpt_id] = np.maximum(heatmap[kpt_id], DrawGaussian(heatmap[kpt_id], kpt))
    heatmap[-1] = 1 - np.sum(heatmap[0:-1], axis=0)
    return heatmap


def get_paf(img, annotations, stroke_transformer, width=5):
    paf = np.zeros((_num_connections, 2, img.size[0], img.size[1]))
    connection_counter = Counter()
    for stroke in annotations:
        for id in range(len(stroke['record'])-1):
            connection_name = '{0}_{1}'.format(stroke['id'], id)
            connection_id = stroke_transformer.get_connection_id(connection_name)
            connection_counter[connection_id] += 1
            kpt_1 = np.asarray(stroke['record'][id])
            kpt_2 = np.asarray(stroke['record'][id+1])
            part_line_segment = kpt_2 - kpt_1
            d = np.linalg.norm(part_line_segment)
            v = part_line_segment / d
            v_per = v[1], -v[0]
            # NOTE: be careful about the order of x and y
            x, y = np.meshgrid(np.arange(img.size[1]), np.arange(img.size[0]))
            dist_along_part = v[0] * (x - kpt_1[0]) + v[1] * (y - kpt_1[1])
            dist_per_part = np.abs(v_per[0] * (x - kpt_1[0]) + v_per[1] * (y - kpt_1[1]))
            mask1 = dist_along_part >= 0
            mask2 = dist_along_part <= d
            mask3 = dist_per_part <= width
            # FIXME: Add mask4 for removing all pixels with low value.
            mask = mask1 & mask2 & mask3
            paf[connection_id, 0] = paf[connection_id, 0] + mask.astype('float32') * v[0]
            paf[connection_id, 1] = paf[connection_id, 1] + mask.astype('float32') * v[1]

    for connection_id in range(len(paf)):
        if connection_id in connection_counter:
            paf[connection_id] /= connection_counter[connection_id]

    return paf


def old_get_paf(img, annotations, all_strokes, all_connections, sigma_paf=5):
    paf_all = {}
    for stroke_name in all_strokes:
        # list of all connections of specified stroke
        # e.g. [(1, 2), (2, 3), (3, 4), (4, 5)]
        # NOTE: index is starting from 1
        connections = all_connections[stroke_name]
        out_pafs = np.zeros((len(connections), 2, img.shape[0], img.shape[1]))
        n_stroke_part = np.zeros(len(connections), img.shape[0], img.shape[1])
        # shape (N, M, 2)
        annotaion = annotations[stroke_name]
        for stroke_id in range(annotaion):
            keypoints = annotaion[stroke_id]
            for connection in range(connections):
                # NOTE: kpt_id_1 and kpt_id_2 must be integer
                kpt_id_1, kpt_id_2 = connection
                kpt_id_1 -= 1
                kpt_id_2 -= 1
                kpt_1 = keypoints[kpt_id_1]
                kpt_2 = keypoints[kpt_id_2]
                part_line_segment = kpt_2 - kpt_1
                d = np.linalg.norm(part_line_segment)
                if d > 1e-2:
                    sigma = sigma_paf
                    v = part_line_segment / d
                    v_per = v[1], -v[0]
                    # NOTE: be careful about the order of x and y
                    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                    dist_along_part = v[0] * (x - kpt_1[0]) + v[1] * (y-kpt_1[1])
                    dist_per_part = np.abs(v_per[0] * (x - kpt_1[0]) + v_per[1] * (y - kpt_1[1]))
                    mask1 = dist_along_part >= 0
                    mask2 = dist_along_part <= d
                    mask3 = dist_per_part <= sigma
                    # mask4: filter all pixels with low values.
                    mask = mask1 & mask2 & mask3
                    out_pafs[connection, 0] = out_pafs[connection, 0] + mask.astype('float32') * v[0]
                    out_pafs[connection, 1] = out_pafs[connection, 1] + mask.astype('float32') * v[1]
                    n_stroke_part[connection] += mask.astype('float32')

        n_stroke_part = n_stroke_part.reshape(out_pafs.shape[0], 1, img.shape[0], img.shape[1])
        out_pafs = out_pafs / (n_stroke_part + 1e-8)
        if stroke_name not in paf_all:
            paf_all[stroke_name] = out_pafs

    return paf_all


def main():
    pass


if __name__ == '__main__':
    main()