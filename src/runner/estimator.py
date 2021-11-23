import os
import sys
import math
import json
from copy import copy
import argparse
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.io import savemat, loadmat
from scipy import signal
import torch
from torchvision import transforms
import math
from src.utils.config import Config
from src.modeling.openpose import SimpleNet, OpenPose
import pickle


import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-name', type=str)
    parser.add_argument('--char-id', type=int)
    return parser.parse_args()


class StructureEstimator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.model = OpenPose('test', 'resnet50')
        if cfg.snapshot:
            if self.cfg.device == 'cpu':
                self.model.load_state_dict(torch.load(cfg.snapshot, map_location=torch.device('cpu'))['model'])
            else:
                self.model.load_state_dict(torch.load(cfg.snapshot)['model'])
        self.model.eval()
        self.bin_threshold = cfg.bin_threshold
        with open(cfg.character_dict, 'r', encoding='utf8') as fb:
            self.character_dict = json.load(fb)
        with open(cfg.stroke_dict, 'r', encoding='utf8') as fb:
            self.stroke_dict = json.load(fb)


    def _get_blob(self, ori_image, scale):
        assert scale is not None
        resize = transforms.Resize(size=scale)
        to_gray = transforms.Grayscale(num_output_channels=3)
        to_tensor = transforms.ToTensor()
        # normalize = transforms.Normalize([0.85, 0.85, 0.85], [0.09, 0.09, 0.09])
        image = resize(ori_image)
        image = to_tensor(image)
        image = to_gray(image)
        image = image > self.bin_threshold
        image = image.float()
        # image = normalize(image)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
        return image[np.newaxis, ...]

    def adaptive_thres(self, img, win=9, beta=0.95):
        if win % 2 == 0: win = win - 1
        # 边界的均值有点麻烦
        # 这里分别计算和和邻居数再相除
        kern = np.ones([win, win])
        sums = signal.correlate2d(img, kern, 'same')
        cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
        means = sums // cnts
        # 如果直接采用均值作为阈值，背景会变花
        # 但是相邻背景颜色相差不大
        # 所以乘个系数把它们过滤掉
        img = np.where(img < means * beta, 0, 255).astype('float')
        return img

    def test_img(self, image_path, char_id):

        ori_image = Image.open(image_path).convert('RGB')
        # ori_image_data = np.asarray(ori_image)
        #
        # print(ori_image_data.shape)
        # ori_image_data = self.adaptive_thres(ori_image_data)
        # ori_image = Image.fromarray(ori_image_data).convert('RGB')
        # ori_image.show()
        # resize to 256 for saving memory
        ori_image = ori_image.resize(self.cfg.test.input_size)
        ori_width, ori_height = ori_image.size
        heatmap_avg = np.zeros((ori_height, ori_width, self.cfg.network.heatmap_out))
        paf_avg = np.zeros((ori_height, ori_width, self.cfg.network.paf_out))

        # (ori_width, ori_height) => input_size => scale * input_size
        scales = [(int(self.cfg.test.input_size[0] * scale), int(self.cfg.test.input_size[1] * scale))
                  for scale in self.cfg.test.search_scale]

        stride = self.cfg.stride
        for idx, scale in enumerate(scales):
            input = dict()
            input['img'] = self._get_blob(ori_image, scale)
            with torch.no_grad():
                out_dict = self.model(input)
                paf_out = out_dict['paf'][-1]
                heatmap_out = out_dict['heatmap'][-1]

                # (c,h,w) => (h,w,c)
                heatmap = heatmap_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)

                paf = paf_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                paf = cv2.resize(paf, (ori_width, ori_height), interpolation=cv2.INTER_CUBIC)

                heatmap_avg = heatmap_avg + heatmap / len(scales)
                paf_avg = paf_avg + paf / len(scales)

        mat_dict = {'heatmap': heatmap_avg, 'paf': paf_avg, 'image_data': np.squeeze(input['img'].detach().cpu().numpy())}
        savemat('output/%s.mat' % image_path.split('/')[-1], mat_dict)
        image_data = np.asarray(ori_image)
        image_ten = input['img'].detach().cpu().numpy()
        all_peaks = self._extract_heatmap_info(heatmap_avg, char_id)

        # fig, ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # ax.imshow(ori_image)
        # for pid, kpts in enumerate(all_peaks):
        #     if kpts:
        #         # print(kpts)
        #         if pid not in [0, 1]: continue
        #         for x, y, confidence, _ in kpts:
        #             circ = Circle((x, y), confidence * 3)
        #             ax.add_patch(circ)
        # plt.show()

        # mat_dict = {'image_data': image_data, 'paf': paf_avg, 'heatmap': heatmap}
        # mat_dict = {'peaks': dict()}
        # for id, peaks in enumerate(all_peaks):
        #     mat_dict['peaks']['kpt_' + str(id)] = peaks
        # print(mat_dict)
        # savemat('output/test.mat', mat_dict)

        special_k, connection_all = self._extract_paf_info(ori_image, paf_avg, all_peaks, char_id)
        char_structure = self._get_subsets(connection_all, special_k, all_peaks, char_id)

        mat_dict = {'image_ten': image_ten, 'image_name': image_path, 'image_size': [ori_width, ori_height], 'image_data': np.asarray(ori_image), 'structure': char_structure, 'heatmap': heatmap_avg, 'paf': paf_avg}
        savemat('output/char_structure.mat', mat_dict)
        return mat_dict

    def _get_subsets(self, connection_all, special_k, all_peaks, char_id):
        """
        :param connection_all:
        :param special_k:
        :param all_peaks:
        :param char_id:
        :return:
        char_structure: list
        list contains all connection information.
        items in it has the following format:
        [conn_id, kpt_1_x, kpt_1_y, kpt_2_x, kpt_2_y, conn_score, kpt_1_score, kpt_2_score]
        - conn_id: id of the current connection
        - kpt_1_x: x coordinate of the starting point
        - kpt_1_y: y coordinate of the starting point
        - kpt_2_x: x coordinate of the ending point
        - kpt_2_y: y coordinate of the ending point
        - conn_score: confidence score of the connection
        - kpt_1_score: confidence score of the starting point
        - kpt_2_score: confidence score of the ending point

        """
        char_id = str(char_id)
        char_conn_info = self.character_dict[char_id]

        results = []
        temp_connection_all = [[] if isinstance(r, list) else r.tolist() for r in connection_all]
        print(temp_connection_all[0])
        for id, connections in enumerate(temp_connection_all):
            if connections:
                temp_connection_all[id] = sorted(connections, key=lambda x: x[2])
        print('sorted')
        print(temp_connection_all[0])
        '''
        result: list
        result contains all connection information for the current character.
        record in result has the following format:
        [id_1, id_2, score, i, j, conn_id]
        - id_1 is the unique id for the starting keypoint
        - id_2 is the unique id for the ending keypoint
        - score is the confidence score of this connection
        - i and j are temporal variable, not used here
        - conn_id is the type id of this connection
        '''
        for conn_id in char_conn_info['conn_seq']:
            if temp_connection_all[conn_id]:
                # print(temp_connection_all[conn_id])
                results.append(temp_connection_all[conn_id].pop() + [conn_id])

        '''
        kpt_dict: dictionary
        kpt_dict contains all information about the extracted keypoints.
        record in kpt_dict has the following format:
        {kpt_id: (x, y, score, kpt_id)}
        - kpt_id is the unique id of the extracted keypoint
        - x and y are the coordinates of this keypoint
        - score is the confidence score of this keypoint
        '''
        kpt_dict = {}
        for peaks in all_peaks:
            for kpt in peaks:
                kpt_dict[kpt[-1]] = kpt

        char_structure = []
        for conn_info in results:
            kpt_1_id, kpt_2_id, score, _i, _j, conn_id = conn_info

            kpt_1 = kpt_dict[int(kpt_1_id)]
            kpt_2 = kpt_dict[int(kpt_2_id)]

            record = [conn_id, kpt_1[0], kpt_1[1], kpt_2[0], kpt_2[1], score, kpt_1[2], kpt_2[2]]
            char_structure.append(record)

        return char_structure

    def _extract_paf_info(self, ori_image, paf, all_peaks, char_id=None):
        """
        :param ori_image:
        :param paf:
        :param all_peaks:
        :return:
        connection_all: Array of size N, N indicating the number of connections (61 in our application).
        For item k in connection_all, it contains all candidate connections for the k-th connection.
        Each candidate has the following format:
        (id_1, id_2, score, i, j)
        - id_1 is the id of the start keypoint
        - id_2 is the id of the end keypoint
        - score is the confidence score of this candidate connection
        - i and j is the temp variables to filter residual connections
        """
        image_data = np.array(ori_image.convert('L'))
        image_data = (image_data < 150).astype('float')

        connection_all = []
        special_k = []
        for k in range(len(self.cfg.details.conn_seq)):
            score_mid = paf[:, :, [k*2, k*2+1]]
            num_mid_points = self.cfg.res.num_mid_points
            cand_A = all_peaks[self.cfg.details.conn_seq[k][0]]
            cand_B = all_peaks[self.cfg.details.conn_seq[k][1]]
            num_cand_A = len(cand_A)
            num_cand_B = len(cand_B)
            if num_cand_A != 0 and num_cand_B != 0:
                connection_candidates = []
                for i in range(num_cand_A):
                    for j in range(num_cand_B):
                        vec = np.subtract(cand_B[j][:2], cand_A[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-9
                        vec = np.divide(vec, norm)
                        startend = zip(np.linspace(cand_A[i][0], cand_B[j][0], num=num_mid_points),
                                       np.linspace(cand_A[i][1], cand_B[j][1], num=num_mid_points))
                        startend = list(startend)
                        # vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] * image_data[int(round(startend[I][1])), int(round(startend[I][0]))]
                        #                   for I in range(len(startend))])
                        # vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] * image_data[int(round(startend[I][1])), int(round(startend[I][0]))]
                        #                   for I in range(len(startend))])
                        #
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] * image_data[int(round(startend[I][1])), int(round(startend[I][0]))]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] * image_data[int(round(startend[I][1])), int(round(startend[I][0]))]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                        # fig, ax = plt.subplots(1)
                        # ax.set_aspect('equal')
                        # # ax.imshow(ori_image)
                        # if k == 0:
                        #     for x, y in startend:
                        #         print(image_data[int(y), int(x)])
                        #         image_data[int(y), int(x)] = 1
                        #
                        #     ax.imshow(image_data)
                        #     print(sum(score_midpts) / len(score_midpts))
                        #     print(sum(score_midpts))
                        # # for pid, kpts in enumerate(all_peaks):
                        # #     if kpts:
                        # #         print(kpts)
                        # #         if pid not in [4, 5]: continue
                        # #         for x, y, confidence, _ in kpts:
                        # #             circ = Circle((x, y), confidence * 3)
                        # #             ax.add_patch(circ)
                        #     plt.show()
                        if k == 0:
                            np.savetxt('output/%d_%d.txt' % (i, j), score_midpts, fmt='%.3f')


                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)

                        score_with_dist_prior += min(0.5 * ori_image.size[0] / norm - 1, 0)

                        num_positive = np.sum(score_midpts > self.cfg.res.conn_threshold)
                        criterion1 = True
                        criterion2 = True
                        # criterion1 = num_positive > int(self.cfg.res.conn_pos_ratio * len(score_midpts))
                        # criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidates.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + cand_A[i][2] + cand_B[j][2]])
                        else:
                            pass

                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)
                # print(connection_candidates)
                if k == 0:
                    print('*******************')
                    print(connection_candidates)
                connection = np.zeros((0, 5))

                for c in range(len(connection_candidates)):
                    i, j, s = connection_candidates[c][0:3]
                    # one keypoint can only connect to only one keypoint
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        if k == 0:
                            print([cand_A[i][3], cand_B[j][3], s, i, j])
                        connection = np.vstack([connection, [cand_A[i][3], cand_B[j][3], s, i, j]])
                        if len(connection) >= min(num_cand_A, num_cand_B):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return special_k, connection_all

    def _extract_paf_info_v2(self, ori_image, paf, all_peaks, char_id):
        """
        :param ori_image:
        :param paf:
        :param all_peaks:
        :return:
        connection_all: Array of size N, N indicating the number of connections (61 in our application).
        For item k in connection_all, it contains all candidate connections for the k-th connection.
        Each candidate has the following format:
        (id_1, id_2, score, i, j)
        - id_1 is the id of the start keypoint
        - id_2 is the id of the end keypoint
        - score is the confidence score of this candidate connection
        - i and j is the temp variables to filter residual connections
        """
        # list contains all connection_id
        char_id = str(char_id)
        conn_seq = self.character_dict[char_id]['conn_seq']
        gt_conn_seq = [0] * self.cfg._num_connections

        for conn_id in conn_seq:
            gt_conn_seq[conn_id] += 1

        connection_all = []
        special_k = []

        image_data = np.array(ori_image.convert('L'))
        image_data = (image_data < 150).astype('float')

        for k in range(len(self.cfg.details.conn_seq)):
            gt_conn_num = gt_conn_seq[k]

            score_mid = paf[:, :, [k * 2, k * 2 + 1]]
            num_mid_points = self.cfg.res.num_mid_points
            cand_A = all_peaks[self.cfg.details.conn_seq[k][0]]
            cand_B = all_peaks[self.cfg.details.conn_seq[k][1]]
            num_cand_A = len(cand_A)
            num_cand_B = len(cand_B)
            if gt_conn_num != 0 and (num_cand_A == 0 or num_cand_B == 0):
                print("缺少笔画:", k, gt_conn_num)
            if num_cand_A != 0 and num_cand_B != 0 and gt_conn_num !=0:
                connection_candidates = []
                for i in range(num_cand_A):
                    for j in range(num_cand_B):
                        vec = np.subtract(cand_B[j][:2], cand_A[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-9
                        vec = np.divide(vec, norm)
                        startend = zip(np.linspace(cand_A[i][0], cand_B[j][0], num=num_mid_points),
                                       np.linspace(cand_A[i][1], cand_B[j][1], num=num_mid_points))
                        startend = list(startend)

                        score_midpts = np.array([image_data[int(round(startend[I][1])), int(round(startend[I][0]))]
                                          for I in range(len(startend))])

                        if k == 2:
                            np.savetxt('output/%d_%d.txt' % (i, j), score_midpts)
                        #
                        # fig, ax = plt.subplots(1)
                        # ax.set_aspect('equal')
                        # # ax.imshow(ori_image)
                        # if k == 2:
                        #     for x, y in startend:
                        #         print(image_data[int(y), int(x)])
                        #         image_data[int(y), int(x)] = 1
                        #
                        #     ax.imshow(image_data)
                        #     print(sum(score_midpts) / len(score_midpts))
                        #     print(sum(score_midpts))
                        # # for pid, kpts in enumerate(all_peaks):
                        # #     if kpts:
                        # #         print(kpts)
                        # #         if pid not in [4, 5]: continue
                        # #         for x, y, confidence, _ in kpts:
                        # #             circ = Circle((x, y), confidence * 3)
                        # #             ax.add_patch(circ)
                        #     plt.show()

                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        score_with_dist_prior += min(0.5 * ori_image.size[0] / norm - 1, 0)

                        num_positive = np.sum(score_midpts > self.cfg.res.conn_threshold)
                        criterion1 = True
                        criterion2 = True
                        # criterion1 = num_positive > int(self.cfg.res.conn_pos_ratio * len(score_midpts))
                        # criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidates.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + cand_A[i][2] + cand_B[j][2]])
                        else:
                            pass

                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)
                print(connection_candidates)
                print('应有笔画数:', gt_conn_num)
                print('候选笔画数:', len(connection_candidates))

                connection = np.zeros((0, 5))
                for c in range(len(connection_candidates)):
                    i, j, s = connection_candidates[c][0:3]
                    # one keypoint can only connect to only one keypoint
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [cand_A[i][3], cand_B[j][3], s, i, j]])
                        if len(connection) >= min(num_cand_A, num_cand_B):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return special_k, connection_all

    def _extract_heatmap_info(self, heatmap, char_id):
        """
        extract all candidate keypoints
        :param heatmap: input heatmap
        :return: all_peaks
        Array of length N, N indicate the number of keypoints (87 in our application).
        For item k in all_peaks, it contains all candidates for the k-th keypoint.
        The k-th item in `all_peaks` is also an array, each item in it has the following format:
        (x, y, score, id):
        - `x` and `y` is the coordinate of this candidate.
        - `score` is the confidence score of this candidate.
        - `id` is the unique id of this candidate.
        """
        all_peaks = []
        peak_counter = 0
        heatmap_dict = [0] * self.cfg._num_kpts
        char_id = str(char_id)
        conns_in_char = self.character_dict[char_id]['conn_seq']
        conn_dict = {}

        for stroke_id, conn_list in self.stroke_dict.items():
            for conn in conn_list:
                conn_dict[conn[2]] = conn[:2]

        for conn_id in conns_in_char:
            kpt_1, kpt_2 = conn_dict[conn_id]
            heatmap_dict[kpt_1] += 1
            heatmap_dict[kpt_2] += 1

        for part in range(self.cfg._num_kpts):
            max_kpt_num = heatmap_dict[part]
            map_ori = heatmap[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            map_heap = np.max(map_ori)

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down, map_gau > self.cfg.res.heatmap_threshold))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            # FIXME: Apply OKS Algorithm Here

            ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

            '''
            peaks_with_score_and_id: list of tuples
            item in peaks_with_score_and_id has the following format:
            (x, y, score, id)
            '''
            # peaks_with_score_and_id = sorted(peaks_with_score_and_id, key=lambda x: x[2], reverse=True)[:max_kpt_num]
            peaks_with_score_and_id = sorted(peaks_with_score_and_id, key=lambda x: x[2], reverse=True)

            all_peaks.append(peaks_with_score_and_id[:max_kpt_num+1])
            peak_counter += len(peaks)
        return all_peaks

    def _extract_heatmap_info_v2(self, heatmap, char_id):
        """
        extract all candidate keypoints
        :param heatmap: input heatmap
        :return: all_peaks
        Array of length N, N indicate the number of keypoints (87 in our application).
        For item k in all_peaks, it contains all candidates for the k-th keypoint.
        The k-th item in `all_peaks` is also an array, each item in it has the following format:
        (x, y, score, id):
        - `x` and `y` is the coordinate of this candidate.
        - `score` is the confidence score of this candidate.
        - `id` is the unique id of this candidate.
        """
        all_peaks = []
        peak_counter = 0
        heatmap_dict = [0] * self.cfg._num_kpts
        char_id = str(char_id)
        conns_in_char = self.character_dict[char_id]['conn_seq']
        conn_dict = {}

        for stroke_id, conn_list in self.stroke_dict.items():
            for conn in conn_list:
                conn_dict[conn[2]] = conn[:2]

        for conn_id in conns_in_char:
            kpt_1, kpt_2 = conn_dict[conn_id]
            heatmap_dict[kpt_1] += 1
            heatmap_dict[kpt_2] += 1

        for part in range(self.cfg._num_kpts):
            max_kpt_num = heatmap_dict[part]
            map_ori = heatmap[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            # FIXME: Apply OKS Algorithm Here

            ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

            '''
            peaks_with_score_and_id: list of tuples
            item in peaks_with_score_and_id has the following format:
            (x, y, score, id)
            '''
            # peaks_with_score_and_id = sorted(peaks_with_score_and_id, key=lambda x: x[2], reverse=True)[:max_kpt_num]
            peaks_with_score_and_id = sorted(peaks_with_score_and_id, key=lambda x: x[2], reverse=True)

            all_peaks.append(peaks_with_score_and_id[:max_kpt_num+3])
            peak_counter += len(peaks)
        return all_peaks


if __name__ == '__main__':
    cfg = Config.fromfile('conf/default.py')
    args = get_args()
    image_path = 'data/peanut/1000-2.png'
    estimator = StructureEstimator(cfg)
    estimator.test_img(image_path, 70)

