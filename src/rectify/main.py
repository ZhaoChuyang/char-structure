from PIL import ImageFont, ImageDraw, Image, ImageOps
from src.utils.config import Config
from src.runner.estimator import StructureEstimator
# from src.runner.qer_estimator import StructureEstimator
import json
import numpy as np
from scipy.io import savemat
from math import fabs, sqrt
from collections import Counter


cfg = Config.fromfile('conf/default.py')
estimator = StructureEstimator(cfg)
with open(cfg.character_dict, 'r', encoding='utf-8') as fb:
    character_dict = json.load(fb)


def get_target_char(char_id):
    """
    Get the image of input character in target font.
    :param cfg: config file
    :param character: input character, utf-8 encoding
    :return: PIL image
    """
    character = character_dict[str(char_id)]
    font = ImageFont.truetype('data/fonts/TYZ.ttf', cfg.test.input_size[0])
    canvas = Image.new('RGB', cfg.test.input_size, 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), character, fill='black', font=font)
    canvas.save('')
    return canvas


def get_corners(char_structure, width, height, margin=15):
    """
    Get top left and bottom right corners for the given character
    :param char_structure: list, output of estimator
    :return: coordinates of top_left and bottom right corners
    """
    top_left = [float('inf'), float('inf')]
    bottom_right = [0, 0]
    for conn in char_structure:
        conn_id, x1, y1, x2, y2, s1, s2, s3 = conn
        top_left[0] = min(top_left[0], x1)
        top_left[1] = min(top_left[1], y1)
        bottom_right[0] = max(bottom_right[0], x2)
        bottom_right[1] = max(bottom_right[1], y2)
    top_left[0] = max(top_left[0] - margin, 0)
    top_left[1] = max(top_left[1] - margin, 0)
    bottom_right[0] = min(bottom_right[0] + margin, width)
    bottom_right[1] = min(bottom_right[1] + margin, height)
    return top_left, bottom_right


def align(input_structure, input_image, target_structure, target_image):

    input_image = input_image.convert('L')
    input_image_data = np.asarray(input_image)
    input_image_data = (input_image_data > 150).astype('float')
    input_image = Image.fromarray(input_image_data)

    input_counter = Counter()
    target_counter = Counter()
    for conn in input_structure:
        input_counter[conn[0]] += 1
    for conn in target_structure:
        target_counter[conn[0]] += 1

    if (input_counter != target_counter):
        print('[Align]:keypoints not match!')
        return input_structure, input_image, target_structure, target_image

    input_l, input_r, input_t, input_b = 255, 0, 255, 0
    target_l, target_r, target_t, target_b = 255, 0, 255, 0
    for conn in input_structure:
        _, x1, y1, x2, y2, _, _, _ = conn
        if x1 < input_l:
            input_l = x1
        if x2 < input_l:
            input_l = x2
        if x1 > input_r:
            input_r = x1
        if x2 > input_r:
            input_r = x2
        if y1 < input_t:
            input_t = y1
        if y2 < input_t:
            input_t = y2
        if y1 > input_b:
            input_b = y1
        if y1 > input_b:
            input_b = y1

    for conn in target_structure:
        _, x1, y1, x2, y2, _, _, _ = conn
        if x1 < target_l:
            target_l = x1
        if x2 < target_l:
            target_l = x2
        if x1 > target_r:
            target_r = x1
        if x2 > target_r:
            target_r = x2
        if y1 < target_t:
            target_t = y1
        if y2 < target_t:
            target_t = y2
        if y1 > target_b:
            target_b = y1
        if y1 > target_b:
            target_b = y1

    input_width = input_r - input_l
    input_height = input_b - input_t

    target_width = target_r - target_l
    target_height = target_b - target_t

    # print(target_width, target_height, input_width, input_height)
    resize_ratio = min(target_width / input_width, target_height / input_height)
    resized_width = int(resize_ratio * 256)
    resized_height = int(resize_ratio * 256)

    print('resize ratio:', resize_ratio)

    input_image = input_image.resize((resized_width, resized_height))

    for conn in input_structure:
        conn[1] *= resize_ratio
        conn[2] *= resize_ratio
        conn[3] *= resize_ratio
        conn[4] *= resize_ratio

    input_l, input_r, input_t, input_b = 255, 0, 255, 0
    # target_l, target_r, target_t, target_b = 255, 0, 255, 0
    for conn in input_structure:
        _, x1, y1, x2, y2, _, _, _ = conn
        if x1 < input_l:
            input_l = x1
        if x2 < input_l:
            input_l = x2
        if x1 > input_r:
            input_r = x1
        if x2 > input_r:
            input_r = x2
        if y1 < input_t:
            input_t = y1
        if y2 < input_t:
            input_t = y2
        if y1 > input_b:
            input_b = y1
        if y1 > input_b:
            input_b = y1

    print(target_l, input_l)
    print(target_t, input_t)
    x_shift = - target_l + input_l
    y_shift = - target_t + input_t

    input_image_data = np.asarray(input_image)
    input_image_data = 1 - input_image_data
    input_image = Image.fromarray(input_image_data)
    print(input_image.size)
    input_image = input_image.crop((x_shift, y_shift, x_shift+256, y_shift+256))

    input_image_data = np.asarray(input_image)

    # np.savetxt('output/img_0.txt', input_image_data[:,:,0], fmt='%.3f')
    # np.savetxt('output/img_1.txt', input_image_data[:,:,1], fmt='%.3f')
    # np.savetxt('output/img_2.txt', input_image_data[:,:,2], fmt='%.3f')

    print(input_image_data.shape)
    input_image_data = 255 - input_image_data
    input_image = Image.fromarray(input_image_data)

    print(x_shift, y_shift)

    # input_image.show()

    for conn in input_structure:
        conn[1] -= x_shift
        conn[2] -= y_shift
        conn[3] -= x_shift
        conn[4] -= y_shift
        conn[1] = int(conn[1])
        conn[2] = int(conn[2])
        conn[3] = int(conn[3])
        conn[4] = int(conn[4])

    return input_structure, input_image, target_structure, target_image


def rectify(image_name, char_id):
    target_image_path = '%s/%s.png' % (cfg.target_dir, char_id)
    try:
        Image.open(target_image_path)
    except:
        raise FileNotFoundError('Target font image not exist (id=%d)' % char_id)
    print(type(char_id))
    # input_image_path = '%s/%s' % (cfg.image_dir, image_name)
    input_image_path = image_name
    input_result = estimator.test_img(input_image_path, char_id)
    target_result = estimator.test_img(target_image_path, char_id)
    input_image = Image.open(input_image_path).convert('RGB').resize((256, 256))
    # input_image = np.asarray(input_image)
    # input_image = input_image > cfg.bin_threshold
    # input_image = input_image.astype('float')
    # input_image = Image.fromarray(input_image).convert('RGB')

    input_structure = input_result['structure']
    target_structure = target_result['structure']

    target_image = Image.open(target_image_path).resize((256, 256))

    input_structure, input_image, target_structure, target_image = align(
        input_structure,
        input_image,
        target_structure,
        target_image
    )

    # make transforms
    # input_image, input_structure = make_transforms(input_image, input_result['structure'], input_tl, input_br, 256, 256)
    # target_image, target_structure = make_transforms(target_image, target_result['structure'], target_tl, target_br, 256, 256)

    input_mat_dict = {'image_data': np.asarray(input_image), 'structure': input_structure}
    target_mat_dict = {'image_data': np.asarray(target_image), 'structure': target_structure}

    savemat('output/input.mat', input_mat_dict)
    savemat('output/target.mat', target_mat_dict)

    print(input_structure)


    # result_mat_dict = {
    #     'input_structure': input_structure,
    #     'target_structure': target_structure,
    # }
    #
    # input_image.save('input.png')
    # target_image.save('target.png')
    #
    # with open('result.json', 'w') as fb:
    #     json.dump(result_mat_dict, fb)


def make_transforms(image, char_structure, tl, br, new_width, new_height):
    ori_width, ori_height = br[0] - tl[0], br[1] - tl[1]
    multiplier = (new_width / ori_width, new_height / ori_height)
    for i, conn in enumerate(char_structure):
        conn_id, x1, y1, x2, y2, s1, s2, s3 = conn
        char_structure[i] = [conn_id, round((x1-tl[0]) * multiplier[0]), round((y1-tl[1]) * multiplier[1]), round((x2-tl[0]) * multiplier[0]), round((y2-tl[1]) * multiplier[1]), s1, s2, s3]
    print(tl[0], tl[1], br[0], br[1])
    image = image.crop((tl[0], tl[1], br[0], br[1]))
    image = image.resize((new_width, new_height))
    return image, char_structure


def crop(input_path, save_path, left, upper, right, lower, threshold=0.7):
    input_image = Image.open(input_path).convert('RGB')
    output_image = input_image.crop((left-10, upper-10, right+10, lower+10))
    output_image.save(save_path)


if __name__ == '__main__':
    rectify('data/test/ye_1.png', 17)
    # crop('data/peanut/1000-1.png', 'temp.png', 0, 0, 256, 256)
    # image = Image.open('data/targets/70.png').convert('RGB')
    # image = image.crop((5, 5, 100, 100))
    # image.show()


''' 
    font = ImageFont.truetype('data/fonts/TYZ.ttf', 256)
    canvas = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(canvas)
    character = u'云'
    draw.text((0, 0), character, fill='black', font=font)
    canvas.save('output/TYZ-yun.png')
    canvas.show()
'''
