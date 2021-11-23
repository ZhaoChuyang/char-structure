import os
import json
import pandas as pd
import numpy as np


dataset_path = 'data/dataSet_100per'
output_images_info = 'data/images_info.json'


def image_rename():
    images_info = []
    root, dirs, files = next(os.walk(dataset_path))
    for dirname in dirs:
        annotation_file = '%s/%s/images.json' % (dataset_path, dirname)
        with open(annotation_file, 'r', encoding='utf8') as fb:
            image_info = json.load(fb)
            image_info = image_info['images']
        images_info.extend(image_info)

    for record in images_info:
        filename_list = record['fileName'].split('-')
        new_filename = '-'.join(filename_list[:2]) + '.png'
        record['fileName'] = new_filename

    with open('data/images_info.json', 'w', encoding='utf8') as fb:
        json.dump(images_info, fb)

    for dirname in dirs:
        for filename in os.listdir('%s/%s' % (dataset_path, dirname)):
            path = '%s/%s/%s' % (dataset_path, dirname, filename)
            extension = filename.split('.')[-1]
            if extension != 'png':
                continue
            filename_list = filename.split('-')
            new_filename = '-'.join(filename_list[:2]) + '.png'
            os.rename('%s/%s/%s' % (dataset_path, dirname, filename),
                      '%s/%s/%s' % (dataset_path, dirname, new_filename))


def create_dataset_df():
    with open('data/images_info.json', 'r', encoding='utf8') as fb:
        images_info = json.load(fb)


def main():
    pass

