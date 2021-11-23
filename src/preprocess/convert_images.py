import os
from PIL import Image
import tqdm
import numpy as np


img_dir = 'data/peanut'


if __name__ == '__main__':
    for file in os.listdir(img_dir):
        if file.split('.')[-1] != 'png':
            continue
        img_path = '%s/%s' % (img_dir, file)
        img = Image.open(img_path)
        if img.mode in ['1']:
            img = img.convert('RGB')
            img.save(img_path)