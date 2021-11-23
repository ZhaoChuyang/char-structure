import os
import shutil
import json


if __name__ == '__main__':
    with open('data/character_dict.json', 'r', encoding='utf8') as fb:
        char_dict = json.load(fb)

    char_map = {}
    for k, v in char_dict.items():
        char_name = v['name']
        char_id = v['id']
        char_map[char_name] = char_id

    for filename in os.listdir('data/fonts/kaiti'):
        if filename.split('.')[-1] == 'png':
            char_name = filename.split('.')[0]
            char_id = char_map[char_name]
            src_path = 'data/fonts/kaiti/' + filename
            dst_path = 'data/targets/%d.png' % char_id
            os.rename(src_path, dst_path)