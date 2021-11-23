import io
import json
from collections import Counter
import random
import pandas as pd
from tqdm import tqdm

'''
Test Minimum Required Images 
'''
# max_char_count = 100
# num_copies = 50
#
# with open('data/data.json', 'r') as fb:
#     data = json.load(fb)
#
# new_data = {'content': []}
#
# counter = Counter()
#
# all_records = data['content']
# random.shuffle(all_records)
#
# for record in data['content']:
#     cid = record['charId']
#     if counter[cid] <= max_char_count:
#         counter[cid] += 1
#         new_data['content'].append(record)
#
# train_data = {'content': []}
#
# for record in new_data['content']:
#     for _ in range(num_copies):
#         train_data['content'].append(record)
#
# with open('data/part_data.json', 'w', encoding='utf8') as fb:
#     json.dump(new_data, fb, ensure_ascii=False)
#
# with open('data/train_data.json', 'w', encoding='utf8') as fb:
#     json.dump(train_data, fb, ensure_ascii=False)

'''
Merge External Dataset
'''
# with open('data/data.json', 'r', encoding='utf8') as fb:
#     train_data = json.load(fb)
#
# with open('data/data_1-100.json', 'r', encoding='utf8') as fb:
#     data = json.load(fb)
#
# for record in data['content']:
#     if record['charId'] in [70, 2050]:
#         train_data['content'].append(record)
#
# with open('data/data.json', 'w', encoding='utf8') as fb:
#     json.dump(train_data, fb, ensure_ascii=False)
# exit(0)

'''
Create Copies
'''
num_copies = 2

with open('data/all_data.json', 'r') as fb:
    data = json.load(fb)

train_data = {'content': []}

for record in tqdm(data['content']):
    for _ in range(num_copies):
        train_data['content'].append(record)

with open('data/refine_train_data.json', 'w', encoding='utf8') as fb:
    json.dump(train_data, fb, ensure_ascii=False)

exit(0)

def main():
    with open('data/strokes-detail.json', 'r', encoding='utf-8') as fb:
        strokes_info = json.load(fb)
    with open('data/char-strokes.json', 'r', encoding='utf-8') as fb:
        characters_info = json.load(fb)

    '''
    stroke_dict
    record format: (i, j, k)
    i is the start keypoint id, j is the end keypoint id, k is the connection id.  
    '''
    stroke_dict = dict()

    kpt_counter = 0
    conn_counter = 0
    for stroke in strokes_info['strokes']:
        stroke_id = stroke['id']
        num_kpts = stroke['strokeOrderLength']
        stroke_dict[stroke_id] = []
        for i in range(num_kpts-1):
            stroke_dict[stroke_id].append((i + kpt_counter, i + 1 + kpt_counter, i + conn_counter))
        kpt_counter += num_kpts
        conn_counter += num_kpts - 1

    print(stroke_dict)
    character_dict = dict()
    for char_record in characters_info['characters']:
        cid = char_record['cId']
        character_dict[cid] = dict()
        character_dict[cid]['name'] = char_record['name']
        character_dict[cid]['id'] = char_record['cId']
        character_dict[cid]['conn_seq'] = []
        for stroke_id in char_record['strokes']:
            for conn in stroke_dict[stroke_id]:
                character_dict[cid]['conn_seq'].append(conn[2])

    # with open('data/character_dict.json', 'w', encoding='utf-8') as fb:
    #     json.dump(character_dict, fb, ensure_ascii=False)



if __name__ == '__main__':
    main()