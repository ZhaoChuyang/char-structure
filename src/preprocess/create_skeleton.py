import json
from src.preprocess.peanut import OpenPoseDataset, Stroke
import pickle

if __name__ == '__main__':
    with open('data/images_info.json', 'r', encoding='utf8') as fb:
        images_info = json.load(fb)
    with open('data/strokes-detail.json', 'r', encoding='utf8') as fb:
        strokes_info = json.load(fb)
    with open('data/char-strokes.json', 'r', encoding='utf8') as fb:
        char2strokes = json.load(fb)
    with open('data/all_data.json', 'r', encoding='utf8') as fb:
        annotations = json.load(fb)

    strokes_info = strokes_info['strokes']
    stroke_tool = Stroke(strokes_info)
    stride = 8

    normal_data = {'content': []}
    for record in annotations['content']:
        if record['fileName'].split('_')[0] == 'ttf29' and record['fileName'].split('_')[-1] == '10.png':
            normal_data['content'].append(record)

    # with open('data/std_data.json', 'w', encoding='utf8') as fb:
    #     json.dump(normal_data, fb, ensure_ascii=False)

    dataset = OpenPoseDataset(
        annotations=normal_data['content'],
        images_info=images_info,
        imgdir='data/peanut',
        stroke_tool=stroke_tool,
        stride=stride,
        mode='test',
        mask=True
    )

    save_path = 'data/std_skeleton.pkl'
    with open(save_path, 'wb') as fb:
        pickle.dump(dataset, fb)

    print('standard skeleton created at \'%s\'' % save_path)

