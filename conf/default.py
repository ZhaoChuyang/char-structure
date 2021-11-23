snapshot = 'checkpoints/openpose_model_apex_ep0.pt'
stride = 8
bin_threshold = 0.7
# device = 'cuda:2'
device = 'cpu'

_num_kpts = 87
_num_connections = 61

character_dict = 'data/character_dict.json'
stroke_dict = 'data/stroke_dict.json'
std_skeleton = 'data/std_skeleton.pkl'

target_font = 'data/fonts/TYZ.ttf'
image_dir = 'data/peanut'
target_dir = 'data/targets'

network = dict(
    heatmap_out=_num_kpts+1,
    paf_out=_num_connections*2,
)

test = dict(
    input_size=(256, 256),
    search_scale=[1]
)


res = dict(
    heatmap_threshold=0.05,
    oks_threshold=0.4,
    num_mid_points=200,
    conn_threshold=0.1,
    conn_pos_ratio=0.3,
)

# res = dict(
#     heatmap_threshold=0.2,
#     oks_threshold=0.4,
#     num_mid_points=40,
#     conn_threshold=0.1,
#     conn_pos_ratio=0.7,
# )

details = dict(
    conn_seq=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [11, 12], [13, 14], [15, 16], [16, 17], [17, 18],
                [19, 20], [20, 21], [22, 23], [23, 24], [25, 26], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33],
                [34, 35], [35, 36], [37, 38], [38, 39], [40, 41], [41, 42], [43, 44], [44, 45], [45, 46], [46, 47],
                [48, 49], [49, 50], [51, 52], [52, 53], [53, 54], [54, 55], [56, 57], [57, 58], [58, 59], [60, 61],
                [61, 62], [63, 64], [64, 65], [65, 66], [66, 67], [68, 69], [69, 70], [70, 71], [71, 72], [73, 74],
                [74, 75], [75, 76], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [83, 84], [84, 85], [85, 86]],
)
