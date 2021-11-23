import numpy as np
from PIL import Image
import PIL
import torch
import random

sigma_inp = 7
n = sigma_inp * 6 + 1
g_inp = np.zeros((n, n))


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def Gaussian(sigma):
    if sigma == 7:
        return np.array([0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.2301, 0.5205, 0.8494, 1.0000, 0.8494, 0.5205, 0.2301,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529]).reshape(7, 7)
    elif sigma == n:
        return g_inp
    else:
        raise Exception('Gaussian {} Not Implement'.format(sigma))


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def DrawGaussian(img, pt, sigma=1):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def resize_heatmap(heatmap, stride):
    """
    heatmap shape: (height, width)
    """
    resized_heatmap = Image.fromarray(heatmap)
    h, w = resized_heatmap.size
    print(h, w)
    resized_heatmap = resized_heatmap.resize((h * stride, w * stride), PIL.Image.BILINEAR)
    return np.asarray(resized_heatmap)

def resize_paf_map(paf_map, stride):
    h, w = paf_map[0].shape
    resized_paf_x = Image.fromarray(paf_map[0])
    resized_paf_y = Image.fromarray(paf_map[1])
    resized_paf_x = resized_paf_x.resize((h * stride, w * stride), PIL.Image.BILINEAR)
    resized_paf_y = resized_paf_y.resize((h * stride, w * stride), PIL.Image.BILINEAR)
    resized_paf_map = np.asarray([np.asarray(resized_paf_x), np.asarray(resized_paf_y)])
    return resized_paf_map

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, optim, name, detail, amp=None):
    path = "checkpoints/%s_ep%d.pt" % (name, detail['epoch'])
    if amp:
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "detail": detail,
            "amp": amp.state_dict(),
        }, path)
    else:
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "detail": detail,
        }, path)
    print("save model to %s" % path)


def load_model(path, model, use_gpu=False, optim=None, amp=None):
    # remap everthing onto CPU
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    model.load_state_dict(state["model"])
    if optim:
        print("loading optim")
        optim.load_state_dict(state["optim"])
    else:
        print("not loading optim")
    if amp:
        print("loading amp")
        amp.load_state_dict(state["amp"])
    else:
        print("not loading amp")
    if use_gpu:
        model.cuda()
    detail = state["detail"]
    print("loaded model from %s" % path)
    return detail

