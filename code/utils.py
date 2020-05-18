from typing import List
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from cv2.ximgproc import guidedFilter
from PIL import Image

# fushion stragety


def low_pass(img, k=(30, 30)):
    'img: np.ndarray, float32, 0-1'
    assert img.dtype == np.float32 and img.max() <= 1., \
        'img must be in np.float32 for utils.low_pass'
    base = cv2.blur(img, k)
    detail = img - base
    return base, detail


def decompose(float_imgs):
    bases, details = [], []
    for fimg in float_imgs:
        base, detail = low_pass(fimg)
        bases.append(base)
        details.append(detail)
    return bases, details


def saliency(img, D):
    assert img.dtype == np.uint8, 'img must be in np.uint8 for utils.saliency'
    hist = np.bincount(img.flatten(), minlength=256) / img.size
    sal_values = np.dot(hist, D)
    saliency = sal_values[img]
    return saliency


def sal_weights(imgs):
    D = cdist(np.arange(256).reshape(-1, 1), np.arange(256).reshape(-1, 1))
    Ws = [saliency(img, D) for img in imgs]
    Ws = np.dstack(Ws) + 1e-12
    Ws = Ws / Ws.sum(axis=2, keepdims=True)
    return Ws


def guided_optimize(guides, srcs, r, eps):
    Ws = [guidedFilter(guide.astype(np.float32), src.astype(np.float32), r, eps)
          for guide, src in zip(guides, srcs)]
    Ws = np.dstack(Ws) + 1e-12
    Ws = Ws / Ws.sum(axis=2, keepdims=True)
    return Ws


def weighted_sum(imgs, ws):
    return np.sum(ws * np.dstack(imgs), axis=2)


def cnn_detail_fusion(inp, model, device, relus):
    'inp: [K, 3, H, W], torch.tensor'
    model.to(device)
    model.eval()

    inp = inp.to(device)
    out = inp
    Wls = []  # upsampled L-1 feature map at each layer
    with torch.no_grad():
        for i in range(max(relus) + 1):
            out = model.features[i](out)
            if i in relus:
                l1_feat = (F.interpolate(out, inp.shape[-2:])  # upsampled activation
                            .norm(1, dim=1, keepdim=True))    # L-1 norm along channel dim
                w_l = F.softmax(l1_feat, dim=0)
                Wls.append(w_l)

    saliency_max = -np.inf * torch.ones((3,) + inp.shape[-2:])
    saliency_max = saliency_max.to(device)
    for w_l in Wls:
        saliency_curr = (inp * w_l).sum(0)
        saliency_max = torch.max(saliency_max, saliency_curr)

    fused_detail = saliency_max
    return to_numpy(fused_detail[0])


def split_YCbCr(imgs):
    Y, Y_f, CbCr_f = [None] * len(imgs), [None] * len(imgs), [None] * len(imgs)
    for i, img in enumerate(imgs):
        if is_gray(img):
            Y[i] = img
        else:
            YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            Y[i] = YCrCb[:, :, 0]
            CbCr_f[i] = (YCrCb[:, :, 1:] / 255.).astype(np.float32)
        Y_f[i] = (Y[i] / 255.).astype(np.float32)
    return Y, Y_f, CbCr_f


def YCbCr_to_RGB(CbCrs, fusedY):
    fused = fusedY
    for cbcr in CbCrs:
        if cbcr is not None:
            fused = np.dstack((fusedY, cbcr))
            fused = cv2.cvtColor(fused, cv2.COLOR_YCrCb2RGB)
            fused = np.clip(fused, 0, 1)
    return fused


# numpy and torch converter
def to_tensor(a):
    if a.ndim == 2:
        a = np.expand_dims(a, (0, 1))
    if a.ndim == 3:
        a = np.expand_dims(np.moveaxis(a, -1, 0), 0)
    return torch.from_numpy(a)


def to_numpy(t):
    a = t.squeeze().detach().cpu().numpy()
    if a.ndim == 3:
        np.moveaxis(a, 0, -1)
    return a


def stack_to_tensor(imgs: List[np.ndarray]):
    tmp = []
    for img in imgs:
        if img.dtype == np.uint8:
            img = (img / 255.).astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, (0, 1))
            tmp.append(np.repeat(img, 3, axis=1))
        # if img.ndim == 3:
        #     img = np.moveaxis(img, -1, 0)
        #     tmp.append(np.expand_dims(img, 0))
    return torch.from_numpy(np.vstack(tmp))


# image

def read_image(image_path):
    img = cv2.imread(str(image_path), -1)
    if img is None:
        raise FileNotFoundError(f'cv2 read {str(image_path)} failed')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img: np.ndarray, savePath):
    if img.dtype == np.float32:
        assert img.max() <= 1 and img.min() >= 0, f'image of dtype np.float32 should range in 0-1'
        img = np.rint(img * 255).astype(np.uint8)
    Image.fromarray(img).save(savePath)


def is_gray(img: np.ndarray):
    assert len(img.shape) <= 3, 'Wrong np.ndarray image shape in func utils.is_gray'
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    return False


def _c3(img):
    if img.ndim == 2:
        img = np.dstack((img, img, img))
    return img


def putText(img, text):
    pos = (15, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    img = cv2.putText(img, text, pos, font, 0.8, color, 2)
    return img


_TON_IMSIZE = (360, 270)


def grid_row(*imgs, resized=_TON_IMSIZE):
    row = []
    for img in imgs:
        row.append(cv2.resize(img, resized, interpolation=cv2.INTER_CUBIC))
    return row


def make_grid(nested_list, resized=_TON_IMSIZE, addText=False, hsep=7, wsep=7):
    m, n = len(nested_list), len(nested_list[0])
    w, h = resized

    gH = m * h + (m - 1) * hsep
    gW = n * w + (n - 1) * wsep

    grid = (np.ones((gH, gW, 3)) * 255).astype(np.uint8)
    for i in range(m):
        y = (h + hsep) * i
        for j in range(n):
            x = (w + wsep) * j
            this = _c3(nested_list[i][j])
            if addText:
                text = f'Input-{j+1}' if j < n - 1 else 'Fused'
                this = putText(this, text)
            grid[y:y + h, x:x + w] = this

    return grid
