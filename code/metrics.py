import torch
import numpy as np
from skimage.measure import shannon_entropy


def ssim_f(img1, img2, fused, measure):
    return .5 * (measure(img1, fused) + measure(img2, fused))


def mi_f(img1, img2, fused):
    return MI(img1, fused) + MI(img2, fused)


def _to_uint8(img):
    if not img.dtype == np.uint8:
        if img.dtype == np.float32:
            img = np.rint(img * 255).astype(np.uint8)
        else:
            raise TypeError('Image dtype not understand in nrfm._to_uint8')
    return img


def _joint_entropy(x, y, bins=256, base=2):
    hist2d, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
    Pxy = hist2d / hist2d.sum()
    pos = Pxy > 0
    Hxy = -np.sum(Pxy[pos] * np.log(Pxy[pos]) / np.log(base))
    return Hxy


def Qmi(img1, img2, fused):
    r"""Return the mutual information index (Qmi).

    Reference: M. Hossny, S. Nahavandi, and D. Creighton, “Comments on information measure for performance of image fusion,” Electronics letters, vol. 44, no. 18, 2008.

    Args:
        img1 (np.ndarray): first source image
        img2 (np.ndarray): second source image
        fused (np.ndarray): the fused image

        All image tensor are supposed to be in grayscale.

    Returns:
        np.float: the Qmi value of image fusion
    """
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if len(img1.shape) != 2 or len(img2.shape) != 2:
        raise TypeError(f'Expected source images to be in grayscale(2-d), '
                        f'but got {img1.shape} and {img2.shape}')
    if img1.shape != img2.shape:
        raise TypeError(f'Expected tensors of equal shapes, '
                        f'but got {img1.shape} and {img2.shape}')
    I_AF = MI(img1, fused)
    I_BF = MI(img2, fused)
    en = shannon_entropy
    EN_A, EN_B, EN_F = en(_to_uint8(img1)), en(_to_uint8(img2)), en(_to_uint8(fused))
    Qmi = 2. * (I_AF / (EN_A + EN_F) + I_BF / (EN_B + EN_F))
    return Qmi


def MI(img1, img2, bins=256):
    r"""Return the mutual information (MI).

    Reference: [1] https://en.wikipedia.org/wiki/Mutual_information
               [2] https://users.cs.duke.edu/~tomasi/papers/russakoff/russakoffEccv04.pdf

    Args:
        img1 (np.ndarray): first image
        img2 (np.ndarray): second image

    Returns:
        np.float: the MI of two images
    """
    hist2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=0, keepdims=True)
    py = pxy.sum(axis=1, keepdims=True)
    px_py = py * px
    pos = pxy > 0
    mi = np.sum(pxy[pos] * np.log2(pxy[pos] / px_py[pos]))
    return mi


def _corr2d(A, B):
    r"""Calfulate 2-D correlation coefficient between matrix A and B.

    Args:
        A (torch.tensor or np.ndarray): a tensor of shape [H, W]
        B (torch.tensor or np.ndarray): a tensor of shape [H, W]

    Returns:
        0-dim torch.Tensor or np.float: the 2-D correlation coefficient result
    """
    sqrt = np.sqrt if type(A) is np.ndarray else torch.sqrt
    Abar, Bbar = A.mean(), B.mean()
    Adiff, Bdiff = Abar - A, Bbar - B
    numer = (Adiff * Bdiff).sum()
    denom = sqrt((Adiff**2).sum() * (Bdiff**2).sum())
    return numer / denom


def SCD(img1, img2, fused):
    r"""Return the Sum of the Correlation of Difference (SCD).

    Args:
        img1 (torch.tensor or np.ndarray): first source image
        img2 (torch.tensor or np.ndarray): second source image
        fused (torch.tensor or np.ndarray): the fused image

        All image tensor are supposed to be in grayscale.

    Returns:
        0-dim torch.Tensor or np.float: the SCD value of image fusion
    """
    img1 = img1.squeeze()
    img2 = img2.squeeze()

    if len(img1.shape) != 2 or len(img2.shape) != 2:
        raise TypeError(f'Expected source images to be in grayscale(2-d), '
                        f'but got {img1.shape} and {img2.shape}')
    if img1.shape != img2.shape:
        raise TypeError(f'Expected tensors of equal shapes, '
                        f'but got {img1.shape} and {img2.shape}')

    r1 = _corr2d(img1, fused - img2)
    r2 = _corr2d(img2, fused - img1)
    return r1 + r2
