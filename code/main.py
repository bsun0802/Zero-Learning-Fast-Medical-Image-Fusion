import sys
import pkg_resources
import argparse
from pathlib import Path
from subprocess import check_call

import torch
from torchvision.models.vgg import vgg19

from skimage.measure import shannon_entropy
from pytorch_msssim import SSIM, MS_SSIM

from utils import *
from metrics import *


def parse_args():
    '''Usage:
        python main.py --imagePath=../images/IV_images --imageSource "VIS*.png" "IR*.png"
        python main.py --imagePath=../images/MRI-SPECT --imageSource "MRI*.png" "SPECT*.png"
        python main.py --imagePath=../images/MRI-PET --imageSource "MRI*.png" "PET*.png"
    '''
    parser = argparse.ArgumentParser(description='Image Fusion with guided filter and vgg19')
    parser.add_argument('--imagePath', required=True)
    parser.add_argument('--imageSources', required=True, nargs='+')
    args = parser.parse_args()
    return args


class Args:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    k = (30, 30)  # box blur kernel size
    r, eps = 45, 0.01  # parameters for guided filter

    model = vgg19(pretrained=True)
    relus = [1, 3, 8]  # relus for vgg19

    args = parse_args()

    imagePath = Path(args.imagePath)
    # put images needed to be fused together in a bundle
    imageSources = []
    for pattern in args.imageSources:
        imageSources.append(sorted(imagePath.glob(pattern)))
    bundles = zip(*imageSources)

    # grid_cell_size = (360, 270)  # for IV_images
    grid_cell_size = (320, 320)  # for brain MRI/CT images
    resultPath = Path('../results').joinpath(imagePath.stem)

    # install package
    _installed = {pkg.key for pkg in pkg_resources.working_set}
    if 'pytorch-msssim' not in _installed:
        _python = sys.executable
        check_call(['echo', '[INFO] Install pytorch-msssim'])
        _pipinstall = ['sudo', _python, '-m', 'pip', 'install', 'pytorch-msssim']
        check_call(_pipinstall)


if __name__ == '__main__':
    container = dict(SCD=[], SSIM_f=[], MSSSIM_f=[], Qmi=[], EN=[], MI=[])
    ssim_measure = SSIM(data_range=1.0, size_average=True, channel=1)
    msssim_measure = MS_SSIM(data_range=1.0, size_average=True, channel=1)

    nested_list = []
    Args.resultPath.mkdir(parents=True, exist_ok=True)

    for bundle in Args.bundles:
        print(f'Fusing => f{[fp.name for fp in bundle]}')
        imgs = [read_image(fp) for fp in bundle]  # np.uint8

        Ys, Ys_f, CbCrs_f = split_YCbCr(imgs)

        bases, details = decompose(Ys_f)

        Wb_0 = sal_weights(Ys)
        Wb_0 = np.moveaxis(Wb_0, -1, 0)  # easier indexed in for-loop
        Wb = guided_optimize(Ys_f, Wb_0, Args.r, Args.eps)

        fused_base = weighted_sum(bases, Wb)

        tensor_details = stack_to_tensor(details)
        fused_detial = cnn_detail_fusion(
            tensor_details, Args.model, Args.device, relus=Args.relus)

        fusedY_f = np.clip(fused_base + fused_detial, 0, 1)

        fused_f = YCbCr_to_RGB(CbCrs_f, fusedY_f)

        fused_u8 = np.rint(fused_f * 255).astype(np.uint8)
        name = ''.join(x for x in bundle[0].name if x.isdigit())
        save_image(fused_u8, Args.resultPath.joinpath(f'FUSED-{name}.png'))

        nested_list.append(grid_row(*imgs, fused_u8, resized=Args.grid_cell_size))

        if len([x for x in CbCrs_f if x is not None]) == 0:
            print('Evaluation..')
            container['SCD'].append(SCD(Ys_f[0], Ys_f[1], fused_f))
            container['SSIM_f'].append(
                ssim_f(to_tensor(Ys_f[0]),
                       to_tensor(Ys_f[1]), to_tensor(fused_f), measure=ssim_measure)
            )
            container['MSSSIM_f'].append(
                ssim_f(to_tensor(Ys_f[0]),
                       to_tensor(Ys_f[1]), to_tensor(fused_f), measure=msssim_measure)
            )
            container['Qmi'].append(Qmi(Ys_f[0], Ys_f[1], fused_f))
            container['EN'].append(shannon_entropy(fused_u8))
            container['MI'].append(mi_f(Ys_f[0], Ys_f[1], fused_f))

            for k, v in container.items():
                print(f'{k} : {np.mean(v):.4f}')

            print('Done!\n')

    grid = make_grid(nested_list, Args.grid_cell_size, addText=True)
    save_image(grid, Args.resultPath.joinpath('combined.pdf'))
