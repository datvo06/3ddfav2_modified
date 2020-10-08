# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype, tri

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, alpha=0.6, show_flag=False, wfp=None):
    overlap = img.copy()
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res


def render_color(img, ver_lst, color, wfp=None):
    overlap = np.zeros_like(img)
    # print(overlap.shape)
    # color = _to_ctype(color)
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap, texture=color)
    res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    return res
