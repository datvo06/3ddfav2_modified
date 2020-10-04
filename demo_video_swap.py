# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA, recon_dense
from utils.render import render
from utils.functions import cv_draw_landmark, get_suffix
from utils.serialization import get_colors
from utils.serialization import ser_to_obj_multiple_colors
from utils.tddfa_util import (_parse_param, recon_dense_explicit,
                              recon_dense_base)
import numpy as np
import os
import cv2


def load_vertices_with_colors(fp, face_boxes, tddfa):
    img_orig = cv2.imread(fp)
    box = face_boxes(img_orig)[0]
    param_lst, roi_box_lst = tddfa(img_orig, [box])
    param = param_lst[0]
    roi_box = roi_box_lst[0]
    _, _, alpha_shp, alpha_exp = _parse_param(param)
    ver = recon_dense(param, roi_box, tddfa.size)
    color = get_colors(img_orig, ver)
    return ver, color, param


def load_vertices_set(dirp, fnames, face_boxes, tddfa):
    v_neutral, c_neutral, param = load_vertices_with_colors(
        os.path.join(dirp, fnames[0]), face_boxes, tddfa)
    v_set = []
    c_set = []
    for fname in fnames[1:]:
        new_v, new_c = load_vertices_with_colors(os.path.join(dirp, fname),
                                                 face_boxes, tddfa)
        v_set.append(new_v)
        c_set.append(new_c)
    return np.array(v_neutral), np.array(c_neutral), \
        np.array(v_set), np.array(c_set), param


def main(args):
    # Init TDDFA or TDDFA_ONNX
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    if args.onnx:
        from TDDFA_ONNX import TDDFA_ONNX
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)

    # Initialize FaceBoxes
    face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']
    dense_flag = args.opt in ('3d',)
    nick_orig = cv2.imread('Assets4FacePaper/nick.bmp')
    tv_neutral, tc_neutral, tv_set, tc_set, nick_param = load_vertices_set(
        'Assets4FacePaper',
        ['nick.bmp', 'nick_crop_1.jpg', 'nick_crop_2.jpg',
         'nick_crop_3.jpg', 'nick_crop_4.jpg', 'nick_crop_5.jpg',
         'nick_crop_6.jpg'], face_boxes, tddfa
    )
    sv_neutral, sc_neutral, sv_set, sc_set, thanh_param = load_vertices_set(
        'Assets4FacePaper',
        ['Thanh.jpg', 'Thanh_1.jpg', 'Thanh_2.jpg',
         'Thanh_3.jpg', 'Thanh_4.jpg', 'Thanh_5.jpg',
         'Thanh_6.jpg'], face_boxes, tddfa
    )

    _, _, nick_alpha_shp, nick_alpha_exp = _parse_param(nick_param)
    _, _, thanh_alpha_shp, thanh_alpha_exp = _parse_param(thanh_param)
    nick_ver = tc_neutral
    nick_color = tc_neutral


    suffix = get_suffix(args.video_fp)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'

    # run
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            R, offset, _ , alpha_exp = _parse_param(param_lst[0])
            ver = recon_dense_explicit(R, offset, nick_alpha_shp, alpha_exp,
                                       roi_box_lst[0], tddfa.size)

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            R, offset, _ , alpha_exp = _parse_param(param_lst[0])
            ver = recon_dense_explicit(R, offset, nick_alpha_shp, alpha_exp,
                                       roi_box_lst[0], tddfa.size)

        # Write object
        pre_ver = ver  # for tracking

        ser_to_obj_multiple_colors(nick_color, [ver],
                                   height=int(nick_orig.shape[0]*1.5),
                                   wfp=str(i) +'.obj')
        print(f'Dump to {str(i)}.obj')
        if i > 1000:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '3d'])
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
