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
import torch


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
    '''
    Out: torch tensor
    '''
    v_neutral, c_neutral, param = load_vertices_with_colors(
        os.path.join(dirp, fnames[0]), face_boxes, tddfa)
    v_set = []
    c_set = []
    for fname in fnames[1:]:
        new_v, new_c, param = load_vertices_with_colors(os.path.join(dirp, fname),
                                                 face_boxes, tddfa)
        v_set.append(new_v)
        c_set.append(new_c)
    return torch.from_numpy(np.array(v_neutral)).float(), \
        torch.from_numpy(np.array(c_neutral)).float(), \
        torch.from_numpy(np.array(v_set)).float(), \
        torch.from_numpy(np.array(c_set)).float(), param


def get_delta_vertices_set(v_neutral, v_set):
    '''
    Input:
        v_neutral: N x 3
        v_set: E x N x 3
    '''
    return v_set - v_neutral


def fit_coeffs_aio(v, v_neutral, delta_set, reg_fact=1000):
    '''
    Input:
        v: N x 3
        v_neutral: Nx 3
        delta_set: E x N x 3
    Output:
        w: E x 1
    '''
    N = list(v.size())[0]
    E = list(delta_set.size())[0]
    delta_0 = (v - v_neutral).view(1, 3*N)     # N x 3
    delta_set = delta_set.view(E, 3*N).float()  # E 3N
    inv_coeffs = torch.inverse(torch.matmul(delta_set, delta_set.transpose(0, 1)) + reg_fact).float()  # E E
    rhs = torch.matmul(delta_0, delta_set.transpose(0, 1))      # 1 E
    return torch.transpose(torch.matmul(rhs, inv_coeffs), 0, 1) # E 1


def calc_activation_masks(v_neutral, v_set):
    '''
    Inputs:
        v: N x 3
        v_set: E x N x 3
    Output:
        A: E x N
    '''
    return torch.sum(torch.pow(v_neutral - v_set, 2), dim=-1).float()  # K x N


def normalize_coeffs(a_set, act_masks):
    '''
        act_masks: E x N
        a_set: E x 1
    Return:
        w_0: N
        w_set: E x N
    '''
    w_set = a_set * act_masks   # E N
    sum_wset_per_v = torch.sum(w_set, dim=0) # N
    w_0 = torch.max(torch.tensor([0.]), 1.0 - sum_wset_per_v)  # N
    new_sum_wset = 1.0 - w_0                    # 1 x N
    ratio = new_sum_wset/sum_wset_per_v       # 1 x N
    w_set = w_set* ratio            # E x N
    return w_0, w_set


def calc_colors_w_act_mask(a_set, act_masks, c_neutral, c_set):
    '''
    Input:
        a_set: E x 1(E : num blendshape)
        act_masks: E x N
        c_set: E x N x 3
        c_neutral: N x 3
    '''
    w_0, w_set = normalize_coeffs(a_set, act_masks)
    return (torch.unsqueeze(w_0, -1) * c_neutral +
            torch.sum(torch.unsqueeze(w_set, -1) * c_set, dim=0)
            ).view(list(c_neutral.size())[0], 3).transpose(0, 1)


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
        ['sam_0_0.jpg', 'sam_0_1.jpg', 'sam_0_2.jpg',
         'sam_0_3.jpg', 'sam_0_4.jpg', 'sam_0_5.jpg',
         'sam_0_6.jpg'], face_boxes, tddfa
    )
    sv_neutral, sc_neutral, sv_set, sc_set, thanh_param = load_vertices_set(
        'Assets4FacePaper',
        ['Thanh.jpg', 'Thanh1.jpg', 'Thanh2.jpg',
         'Thanh3.jpg', 'Thanh4.jpg', 'Thanh5.jpg',
         'Thanh6.jpg'], face_boxes, tddfa
    )

    _, _, nick_alpha_shp, nick_alpha_exp = _parse_param(nick_param)
    nick_act_masks = calc_activation_masks(tv_neutral, tv_set)
    delta_set_nick = get_delta_vertices_set(tv_neutral, tv_set)

    _, _, thanh_alpha_shp, thanh_alpha_exp = _parse_param(thanh_param)
    thanh_act_masks = calc_activation_masks(sv_neutral, sv_set)
    delta_set_thanh = get_delta_vertices_set(sv_neutral, sv_set)

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
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver],
                                           crop_policy='landmark')
            R, offset, _ , alpha_exp = _parse_param(param_lst[0])
            ver = recon_dense_explicit(R, offset, nick_alpha_shp, alpha_exp,
                                       roi_box_lst[0], tddfa.size)

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver],
                                           crop_policy='landmark')

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
                                   wfp='./sam_exp_only_obj/' + str(i) +'.obj')
        print(f'Dump to sam_exp_only_obj/{str(i)}.obj')
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
