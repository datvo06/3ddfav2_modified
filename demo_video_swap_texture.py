# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA, recon_dense
from utils.render import render, render_color
from utils.functions import cv_draw_landmark, get_suffix
from utils.serialization import get_colors
from utils.serialization import ser_to_obj_multiple_colors
from utils.tddfa_util import (_parse_param, recon_dense_explicit,
                              recon_dense_base, transform_vertices)
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from FaceBoxes.utils.timer import Timer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vertices_with_colors(fp, face_boxes, tddfa):
    print(fp)
    img_orig = cv2.imread(fp)
    box = face_boxes(img_orig)[0]
    param_lst, roi_box_lst = tddfa(img_orig, [box])
    param = param_lst[0]
    roi_box = roi_box_lst[0]
    _, _, alpha_shp, alpha_exp = _parse_param(param)
    ver = recon_dense(param, roi_box, tddfa.size)
    ver_base = recon_dense_base(alpha_shp, alpha_exp)
    color = get_colors(img_orig, ver)
    return np.transpose(ver_base), color, param


def load_vertices_set(dirp, fnames, face_boxes, tddfa):
    '''
    Out: torch tensor
    '''
    v_neutral, c_neutral, param = load_vertices_with_colors(
        os.path.join(dirp, fnames[0]), face_boxes, tddfa)
    v_set = []
    c_set = []
    for fname in fnames[1:]:
        new_v, new_c, _ = load_vertices_with_colors(os.path.join(dirp, fname),
                                                 face_boxes, tddfa)
        v_set.append(new_v)
        c_set.append(new_c)
    return torch.from_numpy(np.array(v_neutral)).float().to(device), \
        torch.from_numpy(np.array(c_neutral)).float().to(device), \
        torch.from_numpy(np.array(v_set)).float().to(device), \
        torch.from_numpy(np.array(c_set)).float().to(device), param


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
    print(v.size(), v_neutral.size())
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
    w_0 = torch.max(torch.tensor([0.]).to(device), 1.0 - sum_wset_per_v)  # N
    new_sum_wset = 1.0 - w_0                    # 1 x N
    ratio = new_sum_wset/sum_wset_per_v       # 1 x N
    w_set = w_set * ratio            # E x N
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
    min_color = torch.min(c_set, dim=0)[0]
    max_color = torch.max(c_set, dim=0)[0]
    return torch.max(min_color, torch.min(max_color, (torch.unsqueeze(w_0, -1) * c_neutral +
            torch.sum(torch.unsqueeze(w_set, -1) * c_set, dim=0)
            ).view(list(c_neutral.size())[0], 3)))



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
         'nick_crop_6.png'], face_boxes, tddfa
    )
    '''
    tv_neutral, tc_neutral, tv_set, tc_set, nick_param = load_vertices_set(
        'Assets4FacePaper',
        ['sam.jpg', 'sam_1.jpg', 'sam_2.jpg',
         'sam_3.jpg', 'sam_4.jpg', 'sam_5.jpg',
         'sam_6.jpg'], face_boxes, tddfa
    )
    '''

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

    N = list(tv_neutral.size())[0]
    E = list(tv_set.size())[0]
    # run
    pre_ver = None
    total_timer = Timer()
    optimizing_timer = Timer()


    #create axes
    ax1 = plt.subplot(111)
    for i, frame in tqdm(enumerate(reader)):
        total_timer.tic()
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
            R, offset, alpha_shp , alpha_exp = _parse_param(param_lst[0])
            roi_box = roi_box_lst[0]

            ver = recon_dense_explicit(R, offset, thanh_alpha_shp, alpha_exp,
                                       roi_box, tddfa.size)
            # ver_base = recon_dense_base(nick_alpha_shp, alpha_exp)
            ver_base = recon_dense_base(thanh_alpha_shp, alpha_exp)
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver],
                                           crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            R, offset, alpha_shp, alpha_exp = _parse_param(param_lst[0])
            # ver_base = recon_dense_base(nick_alpha_shp, alpha_exp)
            ver_base = recon_dense_base(thanh_alpha_shp, alpha_exp)
            ver = recon_dense_explicit(R, offset, thanh_alpha_shp, alpha_exp,
                                       roi_box, tddfa.size)

        # Write object
        pre_ver = ver  # for tracking
        ver = torch.from_numpy(ver_base.transpose()).float().to(device)
        optimizing_timer.tic()
        a_set = fit_coeffs_aio(ver, sv_neutral, delta_set_thanh, 10)
        # a_set = fit_coeffs_aio(ver, tv_neutral, delta_set_nick, 100)
        # print(a_set)

        tver = torch.matmul(delta_set_nick.view(E, -1).transpose(0, 1), a_set).view(
            N, 3) + tv_neutral
        x1, y1, x2, y2 = roi_box
        '''
        print("Roi box: ", x1, y1, x2, y2)
        print("offset: ", offset)
        x2 = x2 - x1
        y2 = y2 - y1
        x1 = 0
        y1 = 0
        '''

        tver = transform_vertices(R, offset, tver.cpu().numpy().transpose(),
                                  roi_box, tddfa.size)         # 3 N
        cver = calc_colors_w_act_mask(a_set, nick_act_masks, tc_neutral,
                                      tc_set).cpu().numpy()

        optimizing_time = optimizing_timer.toc()
        rendered = cv2.cvtColor(
            render_color(frame, [tver], cver.astype(np.float32)),
            cv2.COLOR_BGR2RGB)
        if i == 0:
            im1 = ax1.imshow(rendered)
            plt.ion()
        else:
            im1.set_data(rendered)
        plt.pause(0.05)
        # plt.show()
        # rendered = render(np.zeros((int(x2)+1, int(y2)+1, 3), dtype=np.uint8), [tver])
        # cv2.imshow("reenacrtment", rendered)
        # cv2.imwrite("rendered.jpg", rendered)
        execution_time = total_timer.toc()
        print("Execution time: {}, Total time: {}".format(optimizing_time,
                                                          execution_time))
        '''
        tver = torch.matmul(delta_set_thanh.view(E, -1).transpose(0, 1), a_set).view(
            N, 3) + sv_neutral
        tver = transform_vertices(R, offset, tver.numpy().transpose(),
                                  roi_box, tddfa.size)         # 3 N
        cver = calc_colors_w_act_mask(a_set, thanh_act_masks, sc_neutral,
                                      sc_set).numpy()

        print(cver.shape)
        ser_to_obj_multiple_colors(cver, [tver],
                                   height=int(nick_orig.shape[0]*1.5),
                                   wfp=str(i) +'.obj')
        print(f'Dump to {str(i)}.obj')
        '''
        if i > 1000:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '3d'])
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
