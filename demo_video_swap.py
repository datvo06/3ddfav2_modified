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
from utils.tddfa_util import _parse_param, recon_dense_explicit
import cv2


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
    nick_box = face_boxes(nick_orig)[0]
    nick_param_lst, nick_roi_box_lst = tddfa(nick_orig, [nick_box])
    nick_param = nick_param_lst[0]
    nick_roi_box = nick_roi_box_lst[0]
    _, _, nick_alpha_shp, nick_alpha_exp = _parse_param(nick_param)
    nick_ver = recon_dense(nick_param, nick_roi_box, tddfa.size)
    nick_color = get_colors(nick_orig, nick_ver)


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
