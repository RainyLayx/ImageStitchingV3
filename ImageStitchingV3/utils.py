import argparse
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import pdb
import cv2
import os
from math import exp
from models.matching import Matching
from torchsummary import summary
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,read_image0,read_image1,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)
def show_Image_Cv(Image, Name="Demo"):
    cv2.namedWindow(Name,cv2.WINDOW_NORMAL)
    cv2.imshow(Name, Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main(opt,name0,name1,matches_path,viz_path):
    config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }
    matching = Matching(config).eval().to(opt.device)
    timer = AverageTimer(newline=True)
    
    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    stem0, stem1 = Path(name0).stem, Path(name1).stem
   
    # Handle --cache logic.
    do_match = True
    do_viz = opt.viz

    rot0, rot1 = 0, 0
    image0, image0_color,inp0, scales0 = read_image0(
        os.path.join(input_dir,name0), opt.device, opt.max_length0, rot0, opt.resize_float)
    
    # # 透视变换图片存入
    # global processedimgs
    # processedimgs.append(image0_color)

    image1, image1_color,inp1, scales1 = read_image1(
        os.path.join(input_dir,name1), opt.device, opt.max_length1, rot1, opt.resize_float)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    timer.update('load_image')

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        xy_F = kpts0 #特征点坐标
        xy_L = kpts1
        confidence = conf   #置信度
        Match = matches     #匹配对，存的是特征点的索引
        # pdb.set_trace()
        PtsA = []
        PtsB = []
        for i in range(len(Match)):
            # if confidence[i]>0.3:
                PtsA.append(xy_F[i])
                PtsB.append(xy_L[Match[i]])
        PtsA = np.float32(PtsA)
        PtsB = np.float32(PtsB)
        # pdb.set_trace()
        Mat, status = cv2.findHomography(PtsB, PtsA, cv2.RANSAC, 10)
        
        # pdb.set_trace()
        im1 = cv2.warpPerspective(image1_color, Mat, (image0_color.shape[1],int(image0_color.shape[0]+image1_color.shape[0])))
        
        flag = 1
        h_warp,w_warp = im1.shape[0:2]
        for col in range(h_warp-1,0,-1):
            if flag:
                for row in range(w_warp):
                    if (im1[col,row]!=np.array([0,0,0])).any():
                        flag = 0
                        break
            else:
                break
        im1 = im1[0:col,:]
        # cv2.imwrite(os.path.join(opt.output_dir,name1.split('.')[0]+'warp'+'.jpg'),im1)
        cv2.imwrite(str(matches_path),im1)
        
        # 作图展示特征点匹配情况
        valid = matches > -1
        valid = conf > 0.8
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        # mconf = mconf > 0.8
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]
        # 把txt里的KA KB整理起来
        make_matching_plot(
            image0_color, image1_color, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, opt.show_keypoints,
            opt.fast_viz, opt.opencv_display, 'Matches', small_text)

        timer.update('viz_match')
        return image0_color

def add_channel(img_3c):
    b_channel, g_channel, r_channel = cv2.split(img_3c) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
    img_4c = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_4c

# 透明化处理，去除透视变换后的黑色部分。
def perspective(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    B,G,R = cv2.split(img)
    _, Alpha= cv2.threshold(R, 1, 255, cv2.THRESH_BINARY)
    B2,G2,R2,A2 = cv2.split(result)
    A2 = Alpha
    img_persp = cv2.merge([B2,G2,R2,A2]) #通道合并
    return img_persp

# 拼接
def smooth(img0,img1):
    rows,cols=img0.shape[:2]
    # print('行列',rows,cols)
    flag = 1
    h_warp,w_warp = img1.shape[0:2]

    for row in range(0,h_warp):
        if flag:
            for col in range(w_warp):
                if (img1[row,col]!=np.array([0,0,0,0])).any():
                    flag = 0
                    break
        else:
            break
    top = row
    bot = img0.shape[0]
    res = np.zeros([rows, cols, 4], np.uint8)
    for row in range(0, rows):
        if row<top:
            res[row, :] = img0[row, :]
        elif row>bot:
            res[row, :] = img1[row, :]
        else:
            for col in range(0,cols):
                srcImgLen = float(abs(row - top))
                testImgLen = float(abs(row - bot))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img0[row, col] * (1-alpha) + img1[row, col] * alpha, 0, 255)
    yy1 = 0
    yy2 = res.shape[0]
    xx1 = 0
    xx2 = res.shape[1]

    alpha_png = res[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
        
    # 开始叠加
    for c in range(0,3):
        img1[yy1:yy2, xx1:xx2, c] = ((alpha_jpg*img1[yy1:yy2, xx1:xx2,c]) + (alpha_png*res[yy1:yy2,xx1:xx2,c]))

    return img1

def parse_init():
    parser = argparse.ArgumentParser(
            description='Image pair matching and pose evaluation with SuperGlue',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument(
    #     '--input_pairs', type=str, default='testdir/annos.txt',
    #     help='Path to the list of image pairs')

    parser.add_argument('--input_dir', type=str, default='./img_cache_dir',
        help='Path to the directory that contains the images')

    parser.add_argument('--output_dir', type=str, default='./img_cache_dir',
        help='The same with input_dir')

    parser.add_argument(
        '--max_length0', type=int, nargs='+', default=1080)
    parser.add_argument(
        '--max_length1', type=int, nargs='+', default=1080)
    parser.add_argument('--resize_float', action='store_true')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
                ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')

    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--device', type=str, default='cpu')
    return parser