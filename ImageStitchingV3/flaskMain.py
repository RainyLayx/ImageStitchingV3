import time
import base64
from flask import Flask,render_template,request
import json
import os
import pdb
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
from models.matching import Matching
from torchsummary import summary
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,read_image0,read_image1,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from utils import parse_init,main,show_Image_Cv,smooth,add_channel,perspective
app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    start = time.perf_counter()
    code_dic = {'200':'请求成功','301':'请求参数不合法','401':'base64编码解码有误','402':'jsonloads报错','403':'其他错误'}
    ret_code = '200'
    
    # 图片临时存储文件夹
    img_cache_dir = r'./img_cache_dir/'
    if not(os.path.exists(img_cache_dir)):
        os.makedirs(img_cache_dir)

    try:
        # data = json.loads(request.body)
        # data_bytes = request.data
        data_bytes = request.get_data()
        data_str = str(data_bytes,'utf-8')
        data = json.loads(data_str)
    except Exception as e402:
        ret_code = '402'
        imgres_64 = ''
        image_info = ''

    if ret_code == '200':
        info_loc = data['info_loc']
        info_time = data['info_time']
        info_name = data['info_name']
        prv_code = data['prv_code']
        major_code = data['major_code']
        images = data['input_data']
        image_info = info_loc+info_time+info_name
        if len(info_loc)!=12 or len(info_time)!=12 or len(info_name)>8:
            ret_code = '301' 
            imgres_64 = ''      
        else:
            try:
                pair = []
                for image_dict in images:
                    image_id = image_dict['image_id']
                    base64_str = image_dict['image_data']
                    if isinstance(base64_str, bytes):
                        base64_str = base64_str.decode('utf-8')
                    image_data = base64.b64decode(base64_str)

                    image_name = image_info+image_id+'.jpg'
                    pair.append(image_name)
                    # 暂存输入图片
                    fwrt = open(os.path.join(img_cache_dir,image_name), "wb")
                    fwrt.write(image_data)
                    fwrt.close()
            except Exception as e401:
                    ret_code = '401'
                    imgres_64 = ''
    # 开始处理
    if ret_code == '200':
        parser = parse_init()
        opt = parser.parse_args()
        opt.input_dir = img_cache_dir
        opt.output_dir = opt.input_dir
        opt.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(opt.device))
        
        input_dir = Path(opt.input_dir)
        print('Looking for data in directory \"{}\"'.format(input_dir))
        output_dir = Path(opt.output_dir)
        print('Will write matches to directory \"{}\"'.format(output_dir))
        
        try:
            processedimgs = []
            if len(pair)==2:            #如果待拼接图像数量等于2，则直接拼接
                name0, name1 = pair[:2]
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                matches_path = output_dir / '{}_{}_match.jpg'.format(stem0, stem1)
                viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
                image0_color = main(opt,name0,name1,matches_path,viz_path)
                processedimgs.append(image0_color)
            else:                       #如果待拼接图像数量大于2，则将当前图片与前一张透视变换后的图片进行匹配。
                matches_path = ''
                for i in range(1,len(pair)):
                    if i==1:
                        name0 = pair[0]
                        stem0 = Path(name0).stem
                    else:              
                        opt.max_length0 = None
                        opt.max_length1 = 1080
                        print('matches path:',matches_path)
                        name0 = os.path.basename(matches_path)
                    name1 = pair[i]
                    stem1 = Path(name1).stem
                    matches_path = output_dir / '{}_{}_match.jpg'.format(stem0, stem1)
                    viz_path = output_dir / '{}_{}_matches_viz.{}'.format(stem0, stem1, opt.viz_extension)
                    image0_color = main(opt,name0,name1,matches_path,viz_path)
                    processedimgs.append(image0_color)
            imglast = cv2.imread(str(matches_path))
            imglast = perspective(imglast)
            
            # 透视变换后的图片全部存在processedimgs里，从后往前逐个叠加。
            for i in range(len(processedimgs)-1,-1,-1):
                img = processedimgs[i]
                img = perspective(img)
                imglast = smooth(img,imglast)
            # 把拼接后图片中的黑块去掉
            flag = 1
            h_warp,w_warp = imglast.shape[0:2]
            for col in range(h_warp-1,0,-1):
                if flag:
                    for row in range(w_warp):
                        if (imglast[col,row]!=np.array([0,0,0,0])).any():
                            flag = 0
                            break
                else:
                    break
            imglast = imglast[0:col,:]
            cv2.imwrite(str(matches_path),imglast) # 最终拼接结果
            f = open(str(matches_path), 'rb')
            imgres_64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e403:
            ret_code = '403'
            imgres_64 = ''
            print(e403)
    
    end = time.perf_counter()
    time_cost = str(round(end - start,4))
    res_data = json.dumps({'ret_code':ret_code,'ret_message':code_dic[ret_code],'image_info':image_info,'time_cost':time_cost,'image_res':imgres_64},ensure_ascii=False)
    os.system('rm ./img_cache_dir/*')
    return res_data

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5000,debug=True)#原来是127.0.0.1
