import cv2
import os
import glob
import pandas as pd
import json
import shutil
import sys
sys.path.append('.')
from utils.utils_xview import coord_iou

def check_prd_gt_iou(f, score_thres=0.01, iou_thres=0.2):
    img = cv2.imread(f)
    image_name = os.path.basename(f)
    img_size = img.shape[0]
    gt_rare_cat = pd.read_csv(os.path.join(txt_save_dir, image_name.replace('.jpg', '.txt')), header=None, delimiter=' ')
    gt_rare_cat = gt_rare_cat.to_numpy()
    gt_rare_cat[:, 1:] = gt_rare_cat[:, 1:] * img_size
    gt_rare_cat[:, 1] = gt_rare_cat[:, 1] - gt_rare_cat[:, 3]/2
    gt_rare_cat[:, 2] = gt_rare_cat[:, 2] - gt_rare_cat[:, 4]/2
    gt_rare_cat[:, 3] = gt_rare_cat[:, 1] + gt_rare_cat[:, 3]
    gt_rare_cat[:, 4] = gt_rare_cat[:, 2] + gt_rare_cat[:, 4]
    res_list = glob.glob('{}/{}/{}/test_on_{}_{}_sd{}/results_{}_*.json'.format(result_dir, synfolder, cmt, real_maker, hyp_cmt, sd, synfolder))
    # print(res_list)
    prd_lbl_rare = json.load(open(res_list[0])) # xtlytlwh
    for px, p in enumerate(prd_lbl_rare):
        # if p['image_name'] != 'IMG_1222_1.jpg':
        #     continue
        if p['image_name'] == image_name and p['score'] > score_thres:
            p_bbx = p['bbox']
            p_bbx[2] = p_bbx[0] + p_bbx[2]
            p_bbx[3] = p_bbx[3] + p_bbx[1]
            p_bbx = [int(x) for x in p_bbx]
            p_cat_id = p['category_id']
            g_lbl_part = gt_rare_cat[gt_rare_cat[:, 0] == p_cat_id, :]
            for g in g_lbl_part:
                g_bbx = [int(x) for x in g[1:]]
                iou = coord_iou(p_bbx, g[1:])
                # print('iou', iou)
                if iou >= iou_thres:
                    print(iou)
                    img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (0, 255, 255), 2)
                    cv2.putText(img, text='conf:{:.4f} iou:{:.3f}'.format(p['score'], iou), org=(p_bbx[0] + 10, p_bbx[1] + 10), # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
                img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2)
    
            cv2.imwrite(os.path.join(iou_check_save_dir,  image_name), img)


if __name__ == '__main__':
    img_dir = '/data/users/yang/data/wind_turbine/wdt_crop'
    txt_save_dir = img_dir + '_label_xcycwh'
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    synfolder = 'syn_WDT'
    real_maker = 'xilin_wdt'
    epochs = 110
    iters = 151
    # hyp_cmt = 'hgiou1_1gpu_{}iter_{}epc'.format(iters, epochs)
    # hyp_cmt = 'hgiou1_1gpu_xilinratio_{}iter_{}epc'.format(iters, epochs)
    hyp_cmt = 'hgiou1_1gpu_xilinratio_aughsv_{}iter_{}epc'.format(iters, epochs)
    sd = 0
    iou_check_save_dir = img_dir + '_gt_prd_bbox'
    if not os.path.exists(iou_check_save_dir):
        os.mkdir(iou_check_save_dir)
    else:
        shutil.rmtree(iou_check_save_dir)
        os.mkdir(iou_check_save_dir)
    cmt = 'syn_wdt_BlueSky_step60_seed17'
    result_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output'

    score_thres=0.1
    iou_thres=0.5
    for f in img_files:
        check_prd_gt_iou(f, score_thres, iou_thres)


   