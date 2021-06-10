'''
creater xuyang_ustb@163.com
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
import sys
sys.path.append('.')
import pandas as pd
import shutil
from pytorchyolo.utils.object_score_util import get_bbox_coords_from_annos_with_object_score_WDT as wdt
from pytorchyolo.utils.utils import kmean_anchors
IMG_FORMAT = '.jpg'
TXT_FORMAT = '.txt'


def group_object_annotation_and_draw_bbox(cmt, px_thresh=20, whr_thres=4, suffix="_xcycwh"):
    '''
    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object,
    and draw bbox for each ground truth files
    '''
    folder_name = '{}_annos'.format(cmt)
    lbl_path = os.path.join(syn_args.syn_data_dir, folder_name)
    print('lbl_path', lbl_path)
    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_all_annos_txt'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres)
    
    syn_annos_dir = syn_args.syn_annos_dir + suffix
    save_txt_path = os.path.join(syn_annos_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)
    wdt.get_object_bbox_after_group(lbl_path, save_txt_path, label_id=0, min_region=syn_args.min_region,
                                    link_r=syn_args.link_r, px_thresh=px_thresh, whr_thres=whr_thres, suffix=suffix)

    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))
    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_all_annos_with_bbox'.format(syn_args.min_region, syn_args.link_r,
                                                                                       px_thresh, whr_thres)
    syn_box_dir = syn_args.syn_box_dir + suffix
    save_bbx_path = os.path.join(syn_box_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
    for g in gt_files:
        gt_name = g.split('/')[-1]
        txt_name = gt_name.replace(IMG_FORMAT, TXT_FORMAT)
        txt_file = os.path.join(save_txt_path, txt_name)
        wdt.plot_img_with_bbx(g, txt_file, save_bbx_path, suffix=suffix)
    
    # hline_folder_name = 'minr{}_linkr{}_px{}whr{}_with_vline'.format(syn_args.min_region, syn_args.link_r,
    #                                                                                    px_thresh, whr_thres)
    # save_hline_path = os.path.join(syn_box_dir, hline_folder_name)
    # if not os.path.exists(save_hline_path):
    #     os.makedirs(save_hline_path)
    # else:
    #     shutil.rmtree(save_hline_path)
    #     os.makedirs(save_hline_path)
    # wdt.get_vline_from_bbox(lbl_path, save_hline_path)
    
    # vline_txt_folder_name = 'minr{}_linkr{}_px{}whr{}_vline_txt'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres)
    # save_vline_txt_path = os.path.join(syn_annos_dir, vline_txt_folder_name)
    # if not os.path.exists(save_vline_txt_path):
    #     os.makedirs(save_vline_txt_path)
    # else:
    #     shutil.rmtree(save_vline_txt_path)
    #     os.makedirs(save_vline_txt_path)   
    # wdt.get_vline_txt_xtlytlxbrybr(save_hline_path, save_vline_txt_path) 


def draw_bbx_on_rgb_images(cmt, px_thresh=20, whr_thres=4, suffix='_xcycwh'):
    img_folder_name = '{}_images'.format(cmt)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, f'*{IMG_FORMAT}')))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_all_annos_txt'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres)
    syn_annos_dir = syn_args.syn_annos_dir + suffix
    annos_path = os.path.join(syn_annos_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_all_images_with_bbox'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres)
    syn_box_dir = syn_args.syn_box_dir + suffix
    save_bbx_path = os.path.join(syn_box_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    print(len(img_files))
    for ix, f in enumerate(img_files):
        txt_file = os.path.join(annos_path, os.path.basename(f).replace(IMG_FORMAT, TXT_FORMAT))
        # print('txt_file', txt_file)
        wdt.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=False, suffix=suffix)


def split_syn_wdt_trn_val(seed=17, comment='syn_color', pxwhr='px15whr6', suffix='_xtlytlxbrybr'):
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_images'.format(comment), '*' + IMG_FORMAT)))
    num_files = len(all_files)

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    data_txt_dir = syn_args.syn_txt_dir
    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    num_val = int(num_files*syn_args.val_percent)
    syn_annos_dir = syn_args.syn_annos_dir + suffix
    lbl_dir = os.path.join(syn_annos_dir, 'minr{}_linkr{}_{}_all_annos_txt'.format(syn_args.min_region, syn_args.link_r, pxwhr))
    for i in all_indices[:num_val]:
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[num_val:]:
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()


def create_syn_data(comment='syn_texture', seed=17):
    data_txt_dir = syn_args.syn_txt_dir

    data_txt = open(os.path.join(data_txt_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
    data_txt.write('train={}/{}_train_img_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('train_label={}/{}_train_lbl_seed{}.txt\n'.format(data_txt_dir, comment, seed))

     #********** syn_0_xview_number corresponds to train*.py the number of train files
    df = pd.read_csv(os.path.join(data_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), header=None)
    data_txt.write('syn_wdt_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))

    data_txt.write('valid={}/{}_val_img_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('valid_label={}/{}_val_lbl_seed{}.txt\n'.format(data_txt_dir, comment, seed))
    data_txt.write('names=data_wdt/wdt.names\n')
    # data_txt.write('eval={}\n'.format(comment))
    # data_txt.write('backup=backup/\n')
    data_txt.close()


def creat_real_test_data(real_folder='xilin_wdt'):
    data_txt_dir = syn_args.real_txt_dir.format(real_folder)
    if not os.path.exists(data_txt_dir):
        os.mkdir(data_txt_dir)
    val_img_txt = open(os.path.join(data_txt_dir, '{}_img_test.txt'.format(real_folder)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_lbl_test.txt'.format(real_folder)), 'w')
    annos_dir = syn_args.real_annos_dir
    annos_files = glob.glob(os.path.join(annos_dir, '*.txt'))
    imgs_dir = syn_args.real_img_dir
    for f in annos_files:
        val_lbl_txt.write('%s\n' % f)
        val_img_txt.write('%s\n' % os.path.join(imgs_dir, os.path.basename(f).replace('.txt', '.jpg')))
    val_img_txt.close()
    val_lbl_txt.close()
    data_txt = open(os.path.join(data_txt_dir, '{}_real_test.data'.format(real_folder)), 'w')
    data_txt.write('test={}/{}_img_test.txt\n'.format(data_txt_dir, real_folder))
    data_txt.write('test_label={}/{}_lbl_test.txt\n'.format(data_txt_dir, real_folder))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('names=data_wdt/wdt.names\n')
    data_txt.close()


def get_args(cmt=''):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--syn_base_dir", type=str,
                        help="base path of synthetic data",
                        default='/data/users/yang/data/synthetic_data_wdt')
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='{}/{}/')
    parser.add_argument("--syn_annos_dir", type=str, default='{}/{}_txt',
                        help="syn label.txt")
    parser.add_argument("--syn_box_dir", type=str, default='{}/{}_gt_bbox',
                        help="syn box files")
    parser.add_argument("--syn_txt_dir", type=str, default='data_wdt/{}',
                        help="syn related txt files")
    
    parser.add_argument("--real_base_dir", type=str, default='/data/users/yang/data/wind_turbine/',
                        help="real base dir")
    parser.add_argument("--real_img_dir", type=str, default='{}/wdt_crop',
                        help="real img files files")
    parser.add_argument("--real_annos_dir", type=str, default='{}/wdt_crop_label_xcycwh',
                        help="real label files")
    parser.add_argument("--real_txt_dir", type=str, default='data_wdt/{}',
                        help="real related txt list in the project")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=50, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.2, help="train:val=0.8:0.2")

    args = parser.parse_args()
    args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
    args.syn_annos_dir = args.syn_annos_dir.format(args.syn_base_dir, cmt)
    args.syn_txt_dir = args.syn_txt_dir.format(cmt)
    args.syn_box_dir = args.syn_box_dir.format(args.syn_base_dir, cmt)
    args.real_img_dir = args.real_img_dir.format(args.real_base_dir)
    args.real_annos_dir = args.real_annos_dir.format(args.real_base_dir)

    # if not os.path.exists(args.syn_annos_dir):
    #     os.makedirs(args.syn_annos_dir)
    if not os.path.exists(args.syn_txt_dir):
        os.makedirs(args.syn_txt_dir)
    # if not os.path.exists(args.syn_box_dir):
    #     os.makedirs(args.syn_box_dir)

    return args


if __name__ == '__main__':

    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    ################################# 
    #######  UAV
    #######
    px_thres= 15 # 23
    whr_thres= 4 # 3
    # cmt = 'syn_wdt_BlueSky_step60'
    cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_step100'
    syn_args = get_args(cmt)
    
    # group_object_annotation_and_draw_bbox(cmt, px_thres, whr_thres, suffix="_xcycwh")
    # group_object_annotation_and_draw_bbox(cmt, px_thres, whr_thres, suffix="_xtlytlxbrybr")
    
    # wdt.get_hline_of_annos_from_bbox()
    # wdt.separate_wdt_bbox()
    # get_annos_from_lbl_image_and_draw_bbox_by_cv_dilation(cmt, px_thres, whr_thres) ## the b-box is not exactly correct

    '''
    draw bbox on rgb images for syn_background data
    '''
    # seed = 17
    # px_thres= 15 # 23
    # whr_thres= 4 # 3
    # # cmt = 'syn_wdt_BlueSky_step60'
    # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_step100'
    # syn_args = get_args(cmt)
    # draw_bbx_on_rgb_images(cmt, px_thres, whr_thres, suffix="_xcycwh")


    '''
    split train val
    '''
    # seed = 17
    # px_thres= 15 # 23
    # whr_thres= 4 # 3
    # pxwhr = 'px{}whr{}'.format(px_thres, whr_thres)
    # base_pxwhrs = '{}_seed{}'.format(pxwhr, seed)
    # # cmt = 'syn_wdt_BlueSky_step60'
    # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_step100'
    # syn_args = get_args(cmt)
    # base_pxwhrs = base_pxwhrs.format(seed)
    # split_syn_wdt_trn_val(seed, cmt, pxwhr, suffix='_xcycwh')

    # ####### create *.data
    # create_syn_data(cmt, seed=17)


    #################################
    '''
    combine real  dataset
    '''
    # seed = 17
    # px_thres= 15 # 23
    # whr_thres= 4 # 3
    # pxwhr = 'px{}whr{}'.format(px_thres, whr_thres)
    # base_pxwhrs = '{}_seed{}'.format(pxwhr, seed)
    # cmt = 'syn_wdt_BlueSky_step60'
    # syn_args = get_args(cmt)
    # real_folder='xilin_wdt'
    # creat_real_test_data(real_folder=real_folder)

    ''' 
    combine real dataset
    '''
    # seed = 17
    # px_thres= 15 # 23
    # whr_thres= 4 # 3
    # pxwhr = 'px{}whr{}'.format(px_thres, whr_thres)
    # base_pxwhrs = '{}_seed{}'.format(pxwhr, seed)
    # cmt = 'syn_wdt_BlueSky_step60'
    # syn_args = get_args(cmt)
    # real_folder='xilin_wdt'
    # real_txt_dir = syn_args.real_txt_dir.format(real_folder)
    # img_path = os.path.join(real_txt_dir, '{}_img_test.txt'.format(real_folder))
    # lbl_path = os.path.join(real_txt_dir, '{}_lbl_test.txt'.format(real_folder))
    # kmean_anchors(img_path, lbl_path)

