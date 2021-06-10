import os
import sys
sys.path.append('.')
import glob
import numpy as np
from PIL import Image
import pandas as pd
import cv2

from utils.object_score_util import get_bbox_coords_from_annos_with_object_score_WDT as wdt


def crop_image(img, shape=(608, 608), save_dir=''):

    height, width, _ = img.shape
    ws, hs = shape

    # w_num, h_num = (int(width / wn), int(height / hn))
    h_num = round(height / hs)
    w_num = round(width / ws)
    images = {}
    image_names = {}
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            hmin = hs * j
            hmax = hs * (j + 1)
            wmin = ws * i
            wmax = ws * (i + 1)

            chip = np.zeros_like((ws, hs, 3))
            if hmax >= height and wmax >= width:
                chip = img[hmin : height, width - ws : width, :3]
            elif hmax >= height and wmax < width:
                chip = img[height - hs : height, wmin : wmax, :3]
            elif hmax < height and wmax >= width:
                chip = img[hmin : hmax, width - ws : width, :3]
                # if h_num < 2:
                #     margin = (height - hs)//2
                #     chip = img[margin : height-margin, width - ws : width, :3]
                # else:
                #     chip = img[hmin : hmax, width - ws : width, :3]
            else:
                chip = img[hmin : hmax, wmin : wmax, :3]

            # remove figures with more than 10% black pixels
            im = Image.fromarray(chip)
            im_gray = np.array(im.convert('L'))
            name = os.path.basename(f)
            image_name = name.split('.')[0] + '_' + str(k) + '.jpg'
            im.save(os.path.join(save_dir, image_name))

            k += 1


def convert_label_string_to_int(label_string_dir, suffix='_xcycwh'):
    '''
    suffix: _xcycwh, _xyxy
    '''
    save_label_dir = label_string_dir + suffix
    if not os.path.exists(save_label_dir):
        os.mkdir(save_label_dir)
    label_files = glob.glob(os.path.join(label_string_dir, '*.txt'))
    label_id = 0
    size = 608
    for f in label_files:
        df = pd.read_csv(f, header=None)
        # print(df.shape)
        f_txt = open(os.path.join(save_label_dir, os.path.basename(f)), 'w')
        for i in range(1, df.shape[0]):
            line = df.loc[i].to_string()
            # print(line)
            coords = line.split(' ')[4:-1]
            # print(coords)
            coords = [int(c) for c in coords]
            xtl = coords[0]
            ytl = coords[1]
            xbr = coords[2]
            ybr = coords[3]
            if suffix == "_xcycwh":
                w = xbr - xtl
                h = ybr - ytl
                xc = xtl + w/2.
                yc = ytl + h/2.
                f_txt.write("%s %.4f %.4f %.4f %.4f\n" % (label_id, xc/size, yc/size, w/size, h/size))
            else: # _xyxy
                f_txt.write("%s %s %s %s %s\n" % (label_id, xtl, ytl, xbr, ybr))



def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def plot_img_with_bbx_by_suffix(img_file, lbl_file, save_path, label_id=0, label_index=False, suffix="_xcycwh"):
    if not is_non_zero_file(lbl_file):
        return
    # print(img_file)
    img = cv2.imread(img_file) # h, w, c
    h, w = img.shape[:2]

    df_lbl = pd.read_csv(lbl_file, header=None, delimiter=' ').to_numpy() # delimiter , error_bad_lines=False
    if suffix == "_xcycwh":
        df_lbl[:, 1] = df_lbl[:, 1]*w
        df_lbl[:, 3] = df_lbl[:, 3]*w

        df_lbl[:, 2] = df_lbl[:, 2]*h
        df_lbl[:, 4] = df_lbl[:, 4]*h

        df_lbl[:, 1] -= df_lbl[:, 3]/2
        df_lbl[:, 2] -= df_lbl[:, 4]/2

        df_lbl[:, 3] += df_lbl[:, 1]
        df_lbl[:, 4] += df_lbl[:, 2]
    # print(df_lbl)
    for ix in range(df_lbl.shape[0]):
        cat_id = int(df_lbl[ix, 0])
        gt_bbx = df_lbl[ix, 1:].astype(np.int64)
        # print(gt_bbx)
        
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), (255, 0, 0), 2)
        pl = ''
        if label_index:
            pl = '{}'.format(ix)
        elif label_id and df_lbl.shape[1]==6:
            mid = int(df_lbl[ix, 5])
            pl = '{}'.format(mid)
        elif label_id and df_lbl.shape[1]==5:
            mid = int(df_lbl[ix, 0])
            pl = '{}'.format(mid)
        else:
             pl = '{}'.format(cat_id)
        cv2.putText(img, text=pl, org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_file)), img)

if __name__ == "__main__":

    data_dir = '/data/users/yang/data/wind_turbine/wdt'
    # ori_images = glob.glob(os.path.join(data_dir, "*.JPG"))
    # save_dir = data_dir + "_crop"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # for f in ori_images:
    #     img = np.array(Image.open(f))
    #     crop_image(img, shape=(608, 608), save_dir=save_dir)

    label_string_dir = '/data/users/yang/data/wind_turbine/wdt_crop_label'
    # suffix = '_xyxy'
    suffix = '_xcycwh'
    convert_label_string_to_int(label_string_dir, suffix)

    
    img_dir = data_dir + "_crop"
    # suffix = '_xyxy'
    suffix = '_xcycwh'
    lbl_dir = label_string_dir + suffix
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    save_path = img_dir + suffix + '_bbox'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print(len(img_files))

    for f in img_files:
        lbl_file = os.path.join(lbl_dir, os.path.basename(f).replace('.jpg', '.txt'))
        # wdt.plot_img_with_bbx(f, lbl_file, save_path, label_id=0, label_index=False, suffix="_xcycwh")
        plot_img_with_bbx_by_suffix(f, lbl_file, save_path, label_id=0, label_index=False, suffix=suffix)
