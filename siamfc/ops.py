from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

#crop一块以object为中心的，变长为size大小的patch，然后将其resize成out_size的大小；
#传入size和center计算出角落坐标形式的正方形patch，即（ymin，xmin，ymax，xmax）；
#因为这样扩大的坐标有可能会超出原来的图片，所以就要计算左上角和右下角相对原图片超出多少，好进行pad，
#然后根据他们超出当中的最大值npad来在原图像周围pad，因为原图像增大了，所以Corner相对坐标也变了了。

def crop_and_resize(img, center, size, out_size,  # size是x_sz
                    border_type=cv2.BORDER_CONSTANT,  # 添加边界框像素（常数），即padding的数值为常数（这里用的是average color）
                    border_value=(0, 0, 0),  # 初始化时用的是padding 0,传參用的是·average color
                    interp=cv2.INTER_LINEAR):  # 线性插值
    # Check if image is valid
    if img is None or img.size == 0:
        print("Warning: Input image is empty")
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # convert box to corners (0-indexed)
    size = round(size)

    # Ensure size is positive and reasonable
    if size <= 0:
        print(f"Warning: Invalid size {size}, using default size 10")
        size = 10

    # Calculate corners
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)  # corners = [y_min, x_min, y_max, x_max]

    # Calculate required padding
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))

    # Handle extreme cases where padding might be excessive
    npad = max(0, int(pads.max()))

    # Limit padding to reasonable size (e.g., not more than image dimensions * 2)
    max_pad = max(img.shape[:2]) * 2
    if npad > max_pad:
        print(f"Warning: Excessive padding ({npad}) detected, limiting to {max_pad}")
        npad = max_pad

    if npad > 0:
        try:
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                border_type, value=border_value)
        except cv2.error as e:
            print(f"Error in copyMakeBorder: {e}")
            # Return a blank image if padding fails
            return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # Adjust corners with padding
    corners = (corners + npad).astype(int)

    # Ensure corners are within bounds
    corners[0] = max(0, min(corners[0], img.shape[0] - 1))
    corners[1] = max(0, min(corners[1], img.shape[1] - 1))
    corners[2] = max(corners[0] + 1, min(corners[2], img.shape[0]))
    corners[3] = max(corners[1] + 1, min(corners[3], img.shape[1]))

    # Check if crop region is valid
    if corners[2] <= corners[0] or corners[3] <= corners[1]:
        print(f"Warning: Invalid crop region: corners={corners}, image shape={img.shape}")
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # crop image patch
    try:
        patch = img[corners[0]:corners[2], corners[1]:corners[3]]

        # Check if patch is empty
        if patch.size == 0:
            print("Warning: Cropped patch is empty")
            return np.zeros((out_size, out_size, 3), dtype=np.uint8)

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size),
                           interpolation=interp)
    except cv2.error as e:
        print(f"Error in cropping/resizing: {e}")
        print(f"  corners: {corners}")
        print(f"  image shape: {img.shape}")
        print(f"  center: {center}, size: {size}, out_size: {out_size}")
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Unexpected error in crop_and_resize: {e}")
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    return patch
#    ------------------------
#    -                      -
#    -                      -
#    -  original image      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    ------------------------

#### 假如search area在original image边界里时就不用padding
#    ------------------------
#    -                      -
#    -    ++++++++++        -
#    -    +        +        -
#    -    + search +        -
#    -    + aera   +        -
#    -    ++++++++++        -
#    -                      -
#    -                      -
#    ------------------------

#### 假如searcg area超出original image边界，做padding，且以超出边界中最大那个长度来padding
#### 则新conner变为原connor+padding num，向左上、右下展开

####                      padding to
####                    ---------------->                 
####                    (左2右2上2下2 总4)

#  +++++++++++++++++                                    *+++++++++++++++**************
#  +               +                                    *+ -           +             *
#  +               +                                    *+ -           +             *
#  + --------------+---------                           *+ ------------+-----------  *
#  + -             +        -                           *+ -           +          -  *
#  + -             +        -                           *+ -           +          -  *
#  +++++++++++++++++        -                           *+++++++++++++++          -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    ------------------------                           *  ------------------------  *                          -
#                                                       *                            *
#                                                       *                            *
#                                                       ******************************  
#
