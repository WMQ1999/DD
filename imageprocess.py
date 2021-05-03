import numpy as np 
import torch
import torchvision
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode

def padding_frame(img, frame_sz, center_x, center_y, patch_sz, avg_chan):
    margin_sz = patch_sz/2
    padding_left = max(0, -int(np.round(center_x - margin_sz)))
    padding_top = max(0, -int(np.round(center_y - margin_sz)))
    padding_right = max(0, int(np.round(margin_sz + center_x  - frame_sz[-1])))
    padding_bottom = max(0, int(np.round(margin_sz + center_y - frame_sz[-2])))
    padding_img = torchvision.transforms.functional.pad(img, [padding_left, padding_top, padding_right, padding_bottom], avg_chan)
    return padding_img, padding_left, padding_top


def crop_frame_z(img, center_x, center_y, x_offset, y_offset, patch_sz, re_sz):
    margin_sz = patch_sz/2
    pos_x = int(center_x + x_offset - margin_sz)
    pos_y = int(center_y + y_offset - margin_sz)
    crop_img = torchvision.transforms.functional.resized_crop(img, pos_y, pos_x, int(patch_sz), int(patch_sz), size=(re_sz, re_sz))
    #crop_img = torchvision.transforms.functional.resize(crop_img, (re_sz, re_sz))
    return crop_img


# def crop_frame_x(img, center_x, center_y, x_offset, y_offset, patch_sz, re_sz):
#     patch_sz0, patch_sz1, patch_sz2 = patch_sz
#     margin_sz = patch_sz2 / 2
#     pos_x = int(center_x + x_offset - margin_sz)
#     pos_y = int(center_y + y_offset - margin_sz)

#     crop_img = torchvision.transforms.functional.crop(img, pos_y, pos_x, int(patch_sz2), int(patch_sz2))
#     crop_img2 = torchvision.transforms.functional.resize(crop_img, size=(re_sz, re_sz))
    
#     offset_0 = int((patch_sz2 - patch_sz0) / 2)
#     offset_1 = int((patch_sz2 - patch_sz1) / 2)
#     crop_img0 = torchvision.transforms.functional.resized_crop(crop_img, offset_0, offset_0, int(patch_sz0), int(patch_sz0), size=(re_sz, re_sz))
#     crop_img1 = torchvision.transforms.functional.resized_crop(crop_img, offset_1, offset_1, int(patch_sz1), int(patch_sz1), size=(re_sz, re_sz))

#     return [crop_img0, crop_img1, crop_img2]

def crop_frame_x(img, center_x, center_y, x_offset, y_offset, patch_sz, re_sz):
    crop_imgs = []  
    for sz in patch_sz:
        pos_x = int(center_x + x_offset - sz/2)
        pos_y = int(center_y + y_offset - sz/2)
        crop_img = torchvision.transforms.functional.resized_crop(img, pos_y, pos_x, int(sz), int(sz), [re_sz, re_sz])
        crop_imgs.append(torch.clone(crop_img))

    return crop_imgs


def show_frame(img, bbox, fig_n, color = 'r', save_path = None):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=color, fill=False)
    img = img.to(torch.uint8)
    img = img.permute((1,2,0))
    ax.imshow(img)
    ax.add_patch(r)
    plt.ion()
    plt.show()
    if save_path:
        fig.savefig(save_path)
    plt.pause(0.001)
    plt.clf()

def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)