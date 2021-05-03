import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import params
import layers
from torchvision.io.image import ImageReadMode, read_image
import os
from imageprocess import *
import time
from evaluation import *
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_gt(gt_dir):
    gts = list()
    with open(gt_dir, 'r') as labels:
        for line in labels:
            temp = list()
            for num in line.split(','):
                temp.append(float(num))
            gts.append(temp)
    return gts

def update_target_position(pos_x, pos_y, score, x_sz):
    p = np.asarray(np.unravel_index(np.argmax(score), score.shape))
    center = float(params.final_score_sz - 1) / 2
    disp_in_area = p - center
    
    # disp_in_xcrop = disp_in_area * float(params.total_stride) / params.response_up
    # disp_in_frame = disp_in_xcrop * x_sz / params.search_sz
    disp_in_frame = disp_in_area * x_sz / params.final_score_sz
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    print('disp: ' + str(disp_in_frame))
    return pos_x, pos_y

def track(img_dir, model, frame_num, pos_x, pos_y, width, height):
    
    netx = model.netx
    netz = model.netz
    corr = model.corr
    netx.eval()
    netz.eval()
    corr.eval()    

    center_x, center_y = pos_x + width/2, pos_y + height/2
    bboxes = np.zeros((frame_num, 4))
    bboxes[0, :] = pos_x, pos_y, width, height
    scale_factor = params.scale_step ** np.linspace(-(params.scale_num//2), (params.scale_num//2), params.scale_num)
    context = (width + height) * params.context
    z_sz = np.sqrt((width + context) * (height + context))
    x_sz = float(params.search_sz) / params.examplar_sz * z_sz

    min_z = params.scale_min * z_sz
    max_z = params.scale_max * z_sz
    min_x = params.scale_min * x_sz
    max_z = params.scale_max * x_sz

    hanning_1d = np.expand_dims(np.hanning(params.final_score_sz), axis = 0)
    penalty = np.transpose(hanning_1d) * hanning_1d
    penalty = penalty / np.sum(penalty)

    img_path = os.path.join(img_dir, f'{1:08d}.png')
    img_z = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
    frame_sz = img_z.shape
    avg_chan = torch.mean(img_z, dim = (-1, -2))
    padding_z, x_offset, y_offset = padding_frame(img_z, img_z.shape, center_x, center_y, z_sz, avg_chan[0].item())
    crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
    crop_z = torch.unsqueeze(crop_z, dim = 0)

    #crop_z = crop_z.to(device)
    template_z = netz(crop_z).squeeze(dim = 0)
    templates_z = torch.stack([template_z] * params.scale_num)

    t_start = time.time()
    #show_frame(img_z, bboxes[0,:], 1, save_path=os.path.join(img_dir, 'siamFC\\res', f'{1:08d}.png'))

    for i in range(1, frame_num):
        z_scale = z_sz * scale_factor
        x_scale = x_sz * scale_factor
        w_scale = width * scale_factor
        h_scale = height * scale_factor
        
        
        img_path = os.path.join(img_dir, f'{i:08d}.png')
        img_x = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
        avg_chan = torch.mean(img_z, dim = (-1, -2))
        padding_x, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, x_scale[-1], avg_chan[0].item())

        x_crops = torch.stack(crop_frame_x(padding_x, center_x, center_y, x_offset, y_offset, x_scale, params.search_sz), dim=0)
        #x_crops = x_crops.to(device)
        templates_x = netx(x_crops)
        scores = corr(templates_x, templates_z)
        scores = torchvision.transforms.functional.resize(scores, (params.final_score_sz, params.final_score_sz), torchvision.transforms.InterpolationMode.BICUBIC)
        scores = torch.squeeze(scores)

        for j in range(params.scale_num):
            if scale_factor[j] != 1:
                scores[j, :, :] = params.scale_penalty * scores[j, :, :]

        scale_index = torch.argmax(torch.amax(scores, dim = (-1, -2)))

        x_sz = (1 - params.scale_lr) * x_sz + params.scale_lr * x_scale[scale_index]
        width = (1 - params.scale_lr) * width + params.scale_lr * w_scale[scale_index]
        height = (1 - params.scale_lr) * height + params.scale_lr * h_scale[scale_index]

        score = scores[scale_index, :, :]
        # print(score.numpy())
        score = score - torch.min(score)
        score = score/torch.sum(score)
        score = (1 - params.window_influence) * score + params.window_influence * penalty
        center_x, center_y = update_target_position(center_x, center_y, score, x_sz)
        center_x = max(0, center_x)
        center_x = min(center_x, frame_sz[-1])
        center_y = max(0, center_y)
        center_y = min(center_y, frame_sz[-2])
        bboxes[i-1, :] = center_x - width/2, center_y - height/2, width, height
        #z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr * z_scale[scale_index]
        
        if params.z_lr > 0:
            img_z, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, z_sz, avg_chan[0].item())
            crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
            crop_z = torch.unsqueeze(crop_z, dim = 0)
            new_template_z = netz(crop_z).squeeze()
            new_templates_z = torch.stack([new_template_z] * params.scale_num)
            templates_z = (1 - params.z_lr) * templates_z + params.z_lr * new_templates_z
        # print(center_x, center_y, width, height)
        
        z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr*z_scale[scale_index]   
        # show_frame(img_x, bboxes[i-1,:], i-1)
        print(i)
        save_path = os.path.join(img_dir, 'siamFC\\res', f'{i:08d}.png')
        show_frame(img_x, bboxes[i-1,:], 1, save_path = save_path)
    speed = frame_num / (time.time() - t_start)
    plt.close('all')
    
    return bboxes, speed 

def trackFu(img_dir, model, frame_num, pos_x, pos_y, width, height):
    netx = model.netx
    netz = model.netz
    corr = model.corr
    netx.eval()
    netz.eval()
    corr.eval()    

    center_x, center_y = pos_x + width/2, pos_y + height/2
    bboxes = np.zeros((frame_num, 4))
    bboxes[0, :] = pos_x, pos_y, width, height
    scale_factor = params.scale_step ** np.linspace(-(params.scale_num//2), (params.scale_num//2), params.scale_num)
    context = (width + height) * params.context
    z_sz = np.sqrt((width + context) * (height + context))
    x_sz = float(params.search_sz) / params.examplar_sz * z_sz

    min_z = params.scale_min * z_sz
    max_z = params.scale_max * z_sz
    min_x = params.scale_min * x_sz
    max_z = params.scale_max * x_sz

    hanning_1d = np.expand_dims(np.hanning(params.final_score_sz), axis = 0)
    penalty = np.transpose(hanning_1d) * hanning_1d
    penalty = penalty / np.sum(penalty)

    img_path = os.path.join(img_dir, f'{1:08d}.png')
    img_z = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
    frame_sz = img_z.shape
    avg_chan = torch.mean(img_z, dim = (-1, -2))
    padding_z, x_offset, y_offset = padding_frame(img_z, img_z.shape, center_x, center_y, z_sz, avg_chan[0].item())
    crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
    crop_z = torch.unsqueeze(crop_z, dim = 0)

    #crop_z = crop_z.to(device)
    z1, z2 = netz(crop_z)
    z1_sz = z1.shape[-1]
    z2_sz = z2.shape[-1]
    # z_pad = (z1_sz - z2_sz) // 2
    # z2 = nn.functional.pad(z2, (z_pad, z1_sz - z_pad - z2_sz, z_pad, z1_sz - z_pad - z2_sz))
    z2 = torch.nn.functional.upsample(z2, (z1_sz, z1_sz), mode = 'bicubic')
    template_z = torch.cat([z1, z2], dim = 1).squeeze(dim = 0)
    templates_z = torch.stack([template_z] * params.scale_num)

    t_start = time.time()
    #show_frame(img_z, bboxes[0,:], 1, save_path=os.path.join(img_dir, 'siamFC\\res', f'{1:08d}.png'))

    for i in range(1, frame_num):
        z_scale = z_sz * scale_factor
        x_scale = x_sz * scale_factor
        w_scale = width * scale_factor
        h_scale = height * scale_factor
        
        
        img_path = os.path.join(img_dir, f'{i:08d}.png')
        img_x = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
        avg_chan = torch.mean(img_z, dim = (-1, -2))
        padding_x, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, x_scale[-1], avg_chan[0].item())

        x_crops = torch.stack(crop_frame_x(padding_x, center_x, center_y, x_offset, y_offset, x_scale, params.search_sz), dim=0)
        #x_crops = x_crops.to(device)
        x1, x2 = netx(x_crops)
        x1_sz = x1.shape[-1]
        x2_sz = x2.shape[-1]
        # x_pad = (x1_sz - x2_sz) // 2
        # x2 = nn.functional.pad(x2, (x_pad, x1_sz - x_pad - x2_sz, x_pad, x1_sz - x_pad - x2_sz))
        x2 = torch.nn.functional.upsample(x2, (x1_sz, x1_sz), mode = 'bicubic')
        templates_x = torch.cat([x1, x2], dim = 1)
        scores = corr(templates_x, templates_z)
        scores = torchvision.transforms.functional.resize(scores, (params.final_score_sz, params.final_score_sz), torchvision.transforms.InterpolationMode.BICUBIC)
        scores = torch.squeeze(scores)
        print(torch.amax(scores), torch.amin(scores))
        for j in range(params.scale_num):
            if scale_factor[j] != 1:
                scores[j, :, :] = params.scale_penalty * scores[j, :, :]

        scale_index = torch.argmax(torch.amax(scores, dim = (-1, -2)))
        print(scale_factor[scale_index])
        x_sz = (1 - params.scale_lr) * x_sz + params.scale_lr * x_scale[scale_index]
        width = (1 - params.scale_lr) * width + params.scale_lr * w_scale[scale_index]
        height = (1 - params.scale_lr) * height + params.scale_lr * h_scale[scale_index]

        score = scores[scale_index, :, :]
        # print(score.numpy())
        score = score - torch.min(score)
        score = score/torch.sum(score)
        score = (1 - params.window_influence) * score + params.window_influence * penalty
        center_x, center_y = update_target_position(center_x, center_y, score, x_sz)
        center_x = max(0, center_x)
        center_x = min(center_x, frame_sz[-1])
        center_y = max(0, center_y)
        center_y = min(center_y, frame_sz[-2])
        bboxes[i-1, :] = center_x - width/2, center_y - height/2, width, height
        #z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr * z_scale[scale_index]
        
        if params.z_lr > 0:
            img_z, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, z_sz, avg_chan[0].item())
            crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
            crop_z = torch.unsqueeze(crop_z, dim = 0)
            z1, z2 = netz(crop_z)
            z1_sz = z1.shape[-1]
            z2_sz = z2.shape[-1]
            # z_pad = (z1_sz - z2_sz) // 2
            # z2 = nn.functional.pad(z2, (z_pad, z1_sz - z_pad - z2_sz, z_pad, z1_sz - z_pad - z2_sz))
            z2 = torch.nn.functional.upsample(z2, (z1_sz, z1_sz), mode='bicubic')
            new_template_z = torch.cat([z1, z2], dim = 1).squeeze(dim = 0)
            # new_template_z = netz(crop_z).squeeze()
            new_templates_z = torch.stack([new_template_z] * params.scale_num)

            templates_z = (1 - params.z_lr) * templates_z + params.z_lr * new_templates_z
        # print(center_x, center_y, width, height)
        
        z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr*z_scale[scale_index]   
        # show_frame(img_x, bboxes[i-1,:], i-1)
        print(i)
        save_path = os.path.join(img_dir, 'siamFC\\res', f'{i:08d}.png')
        show_frame(img_x, bboxes[i-1,:], 1, save_path = save_path)
    speed = frame_num / (time.time() - t_start)
    plt.close('all')
    
    return bboxes, speed 


def trackFuV2(img_dir, model, frame_num, pos_x, pos_y, width, height):
    
    netx = model.netx
    netz = model.netz
    corr = model.corr
    # upsample1 = nn.Upsample(size=(13, 13), mode= 'bicubic')
    # upsample2 = nn.Upsample(size=(29, 29), mode= 'bicubic')
    netx.eval()
    netz.eval()
    corr.eval()    
    conv1  = model.conv1
    conv2 = model.conv2
    pool = nn.MaxPool2d(kernel_size=(7, 7), stride = 1)
    center_x, center_y = pos_x + width/2, pos_y + height/2
    bboxes = np.zeros((frame_num, 4))
    bboxes[0, :] = pos_x, pos_y, width, height
    scale_factor = params.scale_step ** np.linspace(-(params.scale_num//2), (params.scale_num//2), params.scale_num)
    context = (width + height) * params.context
    z_sz = np.sqrt((width + context) * (height + context))
    x_sz = float(params.search_sz) / params.examplar_sz * z_sz

    min_z = params.scale_min * z_sz
    max_z = params.scale_max * z_sz
    min_x = params.scale_min * x_sz
    max_z = params.scale_max * x_sz

    hanning_1d = np.expand_dims(np.hanning(params.final_score_sz), axis = 0)
    penalty = np.transpose(hanning_1d) * hanning_1d
    penalty = penalty / np.sum(penalty)

    img_path = os.path.join(img_dir, f'{1:08d}.png')
    img_z = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
    frame_sz = img_z.shape
    avg_chan = torch.mean(img_z, dim = (-1, -2))
    padding_z, x_offset, y_offset = padding_frame(img_z, img_z.shape, center_x, center_y, z_sz, avg_chan[0].item())
    crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
    crop_z = torch.unsqueeze(crop_z, dim = 0)

    z1, z2 = netz(crop_z)
    # z1 = conv1(z1)
    # z2 = upsample1(z2)
    z1 = pool(z1)
    template_z = torch.cat([z1, z2], dim = 1)
    template_z = conv1(template_z).squeeze(dim = 0)
    #crop_z = crop_z.to(device)
    templates_z = torch.stack([template_z] * params.scale_num)

    t_start = time.time()
    #show_frame(img_z, bboxes[0,:], 1, save_path=os.path.join(img_dir, 'siamFC\\res', f'{1:08d}.png'))

    for i in range(1, frame_num):
        z_scale = z_sz * scale_factor
        x_scale = x_sz * scale_factor
        w_scale = width * scale_factor
        h_scale = height * scale_factor
        
        
        img_path = os.path.join(img_dir, f'{i:08d}.png')
        img_x = read_image(img_path, ImageReadMode.RGB).to(dtype = torch.float32)
        avg_chan = torch.mean(img_z, dim = (-1, -2))
        padding_x, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, x_scale[-1], avg_chan[0].item())

        x_crops = torch.stack(crop_frame_x(padding_x, center_x, center_y, x_offset, y_offset, x_scale, params.search_sz), dim=0)
        #x_crops = x_crops.to(device)
        x1, x2 = netx(x_crops)
        x1 = pool(x1)
        # x2 = upsample2(x2)
        # x_pad = (x1_sz - x2_sz) // 2
        # x2 = nn.functional.pad(x2, (x_pad, x1_sz - x_pad - x2_sz, x_pad, x1_sz - x_pad - x2_sz))
        #x2 = torch.nn.functional.upsample(x2, (x1_sz, x1_sz), mode = 'bicubic')
        templates_x = torch.cat([x1, x2], dim = 1)
        templates_x = conv2(templates_x)
        scores = corr(templates_x, templates_z)
        scores = torchvision.transforms.functional.resize(scores, (params.final_score_sz, params.final_score_sz), torchvision.transforms.InterpolationMode.BICUBIC)
        scores = torch.squeeze(scores)

        for j in range(params.scale_num):
            if scale_factor[j] != 1.:
                scores[j, :, :] = params.scale_penalty * scores[j, :, :]

        scale_index = torch.argmax(torch.amax(scores, dim = (-1, -2)))
        print(torch.amax(scores), torch.amin(scores))
        x_sz = (1 - params.scale_lr) * x_sz + params.scale_lr * x_scale[scale_index]
        width = (1 - params.scale_lr) * width + params.scale_lr * w_scale[scale_index]
        height = (1 - params.scale_lr) * height + params.scale_lr * h_scale[scale_index]
        print(scale_factor[scale_index])
        score = scores[scale_index, :, :]
        # print(score.numpy())
        score = score - torch.min(score)
        score = score/torch.sum(score)
        score = (1 - params.window_influence) * score + params.window_influence * penalty
        center_x, center_y = update_target_position(center_x, center_y, score, x_sz)
        center_x = max(0, center_x)
        center_x = min(center_x, frame_sz[-1])
        center_y = max(0, center_y)
        center_y = min(center_y, frame_sz[-2])
        bboxes[i-1, :] = center_x - width/2, center_y - height/2, width, height
        #z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr * z_scale[scale_index]
        
        if params.z_lr > 0:
            img_z, x_offset, y_offset = padding_frame(img_x, img_x.shape, center_x, center_y, z_sz, avg_chan[0].item())
            crop_z = crop_frame_z(img_z, center_x, center_y, x_offset, y_offset, z_sz, params.examplar_sz)
            crop_z = torch.unsqueeze(crop_z, dim = 0)
            z1, z2 = netz(crop_z)
            z1 = pool(z1)
            new_template_z = torch.cat([z1, z2], dim = 1)
            new_template_z = conv1(new_template_z).squeeze(dim = 0)
            #z2 = upsample1(z2)
            new_templates_z = torch.stack([new_template_z] * params.scale_num)
            templates_z = (1 - params.z_lr) * templates_z + params.z_lr * new_templates_z
        # print(center_x, center_y, width, height)
        
        z_sz = (1 - params.scale_lr)*z_sz + params.scale_lr*z_scale[scale_index]   
        # show_frame(img_x, bboxes[i-1,:], i-1)
        print(i)
        save_path = os.path.join(img_dir, 'siamFC\\res', f'{i:08d}.png')
        show_frame(img_x, bboxes[i-1,:], 1, save_path = save_path)
    speed = frame_num / (time.time() - t_start)
    plt.close('all')
    
    return bboxes, speed 
# std = sys.stdout
# outfile = open('out.txt', 'w')
# sys.stdout = outfile
model = layers.SiamFC()
# model = layers.SiamFC()
#model = model.to(device)
# model.load_state_dict(torch.load('SiamFC.pth'))
model.load_state_dict(torch.load('siamfc_birds1.pth'))
model.eval()

with torch.no_grad():
    cnt = len(params.track_videos)
    lengths = np.zeros(cnt)
    percisions = np.zeros(cnt)
    percision_aucs = np.zeros(cnt)
    ious = np.zeros(cnt)
    speeds = np.zeros(cnt)
    
    for i in range(cnt):
        img_dir = os.path.join(params.dataset_dir, params.track_videos[i])
        gt_path = os.path.join(img_dir, params.gt_file)
        gts = get_gt(gt_path)
        frame_num = len(gts)
        pos_x, pos_y, width, height = gts[0]
        bboxes, speeds[i] = track(img_dir, model, frame_num, pos_x, pos_y, width, height)
        lengths[i], percisions[i], percision_aucs[i], ious[i] = complile_results(gts, bboxes, params.dist_threshold)

        print('Track Video: %s-----------------'%params.track_videos[i])
        print(' -- Percision: %.2f'%percisions[i])
        print(' -- Percision AUC: %.2f'%percision_aucs[i])
        print(' -- IOU: %.2f'%ious[i])
        print(' -- Speed: %.2f'%speeds[i])
        print('')
    
    total_frames = np.sum(lengths)
    mean_percision = np.sum(percisions * lengths) / total_frames
    mean_percision_auc = np.sum(percision_aucs * lengths) / total_frames
    mean_speed = np.sum(speeds * lengths) / total_frames
    mean_iou = np.sum(ious * lengths) / total_frames

    print('-- Overall stats for %d frames --'%lengths)
    print(' -- Percision: %.2f'%mean_percision)
    print(' -- Percision AUC: %.2f'%mean_percision_auc)
    print(' -- IOU: %.2f'%mean_iou)
    print(' -- Speed: %.2f'%mean_speed)
    print('')

# outfile.close()
# sys.stdout = std
