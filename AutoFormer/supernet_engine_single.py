import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
import copy
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2
from skimage.transform import resize
from torchvision.transforms import ToPILImage
from PIL import Image
from timm.models import create_model
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

cmap = matplotlib.cm.get_cmap('jet')
cmap.set_bad(color="k", alpha=0.0)

def tensor_to_excel(tensor, file_name='cls_attn_map.xlsx'):
    # 텐서를 NumPy 배열로 변환
    # print('tensor shape : ', tensor.shape)
    tensor = tensor.cpu()
    array = tensor.numpy()
    
    # NumPy 배열을 Pandas DataFrame으로 변환
    df = pd.DataFrame(array)
    
    # DataFrame을 Excel 파일로 저장
    df.to_excel(file_name, index=False)

def to_tensor(img):
    transform_fn = Compose([Resize(249, 3), CenterCrop(224), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_fn(img)

def show_img(img, filename):
    # 입력 데이터가 torch.Tensor 인지 확인
    if isinstance(img, torch.Tensor):
        # 텐서가 GPU에 있다면 CPU로 이동
        if img.is_cuda:
            img = img.cpu()
        # 텐서를 NumPy 배열로 변환
        img = img.numpy()
    
    # 입력 데이터가 PIL.Image 객체인지 확인
    elif isinstance(img, Image.Image):
        # PIL.Image 객체를 NumPy 배열로 변환
        img = np.array(img)
        
    # img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    # 이미지 채널에 따라 적절한 색상 맵 설정
    cmap = 'gray' if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1) else None
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_img2(img1, img2, alpha=0.8, filename='overlay.png'):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    # 플롯 설정
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)  # 첫 번째 이미지 표시
    plt.imshow(img2, alpha=alpha)  # 두 번째 이미지를 투명도와 함께 표시
    plt.axis('off')  # 축 레이블 및 틱 제거

    # 이미지 파일로 저장
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # 파일로 저장
    plt.close()  # 현재 플롯 닫기
    
def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward


def attention_grid_interpolation(im, att):
    print('att shape : ', att.shape)
    origin_img = im.cpu().numpy().astype(np.uint8)
    origin_img = np.transpose(origin_img, (0, 2, 3, 1))[0]

    # 이미지를 numpy 배열로 변환 (혹시 이미 변환되어 있으면 이 부분은 생략 가능)
    if isinstance(im, torch.Tensor):
        im = im.cpu().numpy()
    if len(im.shape) == 4 and im.shape[1] == 3:  # RGB 이미지인지 확인
        im = np.transpose(im, (0, 2, 3, 1))  # (1, 3, 224, 224) -> (1, 224, 224, 3)

    # im 배열의 shape 확인
    print('im shape : ', im.shape)

    # att shape: (14, 1, 4, 197, 197)
    # 레이어별로 각 head에 대해 처리
    # for layer in range(att.shape[0]):
    #     for head in range(att.shape[2]):
    for layer in range(1):
        for head in range(1):
            # CUDA 텐서를 CPU로 이동하고 numpy 배열로 변환
            att_layer_head = att[layer, 0, head].cpu().numpy()

            # Softmax 맵 크기 변경
            opacity = resize(att_layer_head, im.shape[1:3], order=3)
            opacity = opacity * 0.95 + 0.05

            # opacity의 크기를 im의 크기와 맞추기
            opacity = np.expand_dims(opacity, axis=-1)  # (224, 224) -> (224, 224, 1)
            opacity = np.tile(opacity, (1, 1, 3))  # (224, 224, 1) -> (224, 224, 3)

            # 시각화를 위해 원본 이미지와 혼합
            im[0] = opacity * im[0] + (1 - opacity) * 255

    vis_im = im[0].astype(np.uint8)
    # print('vis_im shape : ', vis_im.shape)
    # vis_im = origin_img
    return vis_im

def visualize_pred(im, att_weights): # im_path, boxes, att_weights
    # im = cv2.imread(im_path)
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    # b,g,r,a = cv2.split(im)           # get b, g, r
    # im = cv2.merge([r,g,b,a])
    
    im_grid_att = attention_grid_interpolation(im, att_weights)

    # M = min(len(boxes), len(att_weights))
    # im_ocr_att = attention_bbox_interpolation(im, boxes[:M], att_weights[:M])
    # plt.imshow(im_ocr_att)
    plt.imshow(im_grid_att)
    
    # 이미지 저장
    plt.axis('off')  # 축 제거
    plt.savefig('im_grid_att.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
# 이미지를 PNG 파일로 저장하는 함수
def save_tensor_as_png(tensor, file_name):
    # 텐서를 CPU로 이동하고 numpy 배열로 변환
    tensor = tensor.cpu().detach()

    # 텐서의 shape가 (1, 3, 224, 224)인 경우 (3, 224, 224)로 변경
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # 텐서의 값 범위를 [0, 1]로 조정 (이미지가 0에서 1 사이의 값인지 확인)
    tensor = torch.clamp(tensor, 0, 1)

    # 텐서를 PIL 이미지로 변환
    transform = ToPILImage()
    image = transform(tensor)
    
    # 이미지를 PNG 파일로 저장
    image.save(file_name)

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, image, amp=True, choices=None, net=None, mode='super', retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    ### input 이미지 저장하려면 여기 START ###
    # print('image shape : ', image.shape)
    
    # image = image.to(device, non_blocking=True)
    # image_test = (image - image.min()) / (image.max() - image.min())  # 값 범위를 [0, 1]로 조정
    # save_tensor_as_png(image_test, 'input_image.png')
    ### input 이미지 저장하려면 여기 END ###
    
    # image = image_test

    ###### START #########
    # compute output
    # if amp:
    #     with torch.cuda.amp.autocast():
    #         output = model(image)
    # else:
    #     output = model(image)
    
    # total_attn_maps = []
    # for i, block in enumerate(net.blocks):
    #     attn_maps = block.attn.get_attention_maps()
    #     # print('attn_maps shape : ', attn_maps.shape)
    #     total_attn_maps.append(attn_maps)
        
    # combined_attention_maps = torch.stack(total_attn_maps, dim=0)
    
    # print('combined_attention_maps shape : ', combined_attention_maps.shape)

    # print('output shape: ', output.shape)
    
    # visualize_pred(image, combined_attention_maps)
    
    ########## END ############

    # Now, attention_maps list should be populated with attention weights
    # visualize_attention_maps(image.squeeze(0), attention_maps)
    
    ### START : visualize attention map on input image ###
    img = Image.open('input_image1.png')
    x = to_tensor(img)
    
    # net.blocks[-1].attn.forward = my_forward_wrapper(net.blocks[-1].attn)
    for block in net.blocks:
        block.attn.forward = my_forward_wrapper(block.attn)
    
    net.to(device)
    x = x.to(device)    
    
    y = net(x.unsqueeze(0))
    # attn_map = net.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()
    # cls_weight = net.blocks[-1].attn.cls_attn_map.mean(dim=1).view(14, 14).detach()
    # 평균 어텐션 맵 계산
    attn_map = torch.stack([block.attn.attn_map.mean(dim=1).squeeze(0).detach() for block in net.blocks if hasattr(block.attn, 'attn_map')]).mean(dim=0) if any(hasattr(block.attn, 'attn_map') for block in net.blocks) else None

    # 평균 클래스 어텐션 맵 계산
    cls_weight = torch.stack([block.attn.cls_attn_map.mean(dim=1).view(14, 14).detach() for block in net.blocks if hasattr(block.attn, 'cls_attn_map')]).mean(dim=0) if any(hasattr(block.attn, 'cls_attn_map') for block in net.blocks) else None
    
    # tensor_to_excel(cls_weight)

    img_resized = x.permute(1, 2, 0) * 0.5 + 0.5
    cls_resized = F.interpolate(cls_weight.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1)

    show_img(img, 'result/img.png')
    show_img(attn_map, 'result/attn_map.png')
    show_img(cls_weight, 'result/cls_weight.png')
    show_img(img_resized, 'result/img_resized.png')
    show_img2(img_resized, cls_resized, alpha=0.8, filename='result/img_resized_overlay.png')
    
    ### END : visualize attention map on input image ###
    
    # return output
    return attn_map
    # return "tmp"

    # for images, target in metric_logger.log_every(data_loader, 10, header):
    #     images = images.to(device, non_blocking=True)
    #     target = target.to(device, non_blocking=True)
    #     # compute output
    #     if amp:
    #         with torch.cuda.amp.autocast():
    #             output = model(images)
    #             loss = criterion(output, target)
    #     else:
    #         output = model(images)
    #         loss = criterion(output, target)

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     batch_size = images.shape[0]
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
