import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn as nn

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

# def manual_lr_schedule(epoch, start_epoch=400, total_epochs=80, start_lr=1e-5, min_lr=1e-6):
#     effective_epoch = epoch - start_epoch
#     if effective_epoch < 0 or effective_epoch >= total_epochs:
#         raise ValueError(f"Epoch {epoch} out of fine-tuning range ({start_epoch} ~ {start_epoch + total_epochs - 1})")

#     # Cosine decay 계산
#     cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_epoch / (total_epochs - 1)))
#     current_lr = min_lr + (start_lr - min_lr) * cosine_decay
#     return current_lr

def manual_lr_schedule(epoch):
    start_epoch = 0
    total_epochs = 40
    start_lr = 2e-4  # 0.0002 -> 0.0005
    min_lr = 1e-5   # 0.00001

    effective_epoch = epoch - start_epoch
    if effective_epoch < 0 or effective_epoch >= total_epochs:
        raise ValueError(f"Epoch {epoch} out of range ({start_epoch} ~ {start_epoch + total_epochs - 1})")

    cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_epoch / (total_epochs - 1)))
    current_lr = min_lr + (start_lr - min_lr) * cosine_decay
    return current_lr


def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def sample_configs_curriculum(choices, epoch=None, curriculum_epoch=None):
    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    # depth = 12

    if epoch is None:
        config['embed_dim'] = [192] * depth
        config['mlp_ratio'] = [3.5] * depth
        config['num_heads'] = [3] * depth

    elif epoch < curriculum_epoch[1]:
        config['embed_dim'] = [192] * depth
        config['mlp_ratio'] = [3.5] * depth
        config['num_heads'] = [3] * depth
        # config['embed_dim'] = [192] * depth
        # config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        # config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    elif epoch >= curriculum_epoch[1] and epoch < curriculum_epoch[2]:
        # config['embed_dim'] = [216] * depth
        # embed_dim_choice = random.choice([216, 240])
        embed_dim_choice = random.choices([216, 240], weights=[1, 1])[0] # (1, 3) -> (1, 1)
        config['embed_dim'] = [embed_dim_choice] * depth
        config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    else:
        config['embed_dim'] = [240] * depth
        config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, args=None):
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


    
    curriculum_epoch = [-1, 0, 20]
    # curriculum_epoch = [-1, 0, 41]
    # case_num = None
    case_num = 2 # 이거 괜찮나?

    # if epoch in curriculum_epoch:
    #     case_num = curriculum_epoch.index(epoch) + 1
    for i in range(len(curriculum_epoch)):
        if epoch >= curriculum_epoch[i]:
            case_num = i + 1

        # config = sample_configs_curriculum(choices=choices, epoch=epoch, curriculum_epoch=curriculum_epoch) 
        # model_module = unwrap_model(model)
        # model_module.set_sample_config(config=config, case_num=case_num)

        # # (2) 큰 learning rate를 적용할 그룹과 작은 learning rate를 적용할 그룹 분리
        # large_lr_params = []
        # small_lr_params = []

        # for name, param in model.named_parameters():
        #     # 💡 self.weight, self.bias는 제외
        #     if name.endswith("weight") or name.endswith("bias"):
        #         continue  # 완전 제외, requires_grad 유지

        #     if param.requires_grad:
        #         large_lr_params.append(param)  # 원래 학습 중인 파라미터
        #     else:
        #         param.requires_grad = True    # soft freeze 대상만 학습 가능하게 바꾸고
        #         small_lr_params.append(param) # 작은 lr로 학습

        # optimizer = torch.optim.AdamW([
        #     {'params': large_lr_params, 'lr': args.lr, 'weight_decay': args.weight_decay},           # 일반 학습
        #     {'params': small_lr_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay * 0.1} # soft freeze
        # ])


    print("case_num : ", case_num)

    # # 2. requires_grad=True인 파라미터들만 모아서 확인
    # trainable_params = []
    # print("[Trainable Parameters]")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"  ✅ {name}, shape: {param.shape}")
    #         trainable_params.append(param)

    # print(f"\nTotal trainable param groups: {len(trainable_params)}")

    # if case_num is None:
    #     # 3. 해당 param만 optimize 되도록 optimizer 생성
    #     optimizer = create_optimizer(args, trainable_params)
    # else:
    #     # 최종 optimizer 먼저 생성
    #     optimizer = create_optimizer(args, [p for p in model.parameters() if p.requires_grad])


    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            # config = sample_configs(choices=choices)
            config = sample_configs_curriculum(choices=choices, epoch=epoch, curriculum_epoch=curriculum_epoch) 
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config, case_num=case_num)

            # if case_num is not None:
            #     case_num = None # None이 아닐때마다 gaussian init해주는거라 그거 방지.

            # for name, param in model.named_parameters():
            #     param.requires_grad = True

            # current_lr = manual_lr_schedule(epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = current_lr

            # current_lr = manual_lr_schedule(epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = current_lr

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
                    # (원래) loss = criterion(outputs, targets)  # 여기에 이어서 추가

                    # # regularization 추가
                    # reg_loss = 0.0

                    # for name, module in model.named_modules():
                    #     for attr in ['split_weights', 'split_bias', 'split_biases', 'split_embeddings_v', 'split_embeddings_h']:
                    #         if hasattr(module, attr):
                    #             param_dict = getattr(module, attr)
                    #             if isinstance(param_dict, nn.ParameterDict) and len(param_dict) > 0:
                                    
                    #                 # 기준(A 영역)은 무조건 첫 번째 키
                    #                 first_key = next(iter(param_dict))
                    #                 ref_tensor = param_dict[first_key].detach()
                    #                 mean_ref = ref_tensor.mean()
                    #                 var_ref = ref_tensor.var(unbiased=False)

                    #                 # 이제 requires_grad=True인 것들(B 영역)만 골라서 reg 걸기
                    #                 for key, param in param_dict.items():
                    #                     if param.requires_grad:
                    #                         param_data = param.detach().view(-1)
                    #                         mean_param = param_data.mean()
                    #                         var_param = param_data.var(unbiased=False)

                    #                         # (mean 차이)^2 + (var 차이)^2
                    #                         reg_loss += (mean_param - mean_ref).pow(2) + (var_param - var_ref).pow(2)

                    # # 마지막에 loss에 추가
                    # loss = loss + 0.001 * reg_loss

        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)
                
                # # regularization 추가
                # reg_loss = 0.0

                # for name, module in model.named_modules():
                #     for attr in ['split_weights', 'split_bias', 'split_biases', 'split_embeddings_v', 'split_embeddings_h']:
                #         if hasattr(module, attr):
                #             param_dict = getattr(module, attr)
                #             if isinstance(param_dict, nn.ParameterDict) and len(param_dict) > 0:
                                
                #                 # 기준(A 영역)은 무조건 첫 번째 키
                #                 first_key = next(iter(param_dict))
                #                 ref_tensor = param_dict[first_key].detach()
                #                 mean_ref = ref_tensor.mean()
                #                 var_ref = ref_tensor.var(unbiased=False)

                #                 # 이제 requires_grad=True인 것들(B 영역)만 골라서 reg 걸기
                #                 for key, param in param_dict.items():
                #                     if param.requires_grad:
                #                         param_data = param.detach().view(-1)
                #                         mean_param = param_data.mean()
                #                         var_param = param_data.var(unbiased=False)

                #                         # (mean 차이)^2 + (var 차이)^2
                #                         reg_loss += (mean_param - mean_ref).pow(2) + (var_param - var_ref).pow(2)

                # # 마지막에 loss에 추가
                # loss = loss + 0.005 * reg_loss

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

        # 🔸 마지막 배치에서만 평균 gradient norm 출력
        if i == len(data_loader) - 1:
            total_norm = 0.0
            count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    count += 1
            total_norm = total_norm ** 0.5
            avg_grad = total_norm / (count if count > 0 else 1)
            print(f"[Epoch {epoch}] Avg Grad Norm: {avg_grad:.6f}")

            # # 🔸 blocks.1, blocks.11, blocks.12 에 대해 lambda_log 출력
            # target_blocks = {'blocks.1', 'blocks.11', 'blocks.12'}

            # for name, module in model.module.named_modules():
            #     if any(f'blocks.{idx}' in name for idx in [1, 11, 12]):
            #         # if isinstance(module, LinearSuper):
            #         if hasattr(module, 'lambda_log') and isinstance(module.lambda_log, dict):
            #             if len(module.lambda_log) > 0:
            #                 lambda_str = ', '.join([f"{k}: {v:.4f}" for k, v in module.lambda_log.items()])
            #                 print(f"[{name}] λ: {lambda_str}")


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
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # curriculum_epoch = [-1, 0, 21]
    curriculum_epoch = [-1, 0, 41]

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        # config = sample_configs(choices=choices)
        # config = sample_configs_curriculum(choices=choices, epoch=epoch, curriculum_epoch=curriculum_epoch) 
        # config = {'layer_num': 13, 'mlp_ratio': [4.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 3.5, 4.0, 3.5, 3.5], 'num_heads': [4, 3, 4, 3, 3, 4, 4, 3, 4, 4, 4, 3, 4], 'embed_dim': [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216]}
        # config = {'layer_num': 14, 'mlp_ratio': [4.0, 4.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 3.5, 3.5, 3.5, 3.5], 'num_heads': [3, 3, 4, 4, 3, 3, 4, 4, 4, 4, 4, 3, 4, 3], 'embed_dim': [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]}
        if epoch % 2 == 1:
            config = {
                'layer_num': 13,
                'mlp_ratio': [4.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.5, 3.5, 4.0, 3.5, 3.5],
                'num_heads': [4, 3, 4, 3, 3, 4, 4, 3, 4, 4, 4, 3, 4],
                'embed_dim': [216] * 13
            }
        else:
            config = {
                'layer_num': 14,
                'mlp_ratio': [4.0, 4.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 3.5, 3.5, 3.5, 3.5],
                'num_heads': [3, 3, 4, 4, 3, 3, 4, 4, 4, 4, 4, 3, 4, 3],
                'embed_dim': [240] * 14
            }

        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

        # for name, param in model.named_parameters():
        #     # 💡 self.weight, self.bias는 제외
        #     if name.endswith("weight") or name.endswith("bias"):
        #         continue  # 완전 제외, requires_grad 유지
        #     else:
        #         param.requires_grad = True

    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}