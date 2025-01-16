import os
import argparse
import datetime
import numpy as np
import time
import torch
import sys

import torch.backends.cudnn as cudnn
import json
import warnings
from pathlib import Path
import wandb
import torch
import numpy as np


import sys
from arguments import *
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import torchvision

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import *
from loss import *
from samplers import RASampler
from timm.models import create_model


import deit_models
import teacher_models
import utils
from utils import *
from transmix import Mixup_transmix
import time

import moco.loader
import moco.builder
from teacher_models import resnet_cifar_paco




def main(args):
    if not args.eval:
        args.no_distillation = False   #如果未设置评估模式，则默认不使用蒸馏
    if "distilled" not in args.model:  #如果模型名称中不包含"distilled"，则设置
        args.no_distillation = True
        print("\nNO DISTILLATION\n")
        time.sleep(2)

    name = args.name_exp #获取实验名称

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except:
        local_rank = 0
    #日志记录
    if args.log_results and local_rank == 0:
        wandb.init(project=args.project_name, name=name)
        wandb.run.log_code(".")
        wandb.config.update(args)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.gpu)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Code modified for DeiT-LT，加载数据集
    if args.data_set == "INAT18":
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(
            is_train=False, args=args, class_map=dataset_train.class_map
        )
        args.class_map = dataset_train.class_map

    elif args.data_set == "IMAGENETLT":
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(
            is_train=False, args=args, class_map=dataset_train.class_map
        )
        args.class_map = dataset_train.class_map

    else:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)

    # cls_num_list = dataset_train.get_cls_num_list()
    # args.cls_num_list = cls_num_list
    # # 对于不平衡的数据集，计算每个类的权重per_cls_weights
    # beta = args.beta
    # effective_num = 1.0 - np.power(beta, cls_num_list)
    # per_cls_weights = (1.0 - beta) / np.array(effective_num)
    # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)


    args.categories = []
    if args.data_set == "CIFAR10LT":
        args.categories = [3, 7]
    if args.data_set == "CIFAR100LT":
        args.categories = [36, 71]
    if args.data_set == "IMAGENETLT":
        args.categories = [390, 835]
    if args.data_set == "INAT18":
        args.categories = [842, 4543]

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            print("Repeated Aug ****************")
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            print("[INFORMATION] No Repeated Aug, But with distributed")
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=args.drop_last,
    )
    sampler_val = None if args.data_set not in ['IMAGENETLT', 'INAT18'] else sampler_val
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    #设置混合策略Mixup/CutMix，初始化相应的混合函数。
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None

    if mixup_active:
        if not args.transmix:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )
        else:
            print("USING TRANSMIX\n")
            mixup_fn = Mixup_transmix(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )

    print("[INFORMATION] THe model being used is ", args.model)

    print("[INFORMATION] Model loaded from custom file")
    model = deit_models.__dict__[args.model](
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        mask_attn=args.mask_attn,
        early_stopping=args.early_stopping,
    )
    model.to(device)
    print("Model", model)

    #初始化EMA模型变量为None
    model_ema = None
    if args.model_ema:#如果args.model_ema为真，则创建EMA模型实例，并设置其衰减率、设备（CPU或GPU）等参数。
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
    #如果args.distributed为真，则使用...DistributedDataParallel包装模型，以支持分布式训练
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module #存储分布式包装之前的模型
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters) #计算并打印模型中的可训练参数数量
    #学习率调整
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)

    #根据args.accum_iter的值（梯度累积的迭代次数），决定使用哪种损失缩放器。
    if args.accum_iter > 1:
        loss_scaler = NativeScalerWithGradNormCount()
    else:
        loss_scaler = NativeScaler()

    print("WARMUP EPOCHS = ", args.warmup_epochs)

    # if args.accum_iter == 1:

    lr_scheduler, _ = create_scheduler(args, optimizer)
    #损失函数配置
    if mixup_active:
        print("Critera: SoftTargetCrossEntropy")
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        print("Criteria: BCE Loss")
        criterion = torch.nn.BCEWithLogitsLoss()
    #教师模型配置
    teacher_model = None
    if args.distillation_type != "none":
        print("[INFORMATION] The distillation token im Deit is being used")
        assert args.teacher_path, "Need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        if "regnet" in args.teacher_model:
            teacher_model = create_model(
                args.teacher_model,
                pretrained=False,
                num_classes=args.nb_classes,
                global_pool="avg",
            )
        else:
                #如果未指定args.paco，则直接从teacher_models字典中加载模型。
            if not args.paco:
                teacher_model = teacher_models.__dict__[args.teacher_model](
                    num_classes=args.nb_classes, use_norm=args.use_norm
                )
            else:  #如果指定了，则使用..MoCo构建器创建一个基于MoCo（一种自监督学习方法）的模型。
                if 'CIFAR' in args.data_set:
                    teacher_model = moco.builder.MoCo(
                        getattr(resnet_cifar_paco, args.teacher_model),
                        args.moco_dim,
                        args.moco_k,
                        args.moco_m,
                        args.moco_t,
                        args.mlp,
                        args.feat_dim,
                        args.normalize,
                        num_classes=args.nb_classes,
                    )
                elif args.data_set in ['IMAGENETLT', 'INAT18']:
                    print('Loaded Imagenetlt teacher')
                    print('Moco dim: ', args.moco_dim)
                    print('Moco K: ', args.moco_k)
                    print('Moco m: ', args.moco_m)
                    print('Moco t: ', args.moco_t)
                    print('MLP: ', args.mlp)
                    print('Feat Dim: ', args.feat_dim)
                    print('Normalize: ', args.normalize)
                    print('Classes: ', args.nb_classes)
                    print('\n\n')
                    teacher_model = moco.builder.MoCo(
                        getattr(resnet_imagenet_paco, args.teacher_model),
                        args.moco_dim,
                        args.moco_k,
                        args.moco_m,
                        args.moco_t,
                        args.mlp,
                        args.feat_dim,
                        args.normalize,
                        num_classes=args.nb_classes,
                    )
                    teacher_model.to(device)

        #加载权重
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True
            )
        else:
            print("[INFORMATION] Loading teacher model from path ", args.teacher_path)
            checkpoint = torch.load(args.teacher_path, map_location="cpu")


        if not args.paco:
            teacher_model.load_state_dict(
                checkpoint["model"]
                if "model" in checkpoint.keys()
                else checkpoint["state_dict"]
            )
        else:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[7:]] = v
            teacher_model.load_state_dict(state_dict, strict=True)

        teacher_model.to(device)
        teacher_model.eval()

    else:
        print("[INFORMATION] Teacher model is None.")

    ##############################################################################加进去的############################################################################
    model.eval()  # 切换到评估模式
    cls_num_list = dataset_train.get_cls_num_list()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    cls_num = len(cls_num_list)
    feature_mean_end = torch.zeros(cls_num, 768)
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()

            input = input.cuda()
            # 设置 requires_grad 属性
            input_var = input.requires_grad_(requires_grad=False)
            features, output, _, _ = model(input_var)
            features = features.detach()
            features = features.cpu().data.numpy()

            for out, label in zip(features, target):
                feature_mean_end[label] = feature_mean_end[label] + out

            img_num_list_tensor = torch.tensor(cls_num_list).unsqueeze(1)
            feature_mean_end = torch.div(feature_mean_end, img_num_list_tensor).detach()
        return feature_mean_end

    train_propertype = feature_mean_end.cuda()
    train_propertype = train_propertype.cuda()
    class_num = len(cls_num_list)
    eff_all = torch.zeros(class_num).float().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            if torch.cuda.is_available():
                input = input.cuda()
            # 设置 requires_grad 属性
            input_var = input.requires_grad_(requires_grad=False)
            features, output = model(input_var)
            mu = train_propertype[target].detach()  # batch_size x d
            feature_bz = (features.detach() - mu)  # Centralization
            index = torch.unique(target)  # class subset
            index2 = target.cpu().numpy()
            eff = torch.zeros(class_num).float().cuda()
            # 对每个类别的数据，计算特征的协方差矩阵和特征中心化后的内积
            for i in range(len(index)):  # number of class
                index3 = torch.from_numpy(np.argwhere(index2 == index[i].item()))
                index3 = torch.squeeze(index3)
                feature_juzhen = feature_bz[index3].detach()
                if feature_juzhen.dim() == 1:
                    eff[index[i]] = 1
                else:
                    _matrixA_matrixB = torch.matmul(feature_juzhen, feature_juzhen.transpose(0, 1))
                    _matrixA_norm = torch.unsqueeze(
                        torch.sqrt(torch.mul(feature_juzhen, feature_juzhen).sum(axis=1)),
                        1)
                    _matrixA_matrixB_length = torch.mul(_matrixA_norm, _matrixA_norm.transpose(0, 1))
                    _matrixA_matrixB_length[_matrixA_matrixB_length == 0] = 1
                    r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length)  # R
                    num = feature_juzhen.size(0)
                    a = (torch.ones(1, num).float().cuda()) / num  # a_T
                    b = (torch.ones(num, 1).float().cuda()) / num  # a
                    c = torch.matmul(torch.matmul(a, r), b).float().cuda()  # a_T R a
                    eff[index[i]] = 1 / c
            eff_all = eff_all + eff
        weights = eff_all
        weights = torch.where(weights > 0, 1 / weights, weights).detach()
        # weight计算每个类别的有效权重
        fen_mu = torch.sum(weights)
        weights_new = weights / fen_mu
        weights_new = weights_new * class_num  # Eq.(14)
        print(weights_new)
        weights_new = weights_new.detach()
        return weights_new
    # per_cls_weights = torch.FloatTensor(weights_new).cuda(args.gpu)
    per_cls_weights = weights_new.detach().to(f'cuda:{args.gpu}')

###################################################################3结束点############################################################################

    #蒸馏损失函数配置
    if not args.no_distillation:
        print("Criteria: Distillation Loss")
        if args.drw == None:
            weighted_distillation = (args.weighted_distillation,)
        else:
            weighted_distillation = False

        # Modified distillation loss for multi-crop
        if args.multi_crop:
            print("Multi-crop Distillation")
            distillation_loss = DistillationLossMultiCrop
        else:
            print("Normal Distillation")
            distillation_loss = DistillationLoss

        criterion = distillation_loss(
            criterion,
            teacher_model,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
            args.input_size,
            args.teacher_size,
            weighted_distillation,
            per_cls_weights,
            args,
        )

    output_dir = Path(args.output_dir)
    if args.resume: #恢复训练
        print("RESUMING FROM CHECKPOINT")
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
            print("CHECKPOINT LOADED")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        lr_scheduler.step(args.start_epoch)
        #如果启用了早停（args.early_stopping为真），则进行相应的设置
        if args.early_stopping:
            print("Early Stopping Stage")
            model_without_ddp.early_stopping_setup()
            model.module.head_dist.weight.requires_grad = False
            model.module.head_dist.bias.requires_grad = False
            args.distillation_alpha = 0

    #模型评估
    if args.eval:
        print("EVALUATION OF MODEL")
        # args.no_distillation = True
        test_stats = evaluate(data_loader_val, model, device, args)
        return

    start_time = time.time()
    max_accuracy_avg = 0.0
    max_head_avg = 0.0
    max_med_avg = 0.0
    max_tail_avg = 0.0
    max_accuracy_cls = 0.0
    max_head_cls = 0.0
    max_med_cls = 0.0
    max_tail_cls = 0.0
    max_accuracy_dist = 0.0
    max_head_dist = 0.0
    max_med_dist = 0.0
    max_tail_dist = 0.0
    start_epoch = args.start_epoch
    epochs = args.epochs
    print(f"Start training for {epochs} epochs")
  #初始化训练周期
    for epoch in range(start_epoch, epochs):

        # Code modified for DeiT-LT
        if args.distributed: #分布式训练设置
            data_loader_train.sampler.set_epoch(epoch)
        #类别重加权
        if (
            args.drw is not None and epoch >= args.drw
        ):  # Do reweighting after specified number of epochs
            #根据是否使用加权基础损失.weighted_baseloss和是否使用二分类损失（.bce_loss）来设置基础损失函数。
            if not args.no_distillation:
                if args.weighted_baseloss:
                    print("USING Reweighted CE Class Loss in DRW")
                    #根据每个类别的权重per_cls_weights调整损失函数
                    if args.bce_loss:
                        base_criterion = torch.nn.BCEWithLogitsLoss(
                            weight=per_cls_weights
                        )
                    else:
                        if args.no_mixup_drw:
                            base_criterion = torch.nn.CrossEntropyLoss(
                                weight=per_cls_weights
                            )
                        else:
                            base_criterion = SoftTargetCrossEntropy()
                    #基础损失函数与蒸馏损失函数结合，形成最终的损失函数
                    criterion = DistillationLoss(
                        base_criterion,
                        teacher_model,
                        args.distillation_type,
                        args.distillation_alpha,
                        args.distillation_tau,
                        args.input_size,
                        args.teacher_size,
                        args.weighted_distillation,
                        per_cls_weights,
                        args,
                    )
                else:
                    print("USING CE Class Loss in DRW")
                #如果没有启用知识蒸馏，则直接使用加权交叉熵损失
                    if args.bce_loss:
                        base_criterion = torch.nn.BCEWithLogitsLoss()
                    else:
                        if args.no_mixup_drw:
                            base_criterion = torch.nn.CrossEntropyLoss()
                        else:
                            base_criterion = SoftTargetCrossEntropy()

                    criterion = DistillationLoss(
                        base_criterion,
                        teacher_model,
                        args.distillation_type,
                        args.distillation_alpha,
                        args.distillation_tau,
                        args.input_size,
                        args.teacher_size,
                        args.weighted_distillation,
                        per_cls_weights,
                        args,
                    )
            else:
                print("Using CE Loss in DRW")
                criterion = torch.nn.CrossEntropyLoss(weight=per_cls_weights)

        print("The distillation type is ", str(args.distillation_type))
        #这个函数负责处理模型的前向传播、损失计算、反向传播、优化器更新等步骤。
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            teacher_model=teacher_model,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            data_loader=data_loader_train,
            set_training_mode=args.finetune
            == "",  # keep in eval mode during finetuning
            lr_scheduler=lr_scheduler,
            args=args,
        )

        #日志记录
        print("Loss        = ", train_stats["loss"])
        print("Cls Loss    = ", train_stats["cls_loss"])
        print("Dst Loss    = ", train_stats["dst_loss"])
        #学习率调度
        if args.accum_iter == 1:
            lr_scheduler.step(epoch)
        #模型评估
        test_stats = evaluate(data_loader_val, model, device, args)
        checkpoint_paths = []
        if args.output_dir:
            if args.drw != None and epoch == args.drw - 1:
                checkpoint_paths.append(
                    output_dir / (name + f"_epoch_{str(epoch)}_DRW_checkpoint.pth")
                )

            if (epoch + 1) % args.save_freq == 0 or epoch == 774:
                checkpoint_paths.append(
                    output_dir / (name + f"_epoch_{str(epoch)}_checkpoint.pth")
                )
            #保存检查点
            checkpoint_paths.append(output_dir / (name + "_checkpoint.pth"))
            print(checkpoint_paths)
            for checkpoint_path in checkpoint_paths:
                print("Saving at ", checkpoint_path)
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                        "head_acc_avg": test_stats["head_acc_avg"],
                        "med_acc_avg": test_stats["med_acc_avg"],
                        "tail_acc_avg": test_stats["tail_acc_avg"],
                        "head_acc_cls": test_stats["head_acc_cls"],
                        "med_acc_cls": test_stats["med_acc_cls"],
                        "tail_acc_cls": test_stats["tail_acc_cls"],
                        "head_acc_dist": test_stats["head_acc_dist"],
                        "med_acc_dist": test_stats["med_acc_dist"],
                        "tail_acc_dist": test_stats["tail_acc_dist"],
                    },
                    checkpoint_path,
                )
        #更新最佳性能
        if max_accuracy_avg < test_stats["acc1_avg"]:
            max_accuracy_avg = test_stats["acc1_avg"]
            max_head_avg = test_stats["head_acc_avg"]
            max_med_avg = test_stats["med_acc_avg"]
            max_tail_avg = test_stats["tail_acc_avg"]

            max_accuracy_cls = test_stats["acc1_cls"]
            max_head_cls = test_stats["head_acc_cls"]
            max_med_cls = test_stats["med_acc_cls"]
            max_tail_cls = test_stats["tail_acc_cls"]

            max_accuracy_dist = test_stats["acc1_dist"]
            max_head_dist = test_stats["head_acc_dist"]
            max_med_dist = test_stats["med_acc_dist"]
            max_tail_dist = test_stats["tail_acc_dist"]

            if args.output_dir:
                checkpoint_paths = [output_dir / (name + "_best_checkpoint.pth")]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "model_ema": get_state_dict(model_ema),
                            "scaler": loss_scaler.state_dict(),
                            "args": args,
                            "best_acc_avg": max_accuracy_avg,
                            "head_acc_avg": max_head_avg,
                            "med_acc_avg": max_med_avg,
                            "tail_acc_avg": max_tail_avg,
                            "best_acc_cls": max_accuracy_cls,
                            "head_acc_cls": max_head_cls,
                            "med_acc_cls": max_med_cls,
                            "tail_acc_cls": max_tail_cls,
                            "best_acc_dist": max_accuracy_dist,
                            "head_acc_dist": max_head_dist,
                            "med_acc_dist": max_med_dist,
                            "tail_acc_dist": max_tail_dist,
                        },
                        checkpoint_path,
                    )

        print("\nBEST NUMBERS ----->")
        print("Overall / Head / Med / Tail")
        print(
            "AVERAGE: ",
            round(max_accuracy_avg, 3),
            " / ",
            round(max_head_avg, 3),
            " / ",
            round(max_med_avg, 3),
            " / ",
            round(max_tail_avg, 3),
        )
        print(
            "CLS    : ",
            round(max_accuracy_cls, 3),
            " / ",
            round(max_head_cls, 3),
            " / ",
            round(max_med_cls, 3),
            " / ",
            round(max_tail_cls, 3),
        )
        print(
            "DIST   : ",
            round(max_accuracy_dist, 3),
            " / ",
            round(max_head_dist, 3),
            " / ",
            round(max_med_dist, 3),
            " / ",
            round(max_tail_dist, 3),
        )
        print("\n\n")

        if args.log_results and local_rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "cls_loss": train_stats["cls_loss"],
                    "dst_loss": train_stats["dst_loss"],
                    "val_acc_avg": test_stats["acc1_avg"],
                    "val_acc_cls": test_stats["acc1_cls"],
                    "val_acc_dist": test_stats["acc1_dist"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "head_acc_avg": test_stats["head_acc_avg"],
                    "med_acc_avg": test_stats["med_acc_avg"],
                    "tail_acc_avg": test_stats["tail_acc_avg"],
                    "head_acc_cls": test_stats["head_acc_cls"],
                    "med_acc_cls": test_stats["med_acc_cls"],
                    "tail_acc_cls": test_stats["tail_acc_cls"],
                    "head_acc_dist": test_stats["head_acc_dist"],
                    "med_acc_dist": test_stats["med_acc_dist"],
                    "tail_acc_dist": test_stats["tail_acc_dist"],
                    "best_acc_avg": max_accuracy_avg,
                    "best_head_avg": max_head_avg,
                    "best_med_avg": max_med_avg,
                    "best_tail_avg": max_tail_avg,
                    "sim_12: ": train_stats["sim_12"],
                }
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / (name + "_log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.data_set == "INAT18" or args.data_set == "IMAGENETLT":
        args.name_exp = (
            args.model
            + "_"
            + args.teacher_model
            + "_"
            + str(args.epochs)
            + "_"
            + args.data_set
            + "_"
            + str(args.batch_size)
            + "_"
            + args.experiment
        )
    else:
        args.name_exp = (
            args.model
            + "_"
            + args.teacher_model
            + "_"
            + str(args.epochs)
            + "_"
            + args.data_set
            + "_"
            + "imb"
            + str(int(1 / args.imb_factor))
            + "_"
            + str(args.batch_size)
            + "_"
            + args.experiment
        )
    if args.output_dir:
        Path(os.path.join(Path(args.output_dir), str(args.name_exp))).mkdir(
            parents=True, exist_ok=True
        )

    args.output_dir = Path(os.path.join(Path(args.output_dir), str(args.name_exp)))
    main(args)
