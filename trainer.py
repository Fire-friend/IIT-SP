import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import kornia
from evaluate import VOCColorize
from utils import ramps, transformmasks
from utils.loss import loss_calc, multilabel_focal
from utils.util import weakTransform, strongTransform, save_heat, Bn_Controller
import torch.distributed as dist


def train_iter_warm_up(model, batch, optimizer, i_iter, Cluster_bar, args=None, rank=None, ema_mode=None):
    """
        Warm up an iteration for model.
        :param model:network
        :param batch: a batch data.
        :param optimizer:
        :param i_iter:
        :return:
        """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    images_label, labels, _, _, _, label_class, _, images_label_s, labels_s, label_class_s = batch

    images_label = images_label.cuda()
    label_class = label_class.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    p_l_seg, p_l_cls, p_l_cl_project_list, p_l_seg_project_list = model(images_label)

    ema_mode(images_label)

    # supervised
    p_l_seg_resize = interp(p_l_seg)
    l_seg_loss = loss_calc(p_l_seg_resize, labels, weight=None)
    l_cls_project_loss = 0
    l_seg_project_loss = 0
    for i in range(len(p_l_cl_project_list)):
        l_cls_project_loss += multilabel_focal(p_l_cl_project_list[i], label_class)
        n, c, h, w = p_l_seg_project_list[i].shape
        l_seg_project_loss += F.cross_entropy(p_l_seg_project_list[i],
                                              F.interpolate(labels.unsqueeze(1), (h, w)).squeeze(1).long(),
                                              ignore_index=255)
    l_seg_project_loss /= len(p_l_seg_project_list)
    l_cls_project_loss /= len(p_l_cl_project_list)
    l_cls_loss = multilabel_focal(p_l_cls, label_class)
    SL_loss = l_seg_loss + l_cls_loss
    loss = SL_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logging.info(
        '[warm up]iteration %d : loss : %f, loss_seg: %f, loss_cls: %f, loss_seg_l: %f, lr: %f' %
        (i_iter, loss.item(), l_seg_loss.item(), l_cls_loss.item(), l_seg_project_loss.item(),
         # loss_class.item(), loss_entropy.item(),
         optimizer.state_dict()['param_groups'][0]['lr']))
    # Cluster_bar.update(prediction=p_l_seg_resize, label=labels, ignore=args.ignore_label)


def class_mix_forward(unlabel_input, label_input, label_gt, model, ema, args, Cluster_bar, mix_mask='class',
                      Bn_Controller=None):
    with torch.no_grad():
        index_sh = np.arange(2 * unlabel_input.shape[0])
        np.random.shuffle(index_sh)
        weak_parameters = {"flip": 0}

        unlabel_input_weak, _ = weakTransform(weak_parameters, data=unlabel_input)
        # Bn_Controller.freeze_bn(ema)
        logit_ul_seg_weak, logit_ul_cls_weak, logit_ul_cl_project_list_weak, logit_ul_seg_project_list_weak = ema(
            unlabel_input)
        # Bn_Controller.unfreeze_bn(ema)
        h, w = map(int, args.input_size.split(','))
        logit_ul_seg_weak_resize = F.interpolate(logit_ul_seg_weak, (h, w), mode='bilinear', align_corners=True)

        Cluster_bar.update(prediction=logit_ul_seg_weak_resize, ignore=args.ignore_label)

        logit_ul_seg_weak_inverse, _ = weakTransform(weak_parameters, data=logit_ul_seg_weak_resize)
        p_ul_seg_weak_inverse_pro, ul_seg_weak_inverse_cls = torch.max(torch.softmax(logit_ul_seg_weak_inverse, dim=1), dim=1)

        # cat_seg_weak_cls = torch.cat([label_gt, p_ul_seg_weak_inverse_cls], dim=0)
        cat_seg_weak_cls = ul_seg_weak_inverse_cls
        # cat_seg_weak_cls = cat_seg_weak_cls[index_sh]

        ema_cluster_pro = torch.diag(Cluster_bar.cluster_bar)

        if mix_mask == "class":
            for image_i in range(cat_seg_weak_cls.shape[0]):

                classes = torch.unique(cat_seg_weak_cls[image_i]).long()
                classes = classes[classes != 255]
                # classes = classes[classes != 0]
                nclasses = classes.shape[0]

                if Cluster_bar.step % 2 == 1:
                    cur_cluster_pro = ema_cluster_pro[classes.long()]
                    # mean_cur_cluster_pro = torch.mean(cur_cluster_pro)
                    # cur_cluster_pro[cur_cluster_pro < mean_cur_cluster_pro] = 1
                    sort_index = torch.argsort(cur_cluster_pro)
                    num = int((nclasses - nclasses % 2) / 2)
                    classes = classes[sort_index[:num]]  # hard
                else:
                    if nclasses == 1:
                        classes = (classes[[0]])
                    else:
                        classes = (classes[torch.Tensor(
                            np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2),
                                             replace=False)).long()]).cuda()
                if image_i == 0:
                    MixMask = transformmasks.generate_class_mask(cat_seg_weak_cls[image_i], classes).unsqueeze(0).unsqueeze(1).cuda()
                else:
                    MixMask = torch.cat(
                        (MixMask,
                         transformmasks.generate_class_mask(cat_seg_weak_cls[image_i], classes).unsqueeze(0).unsqueeze(1).cuda()))
        elif mix_mask == 'cut':
            img_size = unlabel_input.shape[2:4]
            for image_i in range(unlabel_input.shape[0]):
                if image_i == 0:
                    MixMask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(
                        0).cuda().float()
                else:
                    MixMask = torch.cat(
                        (MixMask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(
                            0).cuda().float()))
        elif mix_mask == "cow":
            img_size = unlabel_input.shape[2:4]
            sigma_min = 8
            sigma_max = 32
            p_min = 0.5
            p_max = 0.5
            for image_i in range(unlabel_input.shape[0]):
                sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))  # Random sigma
                p = np.random.uniform(p_min, p_max)  # Random p
                if image_i == 0:
                    MixMask = torch.from_numpy(
                        transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(
                        0).cuda().float()
                else:
                    MixMask = torch.cat((MixMask, torch.from_numpy(
                        transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(
                        0).cuda().float()))
        elif mix_mask == None:
            # MixMask = torch.ones((images_unlabel_s_w.shape)).cuda()
            MixMask = torch.ones(size=(
                unlabel_input.shape[0], 1, unlabel_input.shape[2], unlabel_input.shape[3])).cuda()

        random_flip = True
        color_jitter = True
        gaussian_blur = True
        strong_parameters = {"Mix": MixMask}
        if random_flip:
            strong_parameters["flip"] = np.random.randint(0, 1)
        else:
            strong_parameters["flip"] = 0
        if color_jitter:
            strong_parameters["ColorJitter"] = np.random.uniform(0, 1)
        else:
            strong_parameters["ColorJitter"] = 0
        if gaussian_blur:
            strong_parameters["GaussianBlur"] = np.random.uniform(0, 1)
        else:
            strong_parameters["GaussianBlur"] = 0

        # cat_input = torch.cat([label_input, unlabel_input], dim=0)
        cat_input = unlabel_input
        # cat_input = cat_input[index_sh]

        cat_input_strong, _ = strongTransform(strong_parameters, data=cat_input)
        # _, cat_seg_weak_cls_strong = strongTransform(strong_parameters, target=cat_seg_weak_cls.unsqueeze(1))
        # cat_seg_weak_cls_strong = cat_seg_weak_cls_strong.squeeze(1)

        _, cat_logit_seg_weak_cls_strong = strongTransform(strong_parameters,
                                                           target=logit_ul_seg_weak_inverse)
        cat_prob_seg_weak_cls_strong = torch.softmax(cat_logit_seg_weak_cls_strong, dim=1)
        cat_pre_pro_seg_weak_cls_strong, cat_seg_weak_cls_strong_cls = torch.max(cat_prob_seg_weak_cls_strong, dim=1)
        # cat_seg_weak_cls_strong = cat_seg_weak_cls_strong ** (1 / 0.5)

        # p_ul_seg_weak_inverse_strong_pro, p_ul_seg_weak_inverse_strong_cls = torch.max(p_ul_seg_weak_inverse_strong,dim=1)
        # n, _, _, _ = cat_input_strong.shape
        mix_input_strong = cat_input_strong
        mix_label_strong = cat_seg_weak_cls_strong_cls
        # cat_logit_seg_weak_cls_strong = torch.argmax(cat_logit_seg_weak_cls_strong, dim=1)

        kexi = (cat_pre_pro_seg_weak_cls_strong - torch.min(cat_pre_pro_seg_weak_cls_strong)) / \
               (torch.max(cat_pre_pro_seg_weak_cls_strong) - torch.min(cat_pre_pro_seg_weak_cls_strong))
        weight_loss = kexi * (1 - ema_cluster_pro[mix_label_strong]) ** 1
        # weight_loss = cat_pre_pro_seg_weak_cls_strong > 0.96

        mix_cl_strong = torch.zeros(size=(len(cat_seg_weak_cls_strong_cls), args.num_classes)).cuda()
        for i in range(len(cat_seg_weak_cls_strong_cls)):
            cl, nums = torch.unique(cat_seg_weak_cls_strong_cls[i], return_counts=True)
            mix_cl_strong[i][cl[nums > 100]] = 1

        mix_label_strong_list = []
        for i in range(len(logit_ul_seg_project_list_weak)):
            _, temp = strongTransform(strong_parameters, target=logit_ul_seg_project_list_weak[i])
            mix_label_strong_list.append(torch.argmax(temp, dim=1))
        # mix_cl_strong

    # feed strong  p_ul_cl_project_list_weak, p_ul_seg_project_list_weak
    # Bn_Controller.freeze_bn(model)
    p_mix_seg_strong, p_mix_cls_strong, p_mix_cl_project_list_strong, p_mix_seg_project_list_strong = model(
        mix_input_strong)
    # Bn_Controller.unfreeze_bn(model)
    # p_mix_seg_strong = model(mix_input_strong, mode='semi')

    p_mix_seg_strong_resize = F.interpolate(p_mix_seg_strong, (h, w), mode='bilinear', align_corners=True)

    # hard
    # weight = torch.max(p_mix_seg_strong_resize_pro, dim=1)[0] ** 2
    # mix_seg_loss = F.cross_entropy(p_mix_seg_strong_resize, cat_seg_weak_cls_strong, ignore_index=255, reduction='none')
    # mix_seg_loss = torch.sum(mix_seg_loss * weight) / torch.sum(mix_seg_loss != 0)

    # with torch.no_grad():
    #     th = ema_cluster_pro * 0.5 + 0.5  # c
    #     # th = th.view(1, -1, 1, 1)
    #     weight = torch.softmax(p_mix_seg_strong_resize, dim=1)
    #     # index = weight > th
    #     pro, cls = torch.max(weight, dim=1)
    #     index = pro > th[cls]

    if Cluster_bar.step % 2 == -1:
        mix_seg_loss = F.mse_loss(torch.softmax(p_mix_seg_strong_resize, dim=1), mix_label_strong)
    else:
        mix_seg_loss = F.cross_entropy(p_mix_seg_strong_resize, mix_label_strong, reduction='none')
        mix_seg_loss = torch.mean(weight_loss * mix_seg_loss)
    # mix_seg_loss = torch.sum(index * mix_seg_loss) / torch.sum(index)

    with torch.no_grad():
        prob_ul_cls_weak = torch.sigmoid(logit_ul_cls_weak).unsqueeze(2).unsqueeze(3)
        prob_ul_cls_weak[prob_ul_cls_weak > 0.1] = 0
        prob_ul_seg_weak = torch.softmax(logit_ul_seg_weak_inverse, dim=1)
        prob_ul_seg_weak[prob_ul_seg_weak > 0.1] = 0


    mix_cis_cls_seg_loss = prob_ul_cls_weak * torch.softmax(p_mix_seg_strong_resize, dim=1) + torch.sigmoid(
        p_mix_cls_strong).unsqueeze(2).unsqueeze(3) * prob_ul_seg_weak

    with torch.no_grad():
        if Cluster_bar.step % 100 == 0:  # show temp matrix
            # plt.imshow(torch.softmax(Cluster_bar.cluster_bar, dim=1).cpu().detach().numpy(), cmap='jet')
            # plt.savefig('./Cluster_bar/{}_bar.png'.format(Cluster_bar.step))
            save_heat(Cluster_bar.cluster_bar.cpu().detach().numpy(), Cluster_bar.step,
                      './Cluster_bar/')
            input = \
            kornia.augmentation.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(mix_input_strong).permute(
                [0, 2, 3, 1])[0].detach().cpu().numpy()
            input_weak = \
                kornia.augmentation.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    unlabel_input).permute(
                    [0, 2, 3, 1])[0].detach().cpu().numpy()
            colorize = VOCColorize()
            weak_show = colorize(mix_label_strong[0].detach().cpu().numpy()).transpose(1, 2, 0)
            strong_show = colorize(torch.argmax(p_mix_seg_strong_resize, dim=1)[0].detach().cpu().numpy()).transpose(1, 2, 0)
            plt.subplot(1, 4, 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(input_weak)
            plt.subplot(1,4,2)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(input)
            plt.subplot(1,4,3)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(weak_show)
            plt.subplot(1,4,4)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(strong_show)
            plt.savefig('./Cluster_bar/{}_com.png'.format(Cluster_bar.step))

    seg_project_loss = 0
    cl_project_loss = 0
    cl_seg_consistency_loss = 0

    for i in range(len(p_mix_seg_project_list_strong)):
        n, c, _, _ = p_mix_seg_project_list_strong[i].shape
        p_mix_seg_project = F.interpolate(p_mix_seg_project_list_strong[i], (h, w), mode='bilinear', align_corners=True)
        seg_project_loss += torch.mean(
            weight_loss * F.cross_entropy(p_mix_seg_project, mix_label_strong, reduction='none'))

        p_mix_cl_pro_project_list_strong = torch.sigmoid(p_mix_cl_project_list_strong[i])
        with torch.no_grad():
            # beta = np.random.beta(0.4, 0.4)
            # index = np.arange(0, len(p_mix_cl_pro_project_list_strong))
            # np.random.shuffle(index)
            mix_label = 0.5 * p_mix_cl_pro_project_list_strong + 0.5 * mix_cl_strong
        cl_project_loss += F.mse_loss(torch.sigmoid(p_mix_cl_project_list_strong[i]), mix_label)
        p = torch.softmax(p_mix_seg_project_list_strong[i], dim=1)
        index = torch.max(p, dim=1, keepdim=True)[0] < 0.1
        cl_seg_consistency_loss += torch.sum(torch.sigmoid(p_mix_cl_project_list_strong[i].view(n, c, 1, 1)) * \
                                             p * index) / (torch.sum(index) * args.num_classes + 1e-10)

    seg_project_loss /= len(p_mix_seg_project_list_strong)
    cl_project_loss /= len(p_mix_cl_project_list_strong)
    cl_seg_consistency_loss / len(p_mix_seg_project_list_strong)
    # cl_seg_consistency_loss =
    return mix_seg_loss + seg_project_loss + cl_seg_consistency_loss + cl_project_loss


def train_iter_semi(model, label_batch,
                    unlabel_batch,
                    optimizer,
                    i_iter,
                    cur_ratio,
                    args=None,
                    Cluster_bar=None,
                    EMA_model=None,
                    rank=None,
                    Bn_Controller=None):
    """
    Training an iteration for model by [clean batch&dirty batch] provided by another network.
    :param model: network
    :param producer: network to product clean data and dirty data.
    :param label_batch: (n1, c, h, w);
    :param unlabel_batch: (n2, c, h, w);  n1 + n2 = batch size
    :param optimizer:
    :return:e
    """
    optimizer.zero_grad()

    afa = ramps.sigmoid_rampup(i_iter, args.num_steps)

    weak_parameters = {"flip": 0}

    images_label, labels, _, labels_names, _, label_class, _, images_label_s, labels_s, label_class_s = label_batch
    images_label = images_label.cuda()
    labels = labels.cuda()
    label_class = label_class.cuda()
    images_unlabel, _, _, unlabel_names, _, _, seed, images_unlabel_s, _, _ = unlabel_batch
    images_unlabel = images_unlabel.cuda()
    images_unlabel_s = images_unlabel_s.cuda()
    h, w = map(int, args.input_size.split(','))

    # supervised ---------------------------
    p_l_seg, p_l_cls, p_l_cl_project_list, p_l_seg_project_list = model(images_label)
    p_l_seg_resize = F.interpolate(p_l_seg, (h, w), mode='bilinear', align_corners=True)

    # Cluster_bar.update(prediction=p_l_seg_resize, label=labels, ignore=args.ignore_label)

    # l_seg_loss = loss_calc(p_l_seg_resize, labels, weight=None)
    ema_cluster_pro = Cluster_bar.get_pro_class()
    temp_label = labels.long()
    temp_label[temp_label == 255] = 0
    weight_loss = (1 - ema_cluster_pro[temp_label]) ** 2

    l_seg_loss = F.cross_entropy(p_l_seg_resize, labels.long(), ignore_index=255, reduction='none')
    l_seg_loss = torch.sum(l_seg_loss * weight_loss) / torch.sum(labels != 255)

    l_cls_project_loss = 0
    l_seg_project_loss = 0
    for i in range(len(p_l_cl_project_list)):
        l_cls_project_loss += multilabel_focal(p_l_cl_project_list[i], label_class)
        # n, c, h, w = p_l_seg_project_list[i].shape
        p_l_seg_project_resize = F.interpolate(p_l_seg_project_list[i], (h, w), mode='bilinear', align_corners=True)
        l_seg_project_loss += F.cross_entropy(p_l_seg_project_resize,
                                              labels.long(),
                                              ignore_index=255)
    l_seg_project_loss /= len(p_l_seg_project_list)
    l_cls_project_loss /= len(p_l_cl_project_list)

    l_cls_loss = multilabel_focal(p_l_cls, label_class)

    SL_loss = l_seg_loss + l_cls_loss  # + 0.1 * l_cls_project_loss  # + 0.1 * l_seg_project_loss  #  # # + 0.1 * l_cls_project_loss

    # semi-supervised --------------------------
    ul_seg_loss = class_mix_forward(unlabel_input=images_unlabel,
                                    label_input=images_label,
                                    label_gt=labels,
                                    model=model,
                                    ema=EMA_model,
                                    Cluster_bar=Cluster_bar,
                                    args=args,
                                    Bn_Controller=Bn_Controller)
    SSL_loss = ul_seg_loss
    loss = SL_loss + SSL_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(i_iter)
    logging.info(
        '[SEMI]iteration %d : loss : %f, SL_loss:%f, SSL_loss: %f, l_seg_loss: %f, l_cls_loss: %f, l_seg_l_loss: %f, ul_seg_loss: %f, afa: %f, lr: %f, ratio: %f' %
        (i_iter, loss.item(), SL_loss.item(), SSL_loss.item(),
         l_seg_loss.item(), l_cls_project_loss.item(), l_seg_project_loss.item(),
         ul_seg_loss.item(),
         afa, optimizer.state_dict()['param_groups'][0]['lr'], cur_ratio))
