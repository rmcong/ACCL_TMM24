from __future__ import print_function

import os
import copy
import sys
import argparse

import tensorboard_logger as tb_logger
import torch.backends.cudnn as cudnn

from datasets import TinyImagenet
from utils import *
from networks.resnet_big import SupConResNet
from losses_negative_only import SupConLoss
# import cifar100 as datagenerator
import miniimagenet as datagenerator
# import tinyimagenet as datagenerator
# import mulitidatasets as datagenerator
from networks.discriminator import Discriminator

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--exp_num', type=int, default=0, help='exp_num')
    parser.add_argument('--target_task', type=int, default=0, help='resume task')
    parser.add_argument('--replay_policy', type=str, choices=['random'], default='random')

    parser.add_argument('--mem_size', type=int, default=00)

    parser.add_argument('--cls_per_task', type=int, default=5)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # hyper
    parser.add_argument('--seed', type=int, default=820,
                        help='seed')
    parser.add_argument('--num_tasks', type=int, default=20,
                        help='num_tasks')

    parser.add_argument('--workers', type=int, default=4,
                        help='workers')
    parser.add_argument('--pc_valid', type=float, default=0.05,
                        help='pc_valid')
    parser.add_argument('-device', type=str, default='cuda',
                        help='device')

    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['cifar10', 'tiny-imagenet', 'path', 'miniimagenet', 'cifar100', 'multi'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=84, help='parameter for RandomResizedCrop')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/', help='path to save checkpoints')

    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--diff', type=str, default='yes',
                        help='differemtiate loss between share and private feature')
    parser.add_argument('--use_memory', type=bool, default=True,
                        help='previous task samples replay or not')
    parser.add_argument('--s_steps', type=int, default=3,
                        help='shared module update step')
    parser.add_argument('--s_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--s_wd', type=float, default=0.01,
                        help='learning rate decay')
    parser.add_argument('--d_steps', type=int, default=1,
                        help='discriminator module update step')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--d_wd', type=float, default=0.01,
                        help='learning rate decay')
    parser.add_argument('--lr_patience', type=int, default=30,
                        help='lr_decay_patience')
    parser.add_argument('--lr_factor', type=int, default=5,
                        help='lr_factor')
    parser.add_argument('--lr_min', type=float, default=1e-8,
                        help='lr_decay_patience')
    parser.add_argument('--mom', type=float, default=1e-6,
                        help='lr_decay_patience')
    parser.add_argument('--head_unit', type=int, default=128,
                        help='classification head hidden nodes')
    parser.add_argument('--units', type=int, default=175,
                        help='discriminator hidden nodes')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='discriminator hidden nodes')
    parser.add_argument('--lam', type=int, default=1,
                        help='differentiate loss lambda')
    parser.add_argument('--differentiate_index', type=float, default=0.1,
                        help='differentiate loss index')
    parser.add_argument('--adv_index', type=float, default=0.005,
                        help='adversarial loss index')
    parser.add_argument('--contrastive_index', type=float, default=0.15,
                        help='adversarial loss index')

    parser.add_argument('--mu', type=float, default=0.0,
                        help='noise mu')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='noise sigma')
    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.end_task = opt.n_cls // opt.cls_per_task
        opt.size = 32
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 5
        opt.end_task = opt.n_cls // opt.cls_per_task
        opt.size = 64
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 10
        opt.end_task = opt.n_cls // opt.cls_per_task
        opt.size = 64
    elif opt.dataset == 'miniimagenet':
        opt.n_cls = 100
        opt.cls_per_task = 5
        opt.end_task = opt.n_cls // opt.cls_per_task
        opt.size = 84
    elif opt.dataset == 'multi':
        opt.n_cls = 50
        opt.cls_per_task = 10
        opt.end_task = opt.n_cls // opt.cls_per_task
        opt.size = 32
    else:
        pass


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './data/'
    opt.model_path = './save_{}_{}/{}_models'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.tb_path = './save_{}_{}/{}_tensorboard'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.log_path = './save_{}_{}/logs'.format(opt.replay_policy, opt.mem_size, opt.dataset)

    opt.model_name = '{}_{}_{}_lr_{}_bsz_{}'.\
        format(opt.dataset, opt.size, opt.model, opt.s_lr, opt.batch_size
               )

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        # opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:  # cosine learning rate decay
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # checkpoint folder
    opt.checkpoint = os.path.join(opt.checkpoint, str(opt.exp_num))
    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    return opt



def set_model(opt):
    model = SupConResNet(name=opt.model, opt=opt, feat_dim=opt.latent_dim)
    if opt.target_task > 0:  # target 19
        model = prepare_model(opt.target_task, opt)

    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.shared_encoder = torch.nn.DataParallel(model.shared_encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, discriminator, optimizer_dis, model, model2, criterion, optimizer, epoch, opt):
    model.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()
    c_loss = AverageMeter()

    end = time.time()
    for idx, (images, labels, t_labels, tt, td) in enumerate(train_loader['train']):
        data_time.update(time.time() - end)

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)

        warmup_learning_rate(opt, epoch, idx, len(train_loader['train']), optimizer)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.to(opt.device)
            labels = labels.to(opt.device)
            t_labels = t_labels.to(opt.device)
            tt = tt.to(opt.device)

        if opt.use_memory:
            t_current = (opt.target_task) * torch.ones_like(tt)
            body_mask = torch.eq(t_current, tt).cpu().numpy()
            x_task_module = images.clone()
            for index in range(labels.size(0)):
                if body_mask[index] == 0:
                    x_task_module[index] = x_task_module[index].detach()
                    x_task_module[index + labels.size(0)] = x_task_module[index + labels.size(0)].detach()
            x_task_module = x_task_module.to(device=opt.device)

        t_real_D = td.to(opt.device)
        t_fake_D = torch.zeros_like(t_real_D).to(opt.device)
        t_labels = t_labels.repeat(2).view(2, t_labels.size(0)).permute(1, 0).contiguous()
        t_real_D = t_real_D.repeat(2).view(2, t_real_D.size(0)).permute(1, 0).contiguous()
        t_fake_D = t_fake_D.repeat(2).view(2, t_fake_D.size(0)).permute(1, 0).contiguous()

        # ================================================================== #
        #                        Train Shared Module                          #
        # ================================================================== #
        # training S for s_steps
        for s_step in range(opt.s_steps):
            optimizer.zero_grad()
            optimizer_dis.zero_grad()
            model.zero_grad()
            discriminator.zero_grad()
            # warm-up learning rate
            if opt.use_memory:
                shared_features, private_features, shared_embedding, private_embedding = model(images, x_task_module,
                                                                                               return_feat=True)  # [1024, 128]projection head output, [1024, 512] embedding, layer5 ouput
            else:
                shared_features, private_features, shared_embedding, private_embedding = model(images, images,
                                                                                               return_feat=True)  # [1024, 128]projection head output, [1024, 512] embedding, layer5 ouput

            if opt.target_task > 0:
                shared_features1_prev_task = shared_features
                private_feature1_prev_task = private_features

                shared_features1_sim = torch.div(torch.matmul(shared_features1_prev_task, shared_features1_prev_task.T), opt.current_temp)
                private_features1_sim = torch.div(torch.matmul(private_feature1_prev_task, private_feature1_prev_task.T), opt.current_temp)

                logits_mask = torch.scatter(
                    torch.ones_like(shared_features1_sim),
                    1,
                    torch.arange(shared_features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )  # previous samples regard as positive samples, build label
                logits_max1, _ = torch.max(shared_features1_sim * logits_mask, dim=1, keepdim=True)
                private_logits_max1, _ = torch.max(private_features1_sim*logits_mask, dim=1, keepdim=True)
                shared_features1_sim = shared_features1_sim - logits_max1.detach()
                private_features1_sim = private_features1_sim - private_logits_max1.detach()
                row_size = shared_features1_sim.size(0)
                shared_logits1 = torch.exp(shared_features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    shared_features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                private_logits1 = torch.exp(private_features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    private_features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                shared_features1_prev_task = shared_features1_prev_task.view(-1, opt.latent_dim)
                private_feature1_prev_task = private_feature1_prev_task.view(-1, opt.latent_dim)
                shared_private_sim1 = torch.div(torch.matmul(shared_features1_prev_task, private_feature1_prev_task.T), opt.current_temp)
                logits_max_sp1, _ = torch.max(shared_private_sim1, dim=1, keepdim=True)
                s_p_feature_sim1 = shared_private_sim1 - logits_max_sp1.detach()
                s_p_logits1 = torch.exp(s_p_feature_sim1.view(row_size, -1)) / torch.exp(
                    s_p_feature_sim1.view(row_size, -1)).sum(dim=1, keepdim=True)

            bsz = opt.batch_size
            f1, f2 = torch.split(shared_features, [shared_features.size(0)//2, shared_features.size(0)//2], dim=0)  # 512, 128  two augment features same samples
            shared_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 512, 2, 128
            # shared_contrastive_labels = torch.ones_like(labels)
            # share_loss = criterion(shared_features, shared_contrastive_labels, target_labels=list(
            #     range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task)))
            p1, p2 = torch.split(private_features, [private_features.size(0)//2, private_features.size(0)//2], dim=0)
            private_features = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)  # 512, 2, 128
            private_loss = criterion(private_features, labels, target_labels=list(
                range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task)))

            shared_private_labels = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)], dim=0)
            # mask = torch.eq(shared_private_labels, shared_private_labels.T)
            shared_private_loss = 0
            shared_private_contrastive_criterion = SupConLoss(temperature=opt.temp, contrast_mode='one')
            for i in range(shared_features.size(1)):
                shared_private_contrastive_feature = torch.cat([shared_features[:,i], private_features[:, i]], dim=0).view(shared_features.size(0), 2, -1)
                shared_private_loss += shared_private_contrastive_criterion.forward_shared_and_private(shared_private_contrastive_feature,mask=shared_private_labels,target_labels=list(
                range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task)))

            contrastive_loss = private_loss + shared_private_loss
            # contrastive_loss = private_loss

            if opt.target_task > 0:
                with torch.no_grad():
                    share_features2, private_features2 = model2(images, x_task_module)  # previous model output

                    share_features2_prev_task_sim = torch.div(torch.matmul(share_features2, share_features2.T), opt.past_temp)
                    shared_logits_max2, _ = torch.max(share_features2_prev_task_sim * logits_mask, dim=1, keepdim=True)
                    features2_sim = share_features2_prev_task_sim - shared_logits_max2.detach()
                    share_logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                        features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                    private_features2_sim = torch.div(torch.matmul(private_features2, private_features2.T), opt.past_temp)
                    private_logits_max2, _ = torch.max(private_features2_sim * logits_mask, dim=1, keepdim=True)
                    private_features2_sim = private_features2_sim - private_logits_max2.detach()
                    private_logits2 = torch.exp(private_features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                        private_features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                    share_features2 = share_features2.view(-1, opt.latent_dim)
                    private_features2 = private_features2.view(-1, opt.latent_dim)

                    shared_private_sim2 = torch.div(torch.matmul(share_features2, private_features2.T), opt.past_temp)
                    logits_max_sp, _ = torch.max(shared_private_sim2, dim=1, keepdim=True)
                    s_p_feature_sim2 = shared_private_sim2 - logits_max_sp.detach()
                    s_p_logits2 = torch.exp(s_p_feature_sim2.view(row_size, -1)) / torch.exp(
                        s_p_feature_sim2.view(row_size, -1)).sum(dim=1, keepdim=True)

                loss_distill = (-share_logits2 * torch.log(shared_logits1)).sum(1).mean() + \
                               (-private_logits2 * torch.log(private_logits1)).sum(1).mean() +\
                               (-s_p_logits2 * torch.log(s_p_logits1)).sum(1).mean()
                contrastive_loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)

            shared_plus_private_feature = torch.cat([shared_features, private_features], dim=2)
            cls_result = model.forward_cls(shared_plus_private_feature)
            task_loss = model.task_loss(cls_result.view(-1, opt.cls_per_task), t_labels.view(-1))

            dis_out_gen_training = discriminator.forward(shared_features)

            adv_loss = model.adversarial_loss(dis_out_gen_training.view(-1, opt.target_task+2), t_real_D.view(-1))

            if opt.diff == 'yes':
                diff_loss = model.differentiate_loss(shared_features, private_features)
            else:
                diff_loss = torch.tensor(0).to(device=opt.device, dtype=torch.float32)
                opt.differentiate_index = 0

            total_loss = task_loss + opt.adv_index * adv_loss + opt.differentiate_index * diff_loss + contrastive_loss * opt.contrastive_index
            total_loss.backward(retain_graph=True)


            # update metric
            losses.update(total_loss.item(), bsz)
            c_loss.update(contrastive_loss.item())
            # SGD
            # optimizer.zero_grad()
            # loss.backward()
            optimizer.step()
            optimizer_dis.step()

        # ================================================================== #
        #                          Train Discriminator                       #
        # ================================================================== #
        # training discriminator for d_steps
        for d_step in range(opt.d_steps):
            optimizer_dis.zero_grad()
            discriminator.zero_grad()

            # training discriminator on real data
            # gradient has been cleaned, forward the data again
            if opt.use_memory:
                # output = model(images, x_task_module, tt)
                shared_features, private_feature, shared_embedding, priivate_embedding = model(images, x_task_module,
                                                                                               return_feat=True)  # [1024, 128]projection head output, [1024, 512] embedding, layer5 ouput
            else:
                # output = model(images, images)
                shared_features, private_feature, shared_embedding, priivate_embedding = model(images, images, return_feat=True)


            # training discriminator on real data
            f1, f2 = torch.split(shared_features, [shared_features.size(0)//2, shared_features.size(0)//2], dim=0)  # 512, 128  two augment features same samples
            shared_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 512, 2, 128

            dis_real_out = discriminator.forward(shared_features.detach())
            dis_real_loss = model.adversarial_loss(dis_real_out.view(-1, opt.target_task+2), t_real_D.view(-1))
            if opt.dataset == 'miniimagenet':
                dis_real_loss *= opt.adv_index
            dis_real_loss.backward(retain_graph=True)

            # training discriminator on fake data
            z_fake = torch.as_tensor(np.random.normal(opt.mu, opt.sigma, (shared_features.size(0), 2, opt.latent_dim)),
                                     dtype=torch.float32, device=opt.device)
            dis_fake_out = discriminator.forward(z_fake)

            dis_fake_loss = model.adversarial_loss(dis_fake_out.view(-1, opt.target_task+2), t_fake_D.view(-1))
            if opt.dataset == 'miniimagenet':
                dis_fake_loss *= opt.adv_index
            dis_fake_loss.backward(retain_graph=True)

            optimizer_dis.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})\t'
                  'contrastive loss {contrastive_loss.avg:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill, contrastive_loss=c_loss))
            sys.stdout.flush()

    return losses.avg, model2

def prepare_model(task_id, opt):

    # Load a previous model and grab its shared module
    old_net = load_checkpoint(task_id-1, opt)
    old_shared_module = old_net.shared_encoder.state_dict()

    # Instantiate a new model and replace its shared module
    model = SupConResNet(name=opt.model, opt=opt, feat_dim=opt.latent_dim)
    model.shared_encoder.load_state_dict(old_shared_module)
    model = model.to(opt.device)

    return model

def load_checkpoint(task_id, opt):
    print("Loading checkpoint for task {} ...".format(task_id))

    # Load a previous model
    net=SupConResNet(name=opt.model, opt=opt, feat_dim=opt.latent_dim)
    checkpoint=torch.load(os.path.join(opt.checkpoint, 'model_{}.pth.tar'.format(task_id)))
    net.load_state_dict(checkpoint['model_state_dict'])
    net=net.to(opt.device)
    return net

from copy import deepcopy
def get_model(model):
    return deepcopy(model.state_dict())


def get_discriminator(opt, task_id):
    discriminator=Discriminator(opt, task_id).to(opt.device)
    return discriminator

def eval_(model, discriminator, data_loader, task_id, opt):
    loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
    correct_d, correct_t = 0, 0
    num=0
    batch=0

    model.eval()
    discriminator.eval()

    res={}
    with torch.no_grad():
        for idx, (images, labels, t_labels, tt, td) in enumerate(data_loader):
            images = images[0]

            x=images.to(device=opt.device)
            y=t_labels.to(device=opt.device, dtype=torch.long)
            tt=tt.to(device=opt.device)
            t_real_D=td.to(opt.device)

            # Forward
            shared_features, private_features, shared_embedding, priivate_embedding = model(x, x,
                                                                                           return_feat=True)

            prediction = model.forward_cls(torch.cat([shared_features, private_features], dim=-1))
            _, pred = prediction.max(1)
            correct_t += pred.eq(y.view_as(pred)).sum().item()

            # Discriminator's performance:
            output_d = discriminator(shared_features)
            _, pred_d=output_d.max(1)
            correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

            # Loss values
            task_loss=model.task_loss(prediction, y)
            adv_loss=model.adversarial_loss(output_d, t_real_D)

            if opt.diff == 'yes':
                diff_loss=model.differentiate_loss(shared_features, private_features)
            else:
                diff_loss=torch.tensor(0).to(device=opt.device, dtype=torch.float32)
                opt.differentiate_index=0

            total_loss = task_loss + opt.adv_index * adv_loss + opt.differentiate_index * diff_loss

            loss_t+=task_loss
            loss_a+=adv_loss
            loss_d+=diff_loss
            loss_total+=total_loss

            num+=x.size(0)

    res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
    res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
    res['loss_d']=loss_d.item() / (batch + 1)
    res['loss_tot']=loss_total.item() / (batch + 1)
    # res['size']=self.loader_size(data_loader)

    return res

def load_model(model, task_id, opt):

    # Load a previous model
    net=SupConResNet(name=opt.model, opt=opt, feat_dim=opt.latent_dim)
    checkpoint=torch.load(os.path.join(opt.checkpoint, 'model_{}.pth.tar'.format(task_id)))
    net.load_state_dict(checkpoint['model_state_dict'])

    # # Change the previous shared module with the current one
    current_shared_module=deepcopy(model.shared_encoder.state_dict())
    net.shared_encoder.load_state_dict(current_shared_module)

    net = net.to(opt.device)
    return net

def load_discriminator(model, task_id, opt):

    # Load a previous model
    net=get_discriminator(opt, task_id)
    checkpoint=torch.load(os.path.join(opt.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))
    net.load_state_dict(checkpoint['model_state_dict'])

    # # Change the previous shared module with the current one
    current_shared_module=deepcopy(model.state_dict())
    net.load_state_dict(current_shared_module)

    net = net.to(opt.device)
    return net

def load_test_checkpoint(model, task_id, opt):
    print("Loading checkpoint for task {} ...".format(task_id))

    # Load a previous model
    net=SupConResNet(name=opt.model, opt=opt, feat_dim=opt.latent_dim)
    checkpoint=torch.load(os.path.join(opt.checkpoint, 'model_{}.pth.tar'.format(task_id)))
    net.load_state_dict(checkpoint['model_state_dict'])
    shared_encoder_state_dict = deepcopy(model.shared_encoder.state_dict())
    shared_head_state_dict = deepcopy(model.shared_head.state_dict())
    net.shared_encoder.load_state_dict(shared_encoder_state_dict)
    net.shared_head.load_state_dict(shared_head_state_dict)
    net=net.to(opt.device)
    return net

def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def report_val(res):
    # Validation performance
    print(' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
        res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def save_all_models(model, discriminator, task_id, opt):
    print("Saving all models for task {} ...".format(task_id+1))
    dis=get_model(discriminator)
    torch.save({'model_state_dict': dis,
                }, os.path.join(opt.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))

    model=get_model(model)
    torch.save({'model_state_dict': model,
                }, os.path.join(opt.checkpoint, 'model_{}.pth.tar'.format(task_id)))

def main():
    opt = parse_option()

    model2, _ = set_model(opt)  # initialize share encoder model2
    s_lr = opt.s_lr
    d_lr = opt.d_lr
    dis_lr_update = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    original_epochs = opt.epochs

    if opt.end_task is not None:

        opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    # read history acc from record
    acc=np.zeros((opt.num_tasks, opt.num_tasks),dtype=np.float32)
    dataloader = datagenerator.DatasetGen(opt)  # iCifar100

    if opt.target_task > 0:
        acc_txt = os.path.join(opt.checkpoint, '{}_{}_task_{}.txt'.format(opt.dataset, opt.num_tasks, opt.exp_num))
        acc_list = []
        file = open(acc_txt)
        for line in file.readlines():
            line = line.strip('\n')
            data = line.split(' ')
            a = np.array(data, dtype=float)
            acc_list.append(a)
            acc = np.array(np.vstack(acc_list))
        file.close()
        # read dataloader memory
        dataloader_p = torch.load(os.path.join(opt.checkpoint,'{}_{}_task_{}_trainloader.pth'.format(opt.dataset, opt.num_tasks, opt.exp_num)), map_location='cpu')
        dataloader.task_memory = copy.deepcopy(dataloader_p.task_memory)
        del dataloader_p
    lss=np.zeros((opt.num_tasks,opt.num_tasks),dtype=np.float32)

    for target_task in range(0 if opt.target_task == 0 else opt.target_task+1, opt.end_task):
        # build model and criterion
        best_loss = np.inf
        best_loss_d = np.inf
        best_acc = 0
        opt.target_task = target_task
        opt.current_task = target_task + 1
        model, criterion = set_model(opt)

        model2.eval()
        s_lr = opt.s_lr
        d_lr = opt.d_lr
        # build optimizer
        optimizer = set_optimizer(opt, model)


        model2 = copy.deepcopy(model)

        print('Start Training current task {}'.format(opt.target_task))

        # initialize task specific discriminator
        discriminator = get_discriminator(opt, target_task)
        optimizer_dis = torch.optim.SGD(discriminator.parameters(), weight_decay=opt.d_wd, lr=opt.d_lr)

        train_loader = dataloader.get(target_task)
        train_loader = train_loader[target_task]

        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):

            # train for one epoch
            time1 = time.time()
            loss, model2 = train(train_loader, discriminator, optimizer_dis, model, model2, criterion, optimizer, epoch, opt)
            # loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, opt)

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            logger.log_value('learning_rate_{target_task}'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)

            train_res = eval_(model, discriminator, train_loader['train'],task_id=target_task, opt=opt)

            # Valid and adapt learning rate for S and D
            valid_res = eval_(model=model, discriminator=discriminator, data_loader=train_loader['valid'], task_id=target_task, opt=opt)
            report_val(valid_res)
            # Adapt lr for S and D
            if valid_res['loss_tot'] < best_loss:
                best_loss=valid_res['loss_tot']
                best_model=get_model(model)
                patience=opt.lr_patience
                print(' *', end='')
            else:
                patience-=1
                if patience <= 0:
                    s_lr/=opt.lr_factor
                    print(' lr={:.1e}'.format(s_lr), end='')
                    if s_lr < opt.lr_min:
                        print()
                        break
                    patience=opt.lr_patience
                    optimizer=torch.optim.SGD(model.parameters(), weight_decay=opt.s_wd, lr=s_lr)


            if train_res['loss_a'] < best_loss_d:
                best_loss_d=train_res['loss_a']
                best_model_d=get_model(discriminator)
                patience_d=opt.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=opt.lr_factor
                    print(' Dis lr={:.1e}'.format(d_lr))
                    if d_lr < opt.lr_min:
                        dis_lr_update=False
                        print("Dis lr reached minimum value")
                        print()
                    patience_d=opt.lr_patience
                    optimizer_dis=torch.optim.SGD(discriminator.parameters(), weight_decay=opt.d_wd, lr=d_lr)
            print()

        # Restore best validation model (early-stopping)
        model.load_state_dict(copy.deepcopy(best_model))
        discriminator.load_state_dict(copy.deepcopy(best_model_d))
        torch.save(dataloader, os.path.join(opt.checkpoint,'{}_{}_task_{}_trainloader.pth'.format(opt.dataset, opt.num_tasks, opt.exp_num)), _use_new_zipfile_serialization=False)
        save_all_models(model=model, discriminator=discriminator, task_id=target_task, opt=opt)
        for u in range(opt.target_task+1):
            # Load previous model and replace the shared module with the current one
            test_loader_data = datagenerator.DatasetGen(opt)

            test_model = load_model(model, u, opt)
            test_loader = test_loader_data.get_test_dataloader(u)
            test_loader_name = test_loader[u]['name']
            test_loader = test_loader[u]['test']
            test_res = eval_(data_loader=test_loader, task_id=u, model=test_model, discriminator=discriminator, opt=opt)

            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loader_name,
                                                                                          test_res['loss_t'],
                                                                                          test_res['acc_t']))
            acc[target_task, u] = test_res['acc_t']
            lss[target_task, u] = test_res['loss_t']
            del test_loader_data
        del test_model
        # save the last model
        print('Saved accuracies at '+os.path.join(opt.checkpoint,'{}_{}_task.txt'.format(opt.dataset, opt.num_tasks)))
        np.savetxt(os.path.join(opt.checkpoint,'{}_{}_task_{}.txt'.format(opt.dataset, opt.num_tasks, opt.exp_num)),acc,'%.6f')

    avg_acc, gem_bwt = print_log_acc_bwt(opt.num_tasks, acc, lss, output_path=opt.checkpoint)

def print_log_acc_bwt(taskcla, acc, lss, output_path):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run.p')
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt



if __name__ == '__main__':
    main()
