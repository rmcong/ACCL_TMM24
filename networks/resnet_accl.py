# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import torch
import torch.nn as nn
import torch.nn.functional as F
class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()


        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        ncha,size,_ = args.inputsize

        self.pretrained = False

        if args.experiment == 'cifar100':
            hiddens = [64, 128, 256]

        elif args.experiment == 'miniimagenet':
            hiddens = [1024, 512, 256]

        elif args.experiment == 'tiny-imagenet':
            hiddens = [1024, 512, 256]

        elif args.experiment == 'cifar10':
            hiddens = [64, 128, 256]
        else:
            raise NotImplementedError

        # Small resnet
        resnet = SupConResNet(name=args.model)
        self.features = resnet

        if args.experiment == 'miniimagenet':
            # num_ftrs = 4608
            num_ftrs = 2304  # without average pool (-2)

        elif args.experiment == 'cifar100':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
        elif args.experiment == 'tiny-imagenet':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
        elif args.experiment == 'cifar10':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 128
        else:
            raise NotImplementedError

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(num_ftrs,hiddens[0])
        self.fc2=torch.nn.Linear(hiddens[0],hiddens[1])
        self.fc3=torch.nn.Linear(hiddens[1],hiddens[1])
        self.fc4=torch.nn.Linear(hiddens[1], self.latent_dim)

    def forward(self, x):
        x = x.view_as(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.drop2(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.drop2(self.relu(self.fc3(x)))
        x = self.drop2(self.relu(self.fc4(x)))
        return x





class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        ncha,size,_=args.inputsize
        self.image_size = ncha * size * size

        self.taskcla = args.taskcla
        self.latent_dim = args.latent_dim
        self.ntasks = args.num_tasks
        self.samples = args.samples
        self.image_size = ncha * size * size
        self.use_memory = args.use_memory

        self.hidden1 = args.head_units
        self.hidden2 = args.head_units

        self.shared = Shared(args)
        self.private = SupConResNet(name=args.model)
        # self.head = torch.nn.Sequential()
        # if args.use_memory == 'yes':
        #     self.head = []
        #     for i in range(self.ntasks):
        #         classifier = torch.nn.Sequential(
        #                 torch.nn.Linear(2*self.latent_dim, self.hidden1),
        #                 torch.nn.ReLU(inplace=True),
        #                 # torch.nn.Dropout(),
        #                 # torch.nn.Linear(self.hidden1, self.hidden2),
        #                 # torch.nn.ReLU(inplace=True),
        #                 torch.nn.Linear(self.hidden1, self.taskcla[i][1])
        #             )
        #         self.head.append(classifier.to(args.device))
        # else:
        self.head = torch.nn.Sequential(
                    torch.nn.Linear(2*self.latent_dim, self.hidden1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.hidden1, self.taskcla[0][1])
                )


    def forward(self, x_s, x_p, tt=None):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        # x_s = self.shared(x_s)
        # x_p = self.private(x_p)

        # x = torch.cat([x_p, x_s], dim=1)

        # if self.args.experiment == 'multidatasets':
        #     # if no memory is used this is faster:
        #     y=[]
        #     for i,_ in self.taskcla:
        #         y.append(self.head[i](x))
        #     return y[task_id]
        # else:
        #     return torch.stack([self.head[tt[i]].forward(x[i]) for i in range(x.size(0))])


        output = {}
        output['shared'] = self.shared(x_s)
        output['private'] = self.private(x_p)
        concat_features = torch.cat([output['private'], output['shared']], dim=1)
        # if torch.is_tensor(tt):
        #     output['out'] = torch.stack([self.head[tt[i]].forward(concat_features[i]) for i in range(
        #         concat_features.size(0))])
        # else:
        output['out'] = self.head(concat_features)
        return output

        # output['shared'] = self.shared(x_s)
        # output['private'] = self.private(x_p)
        #
        # concat_features = torch.cat([output['private'], output['shared']], dim=1)
        #
        # if torch.is_tensor(tt):
        #
        #     output['out'] = torch.stack([self.head[tt[i]].forward(concat_features[i]) for i in range(concat_features.size(0))])
        # else:
        #     if self.use_memory == 'no':
        #         output['out'] = self.head.forward(concat_features)
        #
        #     elif self.use_memory == 'yes':
        #         y = []
        #         for i, _ in self.taskcla:
        #             y.append(self.head[i](concat_features))
        #         output['out'] = y[task_id]
        #
        #     return output


    # def get_encoded_ftrs(self, x_s, x_p, task_id=None):
    #     return self.shared(x_s), self.private(x_p)


    def print_model_size(self):

        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head[0].parameters() if p.requires_grad)

        print("Size of the network for one task including (S+P+p)")
        print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P       = %s ' % (self.pretty_print(count_P)))
        print('Num parameters in p       = %s ' % (self.pretty_print(count_H)))
        print('Num parameters in P+p    = %s ' % self.pretty_print(count_P + count_H))
        print('-------------------------->   Architecture size in total for all tasks: %s parameters (%sB)' % (
        self.pretty_print(count_S + self.ntasks*count_P + self.ntasks*count_H),
        self.pretty_print(4 * (count_S + self.ntasks*count_P + self.ntasks*count_H))))

        classes_per_task = self.taskcla[0][1]
        print("-------------------------->   Memory size: %s samples per task (%sB)" % (self.samples*classes_per_task,
                                                                                        self.pretty_print(
                                                                                            self.ntasks * 4 * self.samples * classes_per_task* self.image_size)))
        print("------------------------------------------------------------------------------")
        print("                               TOTAL:  %sB" % self.pretty_print(
            4 * (count_S + self.ntasks *count_P + self.ntasks *count_H) + self.ntasks * 4 * self.samples * classes_per_task * self.image_size))

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])




# class _CustomDataParallel(torch.nn.DataParallel):
#     def __init__(self, model):
#         super(_CustomDataParallel, self).__init__(model)
#
#     def __getattr__(self, name):
#         try:
#             return super(_CustomDataParallel, self).__getattr__(name)
#         except AttributeError:
#             return getattr(self.module, name)



#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048]
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def reinit_head(self):
        for layers in self.head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()


    def forward(self, x, return_feat=False, norm=True):
        encoded = self.encoder(x)
        if norm:
            feat = F.normalize(self.head(encoded), dim=1)
        else:
            feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10, two_layers=False):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        if two_layers:
          self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

# class ResNet(nn.Module):
#
#     def __init__(self, shared, block, layers, num_classes, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#
#         # small resnet
#         if shared:
#             hiddens = [32, 64, 128, 256]
#         else:
#             hiddens = [16, 32, 32, 64]
#
#     # original resnet
#         # hiddens = [64, 128, 256, 512]
#
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, hiddens[0], layers[0])
#         self.layer2 = self._make_layer(block, hiddens[1], layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, hiddens[2], layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, hiddens[3], layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(hiddens[3] * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.relu(x)
#         return x
#
    # def forward(self, x):
    #     return self._forward_impl(x)




# def resnet18_small(latend_dim, shared):
#     # r"""ResNet-18 model from
#     # `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     return ResNet(shared, BasicBlock, [2, 2, 2, 2], num_classes=latend_dim)

