"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, shared, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if shared:
            hidden_channel = [32, 64, 128, 256]
        else:
            hidden_channel = [16, 32, 32, 64]


        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, hidden_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_channel[3], num_blocks[3], stride=2)
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

class AlexNet(nn.Module):

    def __init__(self, shared, dataset='cifar100'):

        super(AlexNet, self).__init__()
        if shared:
            hidden_channel = [32, 64, 128, 256]
        else:
            hidden_channel = [16, 32, 32, 64]
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        if dataset == 'cifar100':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, hidden_channel[0], kernel_size=11, stride=4, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # nn.BatchNorm2d(hidden_channel[0]),
                nn.Conv2d(hidden_channel[0], hidden_channel[1], kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # nn.BatchNorm2d(hidden_channel[1]),
                nn.Conv2d(hidden_channel[1], hidden_channel[2], kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden_channel[2], hidden_channel[3], kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                # self.conv1,
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),
                # self.conv2,
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),
                # nn.Flatten(),
                # self.fc1,
                # nn.ReLU()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, hidden_channel[0], kernel_size=11, stride=4, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(hidden_channel[0]),
                nn.Conv2d(hidden_channel[0], hidden_channel[1], kernel_size=5, stride=1, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.BatchNorm2d(hidden_channel[1]),
                nn.Conv2d(hidden_channel[1], hidden_channel[2], kernel_size=3, padding=1), nn.ReLU(),
                nn.BatchNorm2d(hidden_channel[2]),

                nn.Conv2d(hidden_channel[2], hidden_channel[3], kernel_size=3, padding=1), nn.ReLU(),

                nn.MaxPool2d(2,2),
                nn.BatchNorm2d(hidden_channel[3]),

                nn.Flatten(),
                # self.conv1,
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),
                # self.conv2,
                # nn.ReLU(),
                # nn.MaxPool2d(2,2),
                # nn.Flatten(),
                # self.fc1,
                # nn.ReLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        # x = self.encoder2(x)
        # x = self.encoder3(x)
        return x


class SupConMLP(nn.Module):
    def __init__(self, name='mlp', opt=None, head='mlp', feat_dim=128):
        super(SupConMLP, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.shared_encoder = model_fun(True, dataset=opt.dataset)
        self.private_encoder = model_fun(False, dataset=opt.dataset)
        if opt.dataset == 'miniimagenet':
            # num_ftrs = 4608
            num_ftrs = 1024  # without average pool (-2)
            hiddens = [64, 128, 256]
        elif opt.dataset == 'cifar100':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'tiny-imagenet':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'cifar10':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'multi':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        else:
            raise NotImplementedError

        self.shared_head = nn.Sequential(
            nn.Linear(num_ftrs, hiddens[0]),
            nn.ReLU(inplace=True),
            # LinearBatchNorm(2*opt.batch_size),
            nn.Dropout(0.5),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hiddens[1], feat_dim),
            # nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

        )
        self.private_head = nn.Sequential(
            nn.Linear(num_ftrs//4, feat_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(feat_dim, feat_dim),
            # # nn.BatchNorm1d(feat_dim),
            #
            # nn.ReLU(inplace=True),
        )
        self.hidden = opt.head_unit
        self.cls_head = torch.nn.Sequential(
            nn.Linear(2*feat_dim, self.hidden),
            # nn.Sigmoid(),
            # LinearBatchNorm(2*opt.batch_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.hidden, self.hidden),
            # nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, opt.cls_per_task)
                )
        # self.shared_head = nn.Sequential(
        #     nn.Linear(num_ftrs, hiddens[0]),
        #     nn.ReLU(inplace=True),
        #     # LinearBatchNorm(2*opt.batch_size),
        #     nn.Dropout(),
        #     nn.Linear(hiddens[0], hiddens[1]),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     nn.Linear(hiddens[1], feat_dim),
        #     # nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #
        # )
        # self.private_head = nn.Sequential(
        #     nn.Linear(num_ftrs//4, feat_dim),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     #
        #     # nn.Linear(feat_dim, feat_dim),
        #     # # nn.BatchNorm1d(feat_dim),
        #     #
        #     # nn.ReLU(inplace=True),
        # )
        # self.hidden = opt.head_unit
        # self.cls_head = torch.nn.Sequential(
        #     nn.Linear(2*feat_dim, self.hidden),
        #     # nn.Sigmoid(),
        #     # LinearBatchNorm(2*opt.batch_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(self.hidden, self.hidden),
        #     # nn.Sigmoid(),
        #     nn.Dropout(0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hidden, opt.cls_per_task)
        #     )
        self.adversarial_loss = nn.CrossEntropyLoss()
        self.task_loss = nn.CrossEntropyLoss()
        self.differentiate_loss = DiffLoss()
    def forward(self, x_s, x_p, task_num=None, return_feat=False, norm=True):
        share_embedding = self.shared_encoder(x_s)
        share_feature = self.shared_head(share_embedding)
        private_embedding = self.private_encoder(x_p)
        private_feature = self.private_head(private_embedding)
        if norm:
            share_feature = F.normalize(share_feature, dim=1)
            private_feature = F.normalize(private_feature, dim=1)

        if return_feat:
            return share_feature, private_feature, share_embedding, private_embedding
        else:
            return share_feature, private_feature

    def forward_cls(self, x):
        cls_result = self.cls_head(x)
        return cls_result


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_small(shared):
    return ResNet(shared, BasicBlock, [2, 2, 2, 2])

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def mlp(shared, dataset='cifar100'):
    return AlexNet(shared, dataset)


model_dict = {
    'resnet_small': [resnet18_small, 128],
    'resnet18': [resnet18_small, 128],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'alexnet': [mlp, 256],
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

class SupConResNet_ori(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet_ori, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(True)
        if head == 'linear':
            self.head = nn.Linear(feat_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, dim_in),
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


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', opt=None, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        # self.device = opt.device
        model_fun, dim_in = model_dict[name]
        self.shared_encoder = model_fun(True)
        self.private_encoder = model_fun(False)
        if opt.dataset == 'miniimagenet':
            # num_ftrs = 4608
            num_ftrs = 256  # without average pool (-2)
            hiddens = [64, 128, 256]
        elif opt.dataset == 'cifar100':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'tiny-imagenet':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'cifar10':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        elif opt.dataset == 'multi':
            # num_ftrs = 25088  # without average pool
            num_ftrs = 256
            hiddens = [64, 128, 256]
        else:
            raise NotImplementedError

        self.shared_head = nn.Sequential(
            nn.Linear(num_ftrs, hiddens[0]),
            nn.ReLU(inplace=True),
            # LinearBatchNorm(2*opt.batch_size),
            nn.Dropout(0.5),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hiddens[1], feat_dim),
            # nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        )
        self.private_head = nn.Sequential(
            nn.Linear(64, feat_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(feat_dim, feat_dim),
            # # nn.BatchNorm1d(feat_dim),
            #
            # nn.ReLU(inplace=True),
        )
        self.hidden = opt.head_unit
        self.cls_head = torch.nn.Sequential(
            nn.Linear(2*feat_dim, self.hidden),
            # nn.Sigmoid(),
            # LinearBatchNorm(2*opt.batch_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.hidden, self.hidden),
            # nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, opt.cls_per_task)
                )
        self.adversarial_loss = nn.CrossEntropyLoss()
        self.task_loss = nn.CrossEntropyLoss()
        self.differentiate_loss = DiffLoss()


    def reinit_head(self):
        for layers in self.shared_head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()
        for layers in self.private_head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()


    def forward(self, x_s, x_p, task_num=None, return_feat=False, norm=True):
        share_embedding = self.shared_encoder(x_s)
        share_feature = self.shared_head(share_embedding)
        private_embedding = self.private_encoder(x_p)
        private_feature = self.private_head(private_embedding)
        if norm:
            share_feature = F.normalize(share_feature, dim=1)
            private_feature = F.normalize(private_feature, dim=1)

        if return_feat:
            return share_feature, private_feature, share_embedding, private_embedding
        else:
            return share_feature, private_feature

    def forward_cls(self, x):
        cls_result = self.cls_head(x)
        return cls_result



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
    def __init__(self, name='resnet50', num_classes=10, two_layers=False, feat_dim=128):
        super(LinearClassifier, self).__init__()
        # _, feat_dim = model_dict[name]
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


class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))