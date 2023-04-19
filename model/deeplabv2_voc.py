"""
This is the implementation of DeepLabv2 without multi-scale inputs. This implementation uses ResNet-101 by default.
"""
import kornia.filters
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable

# from model import myNetLoss
from utils import transformsgpu, transformmasks
from utils.loss import CrossEntropy2d, multilabel_focal, loss_calc

affine_par = True


def D(p, z, version='simplified', T=1):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean() / T
    else:
        raise Exception


def D_conv(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return F.mse_loss(p, z)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, *_ = features.shape
        return features.view(b, -1)


def pseudo_sharpen(cluster_out, T_po=0.5, cuda=True):
    with torch.no_grad():
        # cluster_out = F.softmax(cluster_out, dim=1)
        pt = cluster_out ** (1 / T_po)
        cluster_out_pseudo = pt / torch.sum(pt)
    return cluster_out_pseudo


class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp") -> None:
        super().__init__()
        assert head_type in ("mlp", "linear")
        if head_type == "mlp":
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
            )
        else:
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, features):
        return self._header(features)


class PredictionHead(nn.Module):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp") -> None:
        super().__init__()
        assert head_type in ("mlp", "linear")
        if head_type == "mlp":
            self._header = nn.Sequential(
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
            )
        else:
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, output_dim),
            )

    def forward(self, features):
        return self._header(features)


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Linear(in_dim, hidden_dim),
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.layer2 = nn.Sequential(
            # nn.Linear(in_dim, hidden_dim),
            nn.Conv2d(hidden_dim, out_dim, 1, 1),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class prediction_FC(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute([0, 2, 3, 1])
        x = x.reshape(-1, c)
        x = self.fc(x)
        x = x.view(n, h, w, -1).permute([0, 3, 1, 2])
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.project_num = [3, 4]
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.cls = nn.Sequential(
            nn.Conv2d(2048, num_classes, 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )

        # res_model = torchvision.models.resnet101(pretrained=False)
        # self.conv1 = res_model.conv1
        # self.bn1 = res_model.bn1
        # self.relu = res_model.relu
        # self.maxpool = res_model.maxpool  # change
        # self.layer1 = res_model.layer1
        # self.layer2 = res_model.layer2
        # self.layer3 = res_model.layer3
        # self.layer4 = res_model.layer4
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        # self.log_sigma = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1)

        # self.log_sigma = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=1, bias = True)
        self.num_classes = num_classes

        self.predictor = nn.ModuleList([
            # nn.Conv2d(64, num_classes, kernel_size=1),
            # nn.Conv2d(256, num_classes, kernel_size=1),
            # nn.Conv2d(512, num_classes, kernel_size=1),
            # nn.Conv2d(1024, num_classes, kernel_size=1),
            # nn.Conv2d(2048, num_classes, kernel_size=1)

            # prediction_MLP(64, hidden_dim=32, out_dim=num_classes),
            # prediction_MLP(256, hidden_dim=128, out_dim=num_classes),
            # prediction_MLP(512, hidden_dim=256, out_dim=num_classes),
            # prediction_MLP(1024, hidden_dim=512, out_dim=num_classes),
            # prediction_MLP(2048, hidden_dim=1024, out_dim=num_classes),

            prediction_FC(64, num_classes),
            prediction_FC(256, num_classes),
            prediction_FC(512, num_classes),
            prediction_FC(1024, num_classes),
            prediction_FC(2048, num_classes),
        ])

        self.predictor_m_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, num_classes, 3, 2),
                nn.AdaptiveAvgPool2d(1),
            ),
            nn.Sequential(
                nn.Conv2d(256, num_classes, 3, 2),
                nn.AdaptiveAvgPool2d(1),
            ),
            nn.Sequential(
                nn.Conv2d(512, num_classes, 3, 2),
                nn.AdaptiveAvgPool2d(1),
            ),
            nn.Sequential(
                nn.Conv2d(1024, num_classes, 3, 2),
                nn.AdaptiveAvgPool2d(1),
            ),
            nn.Sequential(
                nn.Conv2d(2048, num_classes, 3, 2),
                nn.AdaptiveAvgPool2d(1),
            )
        ])

        # self.muti_class = nn.Sequential(
        #     nn.Conv2d(2048, num_classes, 3, 2),
        #     nn.AdaptiveAvgPool2d(1),
        # )

        self.interp = nn.Upsample(size=(321, 321), mode='bilinear', align_corners=True)

        # self.class_sigma = nn.Conv2d(2048, num_classes, 3, 2)
        # self.bn_class = nn.BatchNorm2d(2048, affine = affine_par)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def new_forward(self, input, mode='weak'):
        feature_list = []

        x = self.conv1(input)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x_encode = self.layer4(x4)
        # log_sigma_x = self.log_sigma(x)
        x6 = self.layer5(x_encode)

        feature_list.append(x1)
        feature_list.append(x2)
        feature_list.append(x3)
        feature_list.append(x4)
        feature_list.append(x_encode)
        # class_x = self.adpool(self.class_mu(x_encoder)).view(x6.shape[0], x6.shape[1])
        return x6, feature_list

    def forward(self, label_input, unlabel_input=None, y_seg=None, y_class=None, mode='sup', method='all'):

        p_l_seg, feature_list = self.new_forward(label_input, mode='sup')
        cls = self.cls(feature_list[-1]).view(feature_list[-1].shape[0], -1)

        project_list = []
        project_m_c_list = []
        for i in self.project_num:
            # gradient = kornia.filters.Sobel(feature_list[i])
            # import matplotlib.pyplot as plt
            # plt.imshow(torch.mean(gradient.normalized, dim=1)[0].detach().cpu().numpy(), cmap='gray')
            # plt.show()
            index = i
            project_list.append(self.predictor[index](feature_list[index]))
            project_m_c_list.append(self.predictor_m_c[index](feature_list[index]).view(p_l_seg.shape[0], -1))

            # pred = torch.argmax(project_list[-1], dim=1)[0].detach().cpu().numpy()
            # plt.imshow(pred, cmap='tab20')
            # plt.show()

        # pred = torch.argmax(p_l_seg, dim=1)[0].detach().cpu().numpy()
        # plt.imshow(pred, cmap='tab20')
        # plt.show()
        return p_l_seg, cls, project_m_c_list, project_list

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.predictor_m_c)
        b.append(self.predictor)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        
        self.conv_sigma = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=1, bias = True)
        #self.mu = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias = True)
        
        self.adpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_classes, num_classes)
        self.class_mu = nn.Conv2d(2048, num_classes, 3, 2)
        self.class_sigma = nn.Conv2d(2048, num_classes, 3, 2)
        self.bn_class = nn.BatchNorm2d(2048, affine 
        """
        b = []
        b.append(self.layer5.parameters())
        # b.append(self.predictor_m_c.parameters())
        # b.append(self.predictor.parameters())
        # b.append(self.projector1.parameters())
        # b.append(self.projector2.parameters())
        # b.append(self.predictor1.parameters())
        # b.append(self.predictor2.parameters())
        # b.append(self.class_mu.parameters())
        # b.append(self.MI_f.parameters())
        # b.append(self.MI_local.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
