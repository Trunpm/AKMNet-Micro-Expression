import torch
import torch.nn as nn
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



# class BinarizedF(Function):
#     def forward(self, input):
#         self.save_for_backward(input)
#         ones = torch.ones_like(input)
#         zeros = torch.zeros_like(input)
#         output = torch.where(input>0,ones,zeros)
#         return output
#     def backward(self, output_grad):
#         input, = self.saved_tensors
#         ones = torch.ones_like(input)
#         zeros = torch.zeros_like(input)
#         input_grad = output_grad*torch.where((0<input)&(input<1), ones, zeros)
#         return input_grad
# class BinarizedModule(nn.Module):
#   def __init__(self):
#     super(BinarizedModule, self).__init__()
#   def forward(self,input):
#     output =BinarizedF()(input)
#     return output

class BinarizedF(Function):
    def forward(self, input):
        self.save_for_backward(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        output = input.clone()
        for n in range(input.shape[0]):
            output[n,:] = torch.where(input[n,:]>=torch.mean(input[n,:]),ones,zeros)
        return output
    def backward(self, output_grad):
        input, = self.saved_tensors
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = output_grad.clone()
        for n in range(input.shape[0]):
            input_grad[n,:] = output_grad[n,:]*torch.where((1>torch.mean(input[n,:]))&(0<torch.mean(input[n,:])), ones, zeros)
        return input_grad
class BinarizedModule(nn.Module):
  def __init__(self):
    super(BinarizedModule, self).__init__()
  def forward(self,input):
    output =BinarizedF()(input)
    return output



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())
        self.BinarizedModulev1 = BinarizedModule()

        self.gru = nn.GRU(512*1*1, 32, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(planes[3] * block.expansion * 16 // 8, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.uniform_(m.bias_ih_l0)
                nn.init.uniform_(m.bias_hh_l0)
                nn.init.orthogonal_(m.weight_ih_l1)
                nn.init.orthogonal_(m.weight_hh_l1)
                nn.init.uniform_(m.bias_ih_l1)
                nn.init.uniform_(m.bias_hh_l1)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, Volume):
        Volume = Volume.permute(0, 2, 1, 3, 4).contiguous()   ###N C T H W  to  N T C H W
        x = Volume.view(-1, Volume.size(2), Volume.size(3), Volume.size(4))  ### N*T C H W
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) ### N*T 512 4 4


        BX1 = self.avgpool(x).squeeze(3).squeeze(2) ### N*T 512
        BX2 = BX1*self.alpha(BX1)### N*T 512
        BXwhole = BX2.view(Volume.size(0),Volume.size(1),512).sum(dim=1)### N 512
        Bw = torch.matmul(nn.functional.normalize(BX1).view(Volume.size(0),Volume.size(1),512),nn.functional.normalize(BXwhole).unsqueeze(2)).squeeze(2)### N T
        B = self.BinarizedModulev1(Bw) ### N T     0 1
        

        ###for N==1 so:
        x = x.view(Volume.size(0),Volume.size(1),512,4,4).mul(B.repeat(512,4,4,1,1).permute(3, 4, 0, 1, 2).contiguous()) ### N T 512 4 4
        xBO = torch.index_select(x, 1, (B.squeeze()==1).nonzero().squeeze()) ###N ? 512 4 4
        xBO = xBO.permute(0, 3, 4, 1, 2).contiguous() #N 4 4 ? 512
        xBO = xBO.view(-1, xBO.size(3), 512) #N*4*4 ? 512
        out, _ = self.gru(xBO) #N*4*4 ? 64
        # out = out[:,-1,:] #N*4*4 1 64
        out = torch.mean(out,dim=1) #N*4*4 64
        out = out.view(-1, 4*4*64) #N 4*4*64
        out = self.fc(out) #N 4

        return Bw,B,out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model