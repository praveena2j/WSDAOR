import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import vgg16
import collections
from models.inflate import inflate_vgg_features
import sys
from utils.functions import ReverseLayerF
from models.VGG_VD_16 import Vgg_vd_face_fer_dag
from models.VGGNet_modified import Vgg_m_face_bn_fer_dag
import numpy as np

class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        #VGG-VD-16 trained on VGG-Faces by S. Albanie (Oxford)
        # 3 224 224 done 512 7 7
        # TODO Try prepend model with cnn to create 3 channels from 1
        #instead of 3 times the grayscale

        #loaded_model = Vgg_vd_face_fer_dag()
        loaded_model = Vgg_m_face_bn_fer_dag()
        #loaded_vggmodel = vgg16(False, num_classes=7) # not pretrained
        self.is_3d = False

        weights = torch.load('pretrainedweights/vgg_m_face_bn_fer_dag.pth')
        loaded_model.load_state_dict(weights)

#===================================
# Transform pytorch VGG into finetuned and inflated
#======================================

        self.features = loaded_model
        self.inflate_features()
        self.avgpool = nn.Sequential(torch.nn.AvgPool2d(kernel_size=(5,5)))
        self.inflate_featurepool()
        self.class_predictions = nn.Sequential(
                    Unit3D(in_channels=512, output_channels=1,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
                    )

        self.domain_predictions = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=2)
        )
#===============================================

        for p in self.features.parameters():
          p.requires_grad = True
        for p in self.class_predictions.parameters():
          p.requires_grad = True
        for p in self.domain_predictions.parameters():
          p.requires_grad = True

        loaded_model = None

    def inflate_features(self):
        self.features = inflate_vgg_features(self.features)
        self.features.cuda()
        self.is_3d = True

    def inflate_featurepool(self):
        self.avgpool = inflate_vgg_features(self.avgpool)
        self.avgpool.cuda()
        self.is_3d = True

    def forward(self, x, alpha):
        B,C,L,H,W = x.size()
        if self.is_3d:
            feature = self.features(x) #BCLHW features
            features = self.avgpool(feature)
            #print(feature.size())
            #print(features.size())
            #sys.exit()

            #print("hello")
            features_domain = torch.max(features, dim=2)[0].squeeze(2).squeeze(2)
            #features_domain = features.permute(2,0,1,3,4) # LBC..
            #features_domain = features_domain.contiguous().view(L*B, 512)

            #x = x.contiguous().view(L,B, 25088)

            #print(features_domain.size())
            #sys.exit()
            #x = x.permute(2,0,1,3,4) # LBC..
            #x = x.contiguous().view(L,B, 25088)
            # L,B ,C is seen as L*B,C for the linear module
        else :
            print("entered")
            #x = x.permute(2,0,1,3,4) # LBC..
            #x = x.contiguous().view(L*B, C,H,W)
            x = self.features(x) # CHW features
            #x = x.contiguous().view(L,B, 25088)
       
        reverse_feature = ReverseLayerF.apply(features_domain, alpha)
        class_output = self.class_predictions(features)
        domain_output = self.domain_predictions(reverse_feature)
        #print(class_output.size())
       
        return features, class_output, domain_output