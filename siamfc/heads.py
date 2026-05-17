from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiamFC']

import os
import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

#from siamfc.backbones import NONLocalBlock2D, SELayer1, SELayer, ECALayer, SimAM
from siamfc.backbones import SELayer1, SELayer, ECALayer, SimAM


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        # self.att1 = nn.Sequential(NONLocalBlock2D(256))

        #self.att = nn.Sequential(SELayer1(256))

        # self.att2 = nn.Sequential(SimAM(256))

    def forward(self,z, x):
        # a = self.att(z)
        # z = z + a

        #z = z.view(8,-1,z.size(2),z.size(3))
        # out1 = self._fast_xcorr(z1, x1) * self.out_scale
        # out2 = self._fast_xcorr(z2, x2) * self.out_scale
        # out = 0.3 * out1 + 0.7 * out2
        # return out
        #return self._fast_xcorr(z, x) * self.out_scale
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        #fast cross correlation
        #nz和nx为batchsize，即这次喂进去的图片的张数
        nz = z.size(0)  # 1
        nx, c, h, w = x.size()  # 8 128 22 22
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
    
    def xcorr_depthwise(self,z,x):   #z(1,256,5,5)  x(3,256,21,21)  #在线测试
        nz = z.size(0) #nz为batchsize
        nx = x.size(0)
        channel = x.size(1)
        #x = x.reshape(8,channel,x.size(2), x.size(3))
        #z = z.reshape(8, channel, z.size(2), z.size(3))
        out = F.conv2d(x, z, groups=nz)
        out = out.reshape(nx,-1, out.size(2), out.size(3))
        return out
                          
   # def xcorr_depthwise(self,z,x):   #z(8,256,5,5)  x(8,256,21,21)   #离线训练
    #    nz = z.size(0)
    #    nx = x.size(0)
    #    channel = z.size(1)
    #    x = x.view(1, nz*channel, x.size(2), x.size(3))
    #    kernel = z.view(nz*channel, 1, z.size(2), z.size(3))
    #    out = F.conv2d(x, kernel, groups=nz*channel)
    #    out = out.view(-1, 1, out.size(2), out.size(3))
    #    return out



