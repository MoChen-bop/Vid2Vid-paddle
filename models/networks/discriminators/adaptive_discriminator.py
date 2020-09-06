import sys 
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from vid2vid.models.networks.network_lib import Conv4x4BNReLU, Conv4x4
from vid2vid.models.networks.base_network import BaseNetwork
from vid2vid.utils.config import cfg


class AdaptiveDiscriminator(BaseNetwork):

    def __init__(self, nc, ndf=64, n_layers=3, stride=2, get_intern_feat=False, adaptive_layers=1):
        super(AdaptiveDiscriminator, self).__init__()
        self.get_intern_feat = get_intern_feat
        self.n_layers = n_layers
        self.adaptive_layers = adaptive_layers
        self.nc = nc
        self.ndf = ndf 
    
        self.sw = cfg.FINESIZE // 8
        self.sh = int(self.sw / cfg.ASPECTRATIO)
        self.ch = self.sh * self.sw 

        nf = ndf
        fc_0 = self.add_sublayer('fc_0', Linear(self.ch, nc * 4 ** 2))
        encoder_0 = self.add_sublayer('encoder_0', Conv4x4(nc, nf, stride=2, act='leaky_relu'))
        self.fcs = [fc_0]
        self.encoders = [encoder_0]
        for n in range(1, self.adaptive_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            fc = self.add_sublayer('fc_%d' % n, Linear(self.ch, nf_prev * 4 ** 2))
            encoder = self.add_sublayer('encoder_' % n, Conv4x4(nf_prev, nf, stride=2, act='leaky_relu'))
            self.fcs.append(fc)
            self.encoders.append(encoder)
        
        self.determined_convs = []
        for n in range(self.add_parameter, self.n_layers + 1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            conv = self.add_sublayer('conv_%d' % n, Conv4x4BNReLU(nf_prev, nf, stride=stride))
            self.determined_convs.append(conv)


    def pooling(self, reference):
        b, ch, _, _ = reference.shape
        ref = fluid.layers.adaptive_pool2d(input=reference, pool_size=[self.sh, self.sw], pool_type='avg')
        ref = fluid.layers.reshape(ref, (b * ch, -1))
        return ref


    def encode(self, reference):
        encoded_ref = [reference]
        for n in range(self.adaptive_layers):
            ref = self.encoders[n](encoded_ref[-1])
            encoded_ref.append(ref)
        return encoded_ref[1:]

    
    def gen_conv_weights(self, encoded_ref):
        convs = []
        nf = self.ndf
        batch_size = encoded_ref[0].size()[0]
        weights = self.fcs[0](self.pooling(encoded_ref[0]))
        weights = fluid.layers.reshape(weights, (batch_size, nf, self.nc, 4, 4))
        conv0 = []
        for b in range(batch_size):
            conv = None
            conv0.append(conv)
        convs.append(conv0)

        for n in range(1, self.adaptive_layers):
            weights = self.fcs[n](self.pooling(encoded_ref[n]))

            nf_prev = nf
            nf = min(nf * 2, 512)
            weights = fluid.layers.reshape(weights, (batch_size, nf, nf_prev, 4, 4))
            convn = []
            for b in range(batch_size):
                conv = None
                convn.append(conv)
            convs.append(convn)
        return convs
    

    def batch_conv(self, convs, x):
        batch_size = x.shape[0]
        ys = [conv[0](x[0:1])]
        for i in range(batch_size):
            yb = conv[i](x[i:i+1])
            ys.append(yi)
        y = fluid.layers.concat(ys, axis=0)
        return y


    def forward(self, inputs, reference):
        encoded_ref = self.encode(reference)
        convs = self.gen_conv_weights(encoded_ref)
        result = [inputs]

        for conv in convs:
            result.append(self.batch_conv(conv, result[-1]))
        
        for conv in self.determined_convs:
            result.append(conv(result[-1]))
    
        if self.get_intern_feat:
            return result[1:]
        else:
            return result[-1]




if __name__ == '__main__':
    pass