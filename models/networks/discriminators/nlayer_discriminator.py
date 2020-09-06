import sys 
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from vid2vid.models.networks.network_lib import Conv4x4BNReLU, Conv4x4
from vid2vid.models.networks.base_network import BaseNetwork

class NLayerDiscriminator(BaseNetwork):

    def __init__(self, nc, ndf=64, n_layers=3, stride=2, get_intern_feat=False):
        super(NLayerDiscriminator, self).__init__()
        self.get_intern_feat = get_intern_feat
        self.n_layers = n_layers

        self.conv0 = Conv4x4BNReLU(nc, ndf, stride)
        nf = ndf
        self.convs = []
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            conv = self.add_sublayer('conv%d' % n, Conv4x4BNReLU(nf_prev, nf, stride))
            self.convs.append(conv)
        
        nf_prev = nf 
        nf = min(nf * 2, 512)
        self.out_conv0 = Conv4x4BNReLU(nf_prev, nf, 1)
        self.out_conv1 = Conv4x4(nf, 1, 1)

    
    def forward(self, inputs):
        x = self.conv0(inputs)
        result = [x]

        for hidden_conv in self.convs:
            x = hidden_conv(x)
            result.append(x)

        x = self.out_conv0(x)
        result.append(x)

        out = self.out_conv1(x)
        result.append(out)

        if self.get_intern_feat:
            return result
        else:
            return result[-1]


if __name__ == '__main__':
    import numpy as np
    with fluid.dygraph.guard():
        discriminator = NLayerDiscriminator(3, n_layers=4, get_intern_feat=True)

        image = np.zeros([4, 3, 512, 512]).astype('float32')
        image = fluid.dygraph.to_variable(image)
        out = discriminator(image)
        print(len(out))
        for i in range(len(out)):
            print(out[i].shape)
    