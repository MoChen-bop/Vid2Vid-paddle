import sys 
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from vid2vid.models.networks.network_lib import Conv4x4BNReLU, Conv4x4
from vid2vid.models.networks.base_network import BaseNetwork
from vid2vid.models.networks.discriminators.nlayer_discriminator import NLayerDiscriminator
from vid2vid.models.networks.discriminators.adaptive_discriminator import AdaptiveDiscriminator


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, nc, ndf=64, n_layers=3, stride=2, subarch='n_layers', num_D=3,
        get_intern_feat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.get_intern_feat = get_intern_feat
        self.subarch = subarch

        self.netDs = []
        for i in range(num_D):
            netD = self.add_sublayer('netD_%d' % i, self.create_singleD(subarch, nc, ndf, n_layers,
                stride, get_intern_feat))
            self.netDs.append(netD)
        self.downsample = fluid.dygraph.Pool2D(pool_size=3, pool_type='avg', pool_stride=2, pool_padding=[1, 1])
    

    def create_singleD(self, subarch, nc, ndf, n_layers, stride, get_intern_feat):
        if subarch == 'adaptive':
            netD = AdaptiveDiscriminator(nc, ndf, n_layers, stride, get_intern_feat)
        elif subarch == 'n_layers':
            netD = NLayerDiscriminator(nc, ndf, n_layers, stride, get_intern_feat)
        else:
            raise ValueError('Unrecognized discriminator sub architecture %s' % subarch)
        return netD
    

    def singleD_forward(self, model, inputs, reference):
        if self.subarch == 'adaptive':
            return model(inputs, reference)
        elif self.get_intern_feat:
            return model(inputs)
        else:
            return [model(inputs)]
    

    def forward(self, inputs, reference=None):
        result = []
        input_downsampled = inputs
        ref_downsampled = reference
        for netD in self.netDs:
            result.append(self.singleD_forward(netD, input_downsampled, ref_downsampled))
            input_downsampled = self.downsample(input_downsampled)
            ref_downsampled = self.downsample(ref_downsampled) if reference is not None else None

        return result


if __name__ == '__main__':
    import numpy as np
    with fluid.dygraph.guard():
        discriminator = MultiscaleDiscriminator(3, n_layers=4, get_intern_feat=True)

        image = np.zeros([4, 3, 512, 512]).astype('float32')
        image = fluid.dygraph.to_variable(image)
        outs = discriminator(image)
        
        for out in outs:
            for o in out:
                print(o.shape)
            print()














