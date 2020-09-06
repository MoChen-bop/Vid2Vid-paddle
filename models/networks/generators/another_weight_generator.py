import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import reshape

from vid2vid.models.networks.network_utils import simple_reshape_weight as reshape_weight
from vid2vid.models.networks.network_lib import LinearSNReLU


class WeightGenerator(fluid.dygraph.Layer):

    def __init__(self, ngf=32):
        super(WeightGenerator, self).__init__()

        self.channels = [16 * ngf, 8 * ngf, 4 * ngf, 2 * ngf]

        self.linear_conv_0s = []
        self.linear_conv_1s = []
        self.linear_conv_ss = []

        self.linear_SPADE_0s = []
        self.linear_SPADE_1s = []
        self.linear_SPADE_ss = []

        for i, c in enumerate(self.channels):
            in_nc = c
            ch_out = c * 2 * 9
            self.linear_conv_0s.append(
                fluid.dygraph.Sequential(
                    ('fc_conv_0_%d_1' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_conv_0_%d_2' % i, LinearSNReLU(ch_out, ch_out))
                )
            )
            ch_out = c * 9
            self.linear_conv_1s.append(
                fluid.dygraph.Sequential(
                    ('fc_conv_1_%d_1' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_conv_1_%d_2' % i, LinearSNReLU(ch_out, ch_out))
                )
            )
            ch_out = c * 2 * 1
            self.linear_conv_ss.append(
                fluid.dygraph.Sequential(
                    ('fc_conv_s_%d_1' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_conv_s_%d_2' % i, LinearSNReLU(ch_out, ch_out))
                )
            )
            ch_out = c * 2 * 1 * 2
            self.linear_SPADE_0s.append(
                fluid.dygraph.Sequential(
                    ('fc_SPADE_0_%d_0' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_SPADE_0_%d_1' % i, LinearSNReLU(ch_out, ch_out))
                )
            )
            ch_out = c * 1 * 2
            self.linear_SPADE_1s.append(
                fluid.dygraph.Sequential(
                    ('fc_SPADE_1_%d_0' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_SPADE_1_%d_1' % i, LinearSNReLU(ch_out, ch_out))
                )
            )
            ch_out = c * 2 * 1 * 2
            self.linear_SPADE_ss.append(
                fluid.dygraph.Sequential(
                    ('fc_SPADE_s_%d_0' % i, LinearSNReLU(in_nc, ch_out)),
                    ('fc_SPADE_s_%d_1' % i, LinearSNReLU(ch_out, ch_out))
                )
            )


    def forward(self, decoded_ref):
        b = decoded_ref[0].shape[0]
        conv_weights = []
        norm_weights = []

        for i, c in enumerate(self.channels):
            feat = decoded_ref[i]
            b, _, h, w = feat.shape
            feat = reshape(feat, (-1, c))
            conv_0_w = reshape(self.linear_conv_0s[i](feat), (b, -1))
            conv_1_w = reshape(self.linear_conv_1s[i](feat), (b, -1))
            conv_s_w = reshape(self.linear_conv_ss[i](feat), (b, -1))

            conv_0_w = reshape_weight(conv_0_w, [c, c * 2, 3, 3])
            conv_1_w = reshape_weight(conv_1_w, [c, c, 3, 3])
            conv_s_w = reshape_weight(conv_s_w, [c, c * 2, 1, 1])

            norm_0_w = reshape(self.linear_SPADE_0s[i](feat), (b, -1))
            norm_1_w = reshape(self.linear_SPADE_1s[i](feat), (b, -1))
            norm_s_w = reshape(self.linear_SPADE_ss[i](feat), (b, -1))
            
            norm_0_w = reshape_weight(norm_0_w, [[c * 2, c, 1, 1]] * 2)
            norm_1_w = reshape_weight(norm_1_w, [[c, c, 1, 1]] * 2)
            norm_s_w = reshape_weight(norm_s_w, [[c * 2, c, 1, 1]] * 2)

            conv_weights.append([conv_0_w, conv_1_w, conv_s_w])
            norm_weights.append([norm_0_w, norm_1_w, norm_s_w])
        
        return conv_weights, norm_weights


if __name__ == '__main__':
    import numpy as np 
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard():
        weight_generator = WeightGenerator()
        decoded_ref = [
            to_variable(np.zeros((1, 512, 512, 1)).astype('float32')),
            to_variable(np.zeros((1, 256, 256, 1)).astype('float32')),
            to_variable(np.zeros((1, 128, 128, 1)).astype('float32')),
            to_variable(np.zeros((1, 64, 64, 1)).astype('float32')),
        ]

        conv_weights, norm_weights = weight_generator(decoded_ref)

        for i in range(4):
            print(conv_weights[i][0].shape)
            print(conv_weights[i][1].shape)
            print(conv_weights[i][2].shape)

            print(len(norm_weights)) # 4
            print(len(norm_weights[0])) # 3
            print(len(norm_weights[0][0])) # 2
            print(norm_weights[i][0][0].shape)
            print(norm_weights[i][1][0].shape)
            print(norm_weights[i][2][0].shape)

            # [1, 512, 1024, 3, 3]
            # [1, 512, 512, 3, 3]
            # [1, 512, 1024, 1, 1]

            # [1, 1024, 512, 1, 1]
            # [1, 512, 512, 1, 1]
            # [1, 1024, 512, 1, 1]
