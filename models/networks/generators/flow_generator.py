import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from vid2vid.models.networks.network_utils import pick_ref, resample
from vid2vid.models.networks.network_lib import Conv2DSNBNReLU, ResnetBlock
from vid2vid.models.networks.network_lib import ConvUpsample, Conv3x3


class FlowGenerator(fluid.dygraph.Layer):

    def __init__(self, img_channels, label_channels, channels, n_frame_G, n_downsample_F, n_res_block):
        super(FlowGenerator, self).__init__()

        self.flow_network = FlowNetwork(img_channels, label_channels, channels,
            n_frame_G, n_downsample_F, n_res_block, for_reference=False)
    

    def forward(self, label, prev):
        label_prev, img_prev = prev
        has_prev = label_prev is not None

        flow, flow_mask, img_warp = None, None, None

        if has_prev:
            flow_prev, flow_mask_prev = self.flow_network(label, label_prev, img_prev)
            img_prev_warp = resample(img_prev[:, -1,], flow_prev)
            flow, flow_mask, img_warp = flow_prev, flow_mask_prev, img_prev_warp

        return flow, flow_mask, img_warp


class FlowNetwork(fluid.dygraph.Layer):

    def __init__(self, img_channels, label_channels, channels, n_frame_G, n_downsample_F, n_res_block, for_reference=False):
        super(FlowNetwork, self).__init__()

        self.n_frame_G = n_frame_G
        self.n_downsample_F = n_downsample_F
        self.flow_multiplier = 20

        if for_reference:
            input_nc = (img_channels + label_channels) * 1 + label_channels
        else:
            input_nc = (img_channels + label_channels) * n_frame_G + label_channels

        down_flow = [('flow_head', Conv2DSNBNReLU(input_nc, channels[0]))]
        for i in range(n_downsample_F):
            ch_in, ch_out = channels[i], channels[i + 1]
            down_flow += [('down_flow_%d' % i, Conv2DSNBNReLU(ch_in, ch_out, stride=2))]
        self.downflow = fluid.dygraph.Sequential(*down_flow)

        res_flow = []
        res_channel = channels[n_downsample_F]
        for i in range(n_res_block):
            res_flow += [('res_flow_%d' % i, ResnetBlock(res_channel, res_channel))]
        self.res_flow = fluid.dygraph.Sequential(*res_flow)

        up_flow = []
        for i in reversed(range(n_downsample_F)):
            ch_in, ch_out = channels[i + 1], channels[i]
            up_flow += [('up_flow_%d' % (n_downsample_F - i), ConvUpsample(ch_in, ch_out))]
        self.up_flow = fluid.dygraph.Sequential(*up_flow)

        self.conv_flow = self.add_sublayer('flow_branch', Conv3x3(channels[0], 2))
        self.conv_mask = self.add_sublayer('mask_branch', Conv3x3(channels[0], 1))


    def forward(self, label, label_prev, img_prev):
        b, c, h, w = label.shape
        label_prev = fluid.layers.reshape(label_prev, (b, -1, h, w))
        img_prev = fluid.layers.reshape(img_prev, (b, -1, h, w))
        label = fluid.layers.concat([label, label_prev, img_prev], axis=1)
        downsample = self.downflow(label)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat)
        flow_mask = self.conv_mask(flow_feat) * self.flow_multiplier
        flow_mask = fluid.layers.sigmoid(flow_mask)

        return flow, flow_mask


if __name__ == '__main__':
    import numpy as np 
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard(fluid.CPUPlace()):
        flow_generator = FlowGenerator(img_channels=3, label_channels=3, 
            channels=[32, 64, 128, 256, 512, 1024, 1024], 
            n_frame_G=1, n_downsample_F=5, n_res_block=3)

        label = to_variable(np.zeros((1, 3, 512, 512)).astype('float32'))
        prev_img = to_variable(np.zeros((1, 3, 512, 512)).astype('float32'))
        prev_label = to_variable(np.zeros((1, 3, 512, 512)).astype('float32'))

        flow, flow_mask, img_warp, ds_ref = flow_generator(label, (prev_label, prev_img))

        print(flow.shape)
        print(flow_mask.shape)
        print(img_warp.shape)
        # [1, 2, 512, 512]
        # [1, 1, 512, 512]
        # [1, 3, 512, 512]





        