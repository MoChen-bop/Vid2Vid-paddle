import sys 
sys.path.append('/home/aistudio')

import paddle 
import paddle.fluid as fluid 

from vid2vid.models.networks.network_lib import Conv3x3BNReLU


class AttentionEncoder(fluid.dygraph.Layer):

    def __init__(self, label_channel, channels, n_reference, n_downsample_A):
        super(AttentionEncoder, self).__init__()

        self.n_reference = n_reference
        self.n_downsample_A = n_downsample_A

        self.key_head_conv = self.add_sublayer('key_head',
            Conv3x3BNReLU(label_channel, channels[0]))
        self.query_head_conv = self.add_sublayer('query_head',
            Conv3x3BNReLU(label_channel, channels[0]))

        self.attn_key_downs = []
        self.attn_query_downs = []
        for i in range(n_downsample_A):
            ch_in, ch_out = channels[i], channels[i + 1]
            key_down = self.add_sublayer('attn_key_down_%d' % i,
                Conv3x3BNReLU(ch_in, ch_out, stride=2))
            query_down = self.add_sublayer('attn_query_down_%d' % i,
                Conv3x3BNReLU(ch_in, ch_out, stride=2))
            self.attn_key_downs.append(key_down)
            self.attn_query_downs.append(query_down)
    

    def encode(self, label, label_ref):
        label = self.query_head_conv(label)
        label_ref = self.key_head_conv(label_ref)
        for i in range(self.n_downsample_A):
            label = self.attn_query_downs[i](label)
            label_ref = self.attn_key_downs[i](label_ref)

        return label, label_ref


    def forward(self, x_img, x_label, label, label_ref):
        b, c, h, w = x_img.shape
        n = self.n_reference
        b = b // n

        attn_query, attn_key = self.encode(label, label_ref)
        attn_query = fluid.layers.reshape(attn_query, (b, c, -1)) # (b, c, h * w)
        
        attn_key = fluid.layers.reshape(attn_key, (b, n, c, -1))
        attn_key = fluid.layers.transpose(attn_key, (0, 1, 3, 2)) # (b, n, h * w, c)
        attn_key = fluid.layers.reshape(attn_key, (b, -1, c)) # (b, n * h * w, c)

        energy = fluid.layers.matmul(attn_key, attn_query) # (b, n * h * w, h * w)
        attention = fluid.layers.softmax(energy, axis=1)

        x_img = fluid.layers.reshape(x_img, (b, n, c, h * w))
        x_img = fluid.layers.transpose(x_img, (0, 2, 1, 3)) # (b, c, n, h * w)
        x_img = fluid.layers.reshape(x_img, (b, c, -1)) # (b, c, n * h * w)

        x_img = fluid.layers.matmul(x_img, attention) # (b, c, h * w)
        x_img = fluid.layers.reshape(x_img, (b, c, h, w))

        x_label = fluid.layers.reshape(x_label, (b, n, c, h * w))
        x_label = fluid.layers.transpose(x_label, (0, 2, 1, 3)) # (b, c, n, h * w)
        x_label = fluid.layers.reshape(x_label, (b, c, -1)) # (b, c, n * h * w)
        x_label = fluid.layers.matmul(x_label, attention) # (b, c, h * w)
        x_label = fluid.layers.reshape(x_label, (b, c, h, w))

        attn_vis = fluid.layers.reshape(attention, (b, n, h * w, h * w))
        attn_vis = fluid.layers.reduce_sum(attn_vis, dim=2)
        attn_vis = fluid.layers.reshape(attn_vis, (b, n, h, w))

        return x_img, x_label, attention, attn_vis


if __name__ == '__main__':
    import numpy as np
    
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        channels = [3, 64, 128, 256, 512, 512]
        attention_encoder = AttentionEncoder(channels[:3], 4, 2)

        label = np.zeros([1, 3, 256, 256]).astype('float32')
        label = fluid.dygraph.to_variable(label)

        label_ref = np.zeros([1 * 4, 3, 256, 256]).astype('float32')
        label_ref = fluid.dygraph.to_variable(label_ref)

        x_img = np.zeros([1 * 4, 128, 64, 64]).astype('float32')
        x_img = fluid.dygraph.to_variable(x_img)
        x_label = np.zeros([1 * 4, 128, 64, 64]).astype('float32')
        x_label = fluid.dygraph.to_variable(x_label)

        x_img, x_label, attention, attn_vis = \
            attention_encoder(x_img, x_label, label, label_ref)
        
        print(x_img.shape)
        print(x_label.shape)
        print(attention.shape)
        print(attn_vis.shape)
    
            






