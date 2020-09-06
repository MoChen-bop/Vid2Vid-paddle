from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

# __all__ = ['VGGNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19']


# class VGGNet():

#     def __init__(self, layers=16):
#         self.layers = layers
    

#     def net(self, input, class_dim=1000):
#         layers = self.layers
#         vgg_spec = {
#             11: ([1, 1, 2, 2, 2]),
#             13: ([2, 2, 2, 2, 2]),
#             16: ([2, 2, 3, 3, 3]),
#             19: ([2, 2, 4, 4, 4])
#         }

#         assert layers in vgg_spec.keys(), \
#             "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)
        
#         nums = vgg_spec[layers]
#         conv1 = self.conv_block(input, 64, nums[0], name="conv1_")
#         conv2 = self.conv_block(conv1, 128, nums[1], name='conv2_')
#         conv3 = self.conv_block(conv2, 256, nums[2], name="conv3_")
#         conv4 = self.conv_block(conv3, 512, nums[3], name="conv4_")
#         conv5 = self.conv_block(conv4, 512, nums[4], name="conv5_")

#         fc_dim = 4096
#         fc_name = ["fc6", "fc7", "fc8"]
#         fc1 = fluid.layers.fc(
#             input=conv5, size=fc_dim, act='relu',
#             param_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "weights"),
#             bias_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_offset"))
#         fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
#         fc2 = fluid.layers.fc(
#             input=fc1, size=fc_dim, act='relu',
#             param_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_weights"),
#             bias_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_offset"))
#         fc2 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
#         out = fluid.layers.fc(
#             input=fc2, size=class_dim,
#             param_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_weights"),
#             bias_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_offset"))
        
#         return out
    

#     def conv_block(self, input, num_filter, groups, name=None):

#         conv = input
#         for i in range(groups):
#             conv = fluid.layers.conv2d(
#                 input=conv,
#                 num_filters=num_filter,
#                 stride=1,
#                 padding=1,
#                 act='relu',
#                 param_attr=fluid.param_attr.ParamAttr(
#                     name=name + str(i + 1) + "_weights"),
#                 bias_attr=False)
#         return fluid.layers.pool2d(
#             input=conv, pool_size=2, pool_type='max', pool_stride=2)


# def VGG11():
#     model = VGGNet(layers=11)
#     return model


# def VGG13():
#     model = VGGNet(layers=13)
#     return model


# def VGG16():
#     model = VGGNet(layers=16)
#     return model


# def VGG19():
#     model = VGGNet(layers=19)
#     return model



if __name__ == '__main__':
    from vgg import VGG19

    model_path = '/home/aistudio/vid2vid/pretrained_models/vgg19/VGG19_pretrained'
    state_dict = fluid.io.load_program_state(model_path,)
    print(state_dict.keys())

    with fluid.dygraph.guard():
# ['conv1.conv1_1.conv.weight', 'conv1.conv1_2.conv.weight', 'conv2.conv2_1.conv.weight',
#  'conv2.conv2_2.conv.weight', 'conv3.conv3_1.conv.weight', 'conv3.conv3_2.conv.weight', 
#  'conv3.conv3_3.conv.weight', 'conv3.conv3_4.conv.weight', 'conv4.conv4_1.conv.weight', 
#  'conv4.conv4_2.conv.weight', 'conv4.conv4_3.conv.weight', 'conv4.conv4_4.conv.weight', 
#  'conv5.conv5_1.conv.weight', 'conv5.conv5_2.conv.weight', 'conv5.conv5_3.conv.weight', 
#  'conv5.conv5_4.conv.weight']
# ['conv5_3_weights', 'conv5_2_weights', 'conv3_2_weights', 'conv1_2_weights', 'conv4_2_weights', 
# 'fc8_weights', 'conv3_1_weights', 'conv2_2_weights', 'conv2_1_weights', 'conv5_4_weights', 'fc7_offset', 
# 'conv5_1_weights', 'fc6_weights', 'fc7_weights', 'conv4_4_weights', 'conv4_3_weights', 'conv3_4_weights',
# 'fc6_offset', 'conv3_3_weights', 'conv4_1_weights', 'fc8_offset', 'conv1_1_weights']
        model = VGG19()
        model.state_dict()['conv1.conv1_1.conv.weight'].set_value(state_dict['conv1_1_weights'])
        model.state_dict()['conv1.conv1_2.conv.weight'].set_value(state_dict['conv1_2_weights'])
        model.state_dict()['conv2.conv2_1.conv.weight'].set_value(state_dict['conv2_1_weights'])
        model.state_dict()['conv2.conv2_2.conv.weight'].set_value(state_dict['conv2_2_weights'])
        model.state_dict()['conv3.conv3_1.conv.weight'].set_value(state_dict['conv3_1_weights'])
        model.state_dict()['conv3.conv3_2.conv.weight'].set_value(state_dict['conv3_2_weights'])
        model.state_dict()['conv3.conv3_3.conv.weight'].set_value(state_dict['conv3_3_weights'])
        model.state_dict()['conv3.conv3_4.conv.weight'].set_value(state_dict['conv3_4_weights'])
        model.state_dict()['conv4.conv4_1.conv.weight'].set_value(state_dict['conv4_1_weights'])
        model.state_dict()['conv4.conv4_2.conv.weight'].set_value(state_dict['conv4_2_weights'])
        model.state_dict()['conv4.conv4_3.conv.weight'].set_value(state_dict['conv4_3_weights'])
        model.state_dict()['conv4.conv4_4.conv.weight'].set_value(state_dict['conv4_4_weights'])
        model.state_dict()['conv5.conv5_1.conv.weight'].set_value(state_dict['conv5_1_weights'])
        model.state_dict()['conv5.conv5_2.conv.weight'].set_value(state_dict['conv5_2_weights'])
        model.state_dict()['conv5.conv5_3.conv.weight'].set_value(state_dict['conv5_3_weights'])
        model.state_dict()['conv5.conv5_4.conv.weight'].set_value(state_dict['conv5_4_weights'])
        
        fluid.save_dygraph(model.state_dict(), '/home/aistudio/vid2vid/pretrained_models/vgg19/dygraph')

        














