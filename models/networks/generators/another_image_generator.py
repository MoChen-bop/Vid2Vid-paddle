import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from vid2vid.models.networks.network_utils import resample
from vid2vid.models.networks.generators.splited_reference_encoder import ReferenceEncoder
from vid2vid.models.networks.generators.merged_reference_decoder import ReferenceDecoder
from vid2vid.models.networks.generators.simple_label_embedder import SimpleLabelEmbedder
from vid2vid.models.networks.generators.another_weight_generator import WeightGenerator
from vid2vid.models.networks.generators.image_adaptive_generator import AdaptiveGenerator
from vid2vid.models.networks.generators.simple_flow_generator import FlowGenerator


class FewShotGenerator(fluid.dygraph.Layer):

    def __init__(self, ngf=32, image_channel=3, label_channel=1, generator_norm_type='batch',
        reference_n=2, prev_n=2):
        super(FewShotGenerator, self).__init__()

        self.reference_encoder = ReferenceEncoder(image_channel, label_channel, ngf, reference_n)
        self.reference_decoder = ReferenceDecoder(ngf)
        self.label_embedder = SimpleLabelEmbedder(label_channel, ngf)
        self.weight_generator = WeightGenerator(ngf)
        self.generator = AdaptiveGenerator(ngf, generator_norm_type)
        self.flow_generator = FlowGenerator(ngf, image_channel, label_channel, prev_n)

    def forward(self, label, label_refs, img_refs, prevs=None):
        encoded_image, encoded_label, attention, attention_vis = self.reference_encoder(img_refs, label_refs, label)
        decoded_ref = self.reference_decoder(encoded_image, encoded_label)
        conv_weights, norm_weights = self.weight_generator(decoded_ref)
        embedded_label = self.label_embedder(label)
        raw_image = self.generator(encoded_image, embedded_label, conv_weights, norm_weights)

        if prevs is None or prevs[0] is None:
            final_image = raw_image
            warp_image, flow, flow_mask = None, None, None
        else:
            flow, flow_mask = self.flow_generator(label, prevs)
            warp_image = resample(prevs[1][:, -1], flow)
            final_image = raw_image * flow_mask + warp_image * (1 - flow_mask)

        return final_image, raw_image, warp_image, flow, flow_mask, attention, attention_vis


if __name__ == '__main__':
    import numpy as np 
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard(fluid.CPUPlace()):
        generator = FewShotGenerator()
        label = to_variable(np.zeros((4, 1, 256, 256)).astype('float32'))
        label_refs = to_variable(np.zeros((4, 2, 1, 256, 256)).astype('float32'))
        img_refs = to_variable(np.zeros((4, 2, 3, 256, 256)).astype('float32'))
        prevs = [
            to_variable(np.zeros((4, 2, 1, 256, 256)).astype('float32')),
            to_variable(np.zeros((4, 2, 3, 256, 256)).astype('float32'))
        ]
        final_image, raw_image, warp_image, flow, flow_mask, attention, attention_vis = \
            generator(label, label_refs, img_refs, prevs)
        print(final_image.shape) # [1, 3, 256, 256]
        print(raw_image.shape)
        print(warp_image.shape)
        print(flow.shape)
        print(flow_mask.shape)
        print(attention.shape)
        print(attention_vis.shape)
        # [1, 3, 256, 256]
        # [1, 3, 256, 256]
        # [1, 3, 256, 256]
        # [1, 2, 256, 256]
        # [1, 1, 256, 256]
        # [1, 2048, 1024]
        # [1, 2, 32, 32]
