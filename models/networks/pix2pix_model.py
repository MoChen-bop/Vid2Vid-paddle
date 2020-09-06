import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from vid2vid.models.networks.generators.simple_reference_encoder import SimpleReferenceEncoder
from vid2vid.models.networks.generators.image_spade_generator import SPADEGenerator

class Pix2pixModel(fluid.dygraph.Layer):

    def __init__(self, cfg):
        super(Pix2pixModel, self).__init__()
        self.reference_encoder = SimpleReferenceEncoder(**cfg.ref_encoder)
        self.generator = SPADEGenerator(**cfg.generator)
    


    def forward(self, label, image_ref, label_ref):

        encoded_ref, attention, attention_vis = self.reference_encoder(image_ref, label_ref, label)
        fake_image = self.generator(encoded_ref, label)

        return fake_image
    


if __name__ == '__main__':
    import numpy as np 
    from paddle.fluid.dygraph import to_variable
    from vid2vid.configs.pix2pix_config import cfg

    with fluid.dygraph.guard():

        model = Pix2pixModel(cfg.model)
        label = to_variable(np.zeros((1, 1, 256, 256)).astype('float32'))
        image_ref = to_variable(np.zeros((1, 4, 3, 256, 256)).astype('float32'))
        label_ref = to_variable(np.zeros((1, 4, 1, 256, 256)).astype('float32'))

        fake_image = model(label, image_ref, label_ref)

        print(fake_image.shape)







