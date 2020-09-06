import sys
sys.path.append('/home/aistudio')
import numpy as np

import paddle
import paddle.fluid as fluid

from vid2vid.models.networks.generators.another_image_generator import FewShotGenerator
from vid2vid.models.networks.discriminators.multi_scale_discriminator import MultiscaleDiscriminator
from vid2vid.models.networks.losses.loss_collect import LossCollector


class FewShotModel():

    def __init__(self, cfg):

        self.n_frames = cfg.n_frames
        self.prev_frames_n = cfg.prev_frames_n
        self.use_true_prev = cfg.use_true_prev
        self.define_networks(cfg)
        self.loss_collector = LossCollector(**cfg.loss)
    

    def define_networks(self, cfg):
        self.netG = FewShotGenerator(**cfg.generator)
        self.netD = MultiscaleDiscriminator(**cfg.discriminator)
        self.netDT = MultiscaleDiscriminator(**cfg.temporal_discriminator)


    def train(self,):
        self.netG.train()
        self.netD.train()
        self.netDT.train()
    

    def eval(self,):
        self.netG.eval()
        self.netD.eval() 
        self.netDT.eval()


    def forward_generator(self, tgt_label, tgt_image, ref_labels, ref_images, prevs=[None] * 3, flow_gt=None):
        [fake_image, fake_raw_image, warped_image, flow, flow_mask, attn_vis], prevs_new = \
            self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)

        data_list = [tgt_image, fake_image]
        loss_GT_GAN = self.loss_collector.compute_GT_GAN_losses(self.netDT, data_list)

        data_list = [tgt_image, fake_image]
        loss_G_GAN = self.loss_collector.compute_GAN_losses(self.netD, data_list)

        fg_mask = None
        loss_F_Flow, loss_F_Warp = self.loss_collector.compute_flow_losses(flow, flow_gt, fg_mask, warped_image, tgt_image)

        loss_F_Mask = self.loss_collector.compute_mask_losses(flow_mask, warped_image, tgt_image)

        loss_VGG = self.loss_collector.compute_VGG_matching_loss(tgt_image, fake_image, fake_raw_image)
        
        loss_list = [loss_G_GAN, loss_GT_GAN, loss_F_Flow, loss_F_Warp, loss_F_Mask, loss_VGG]
        return loss_list, \
            [fake_image, fake_raw_image, warped_image, flow, flow_mask, attn_vis], prevs_new


    def forward_discriminator(self, tgt_label, tgt_image, ref_labels, ref_images, prevs=[None] * 3):
        with fluid.dygraph.no_grad():
            [fake_image, fake_raw_image, _, _, _, _], prevs_new,  = \
                self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)
        
        data_list = [tgt_image, fake_image]
        loss_temp = self.loss_collector.compute_DT_GAN_losses(self.netDT, data_list)
    
        data_list = [tgt_image, fake_image]
        loss_indv = self.loss_collector.compute_D_GAN_losses(self.netD, data_list)
        loss_list = [loss_temp, loss_indv]
        return loss_list, prevs_new


    def generate_images(self, tgt_labels, tgt_images, ref_labels, ref_images, prevs=[None] * 3):
        generated_images = None
        for t in range(self.n_frames):
            tgt_label_t, tgt_image_t, prev_t = self.get_input_t(tgt_labels, tgt_images, prevs, t)
            
            fake_image, raw_image, warp_image, flow, flow_mask, attention, attn_vis = \
                self.netG(tgt_label_t, ref_labels, ref_images, prev_t)

            generated_images = self.concat([generated_images, 
                [fake_image, raw_image, warp_image, flow, flow_mask, attn_vis]], dim=1)

            prevs = self.concat_prev(prevs, [tgt_label_t, tgt_image_t, fake_image])
        return generated_images, prevs


    def get_input_t(self, tgt_labels, tgt_images, prevs, t):
        b, _, _, h, w = tgt_labels.shape
        tgt_label_t = tgt_labels[:, t]
        tgt_image_t = tgt_images[:, t]
        if self.use_true_prev:
            prevs = [prevs[0], prevs[1]]
        else:
            prevs = [prevs[0], prevs[2]]
        return tgt_label_t, tgt_image_t, prevs
    

    def concat_prev(self, prev, now):
        if type(prev) == list:
            return [self.concat_prev(p, n) for p, n in zip(prev, now)]
        
        if prev is None:
            prev = fluid.layers.unsqueeze(now, 1)
            prev = fluid.layers.expand(prev, (1, self.prev_frames_n, 1, 1, 1))
        else:
            prev = fluid.layers.concat([prev[:, 1:], fluid.layers.unsqueeze(now, 1)], axis=1)

        return prev

    
    def concat(self, tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            if isinstance(tensors[0], list):
                tensors_cat = []
                for i in range(len(tensors[0])):
                    if tensors[0][i] is not None:
                        tensors_cat.append(fluid.layers.concat([tensors[0][i],
                            fluid.layers.unsqueeze(tensors[1][i], 1)], dim))
                    else:
                        tensors_cat.append(fluid.layers.unsqueeze(tensors[1][i], 1))
                return tensors_cat
            
            return fluid.layers.concat([tensors[0], 
                fluid.layers.unsqueeze(tensors[1], 1)], dim)
        elif tensors[1] is not None:
            if isinstance(tensors[1], list):
                return [fluid.layers.unsqueeze(t, 1) if t is not None else None for t in tensors[1]]
            return fluid.layers.unsqueeze(tensors[1], 1)
        
        return tensors[0]