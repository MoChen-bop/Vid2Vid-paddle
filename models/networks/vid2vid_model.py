import sys
sys.path.append('/home/aistudio')
import numpy as np

import paddle
import paddle.fluid as fluid

from vid2vid.models.networks.generators.main_generator import FewShotGenerator
from vid2vid.models.networks.discriminators.multi_scale_discriminator import MultiscaleDiscriminator
from vid2vid.models.networks.losses.loss_collect import LossCollector
from vid2vid.utils.config import cfg


class Vid2VidModel():

    def __init__(self, ):
        self.define_networks()
        self.loss_collector = LossCollector()
    

    def define_networks(self, ):
        self.netG = FewShotGenerator()
        self.netD = MultiscaleDiscriminator(nc=cfg.MODEL.DISCRIMINATOR.NETD_INPUT_NC)
        self.netDT = MultiscaleDiscriminator(nc=cfg.MODEL.DISCRIMINATOR.NETDT_INPUT_NC)


    def train(self,):
        self.netG.train()
        self.netD.train()
        self.netDT.train()


    def forward(self, data_list, mode='generator', save_images=False):
        tgt_label, tgt_image, flow_gt, ref_labels, ref_images, prevs = data_list

        if mode == 'generator':
            g_loss, generated, prev = self.forward_generator(tgt_label, tgt_image, ref_labels, 
                ref_images, prevs, flow_gt)
            return g_loss, generated if save_images else [], prev


    def forward_generator(self, tgt_label, tgt_image, ref_labels, 
        ref_images, prevs=[None] * 3, flow_gt=None, conf_gt=[None] * 2):
        [fake_image, fake_raw_image, warped_image, flow, flow_mask], [fg_mask, ref_fg_mask], \
            prevs_new, atn_score = self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)

        data_list = [tgt_image, fake_image]
        loss_GT_GAN = self.loss_collector.compute_GT_GAN_losses(self.netDT, data_list)

        data_list = [tgt_image, fake_image]
        loss_G_GAN = self.loss_collector.compute_GAN_losses(self.netD, data_list)

        loss_F_Flow, loss_F_Warp = self.loss_collector.compute_flow_losses(flow, flow_gt, None, warped_image, tgt_image)

        loss_F_Mask = self.loss_collector.compute_mask_losses(flow_mask, warped_image, tgt_image)
        
        loss_list = [loss_G_GAN, loss_GT_GAN, loss_F_Flow, loss_F_Warp, loss_F_Mask]
        return loss_list, \
            [fake_image, fake_raw_image, warped_image, flow, flow_mask, atn_score], prevs_new


    def forward_discriminator(self, tgt_label, tgt_image, ref_labels, ref_images, prevs=[None] * 3):
        with fluid.dygraph.no_grad():
            [fake_image, fake_raw_image, _, _, _], [fg_mask, ref_fg_mask], prevs_new, atn_score = \
                self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)
        
        data_list = [tgt_image, fake_image]
        loss_temp = self.loss_collector.compute_DT_GAN_losses(self.netDT, data_list)
    
        data_list = [tgt_image, fake_image]
        loss_indv = self.loss_collector.compute_D_GAN_losses(self.netD, data_list)
        loss_list = [loss_temp, loss_indv]
        return loss_list, prevs_new


    def generate_images(self, tgt_labels, tgt_images, ref_labels, ref_images, prevs=[None] * 3):
        generated_images, atn_score = None, None
        generated_masks = [np.array(1), np.array(1)]
        for t in range(cfg.TRAIN.N_FRAMES):
            tgt_label_t, tgt_image_t, prev_t = self.get_input_t(tgt_labels, tgt_images, prevs, t)
            fake_image, flow, flow_mask, fake_raw_image, warped_image, attn_vis, ref_idx = \
                self.netG(tgt_label_t, ref_labels, ref_images, prev_t)

            generated_images = self.concat([generated_images, 
                [fake_image, fake_raw_image, warped_image, flow, flow_mask]], dim=1)

            prevs = self.concat_prev(prevs, [tgt_label_t, tgt_image_t, fake_image])
        return generated_images, generated_masks, prevs, atn_score


    def get_input_t(self, tgt_labels, tgt_images, prevs, t):
        b, _, _, h, w = tgt_labels.shape
        tgt_label_t = tgt_labels[:, t]
        tgt_image_t = tgt_images[:, t]
        prevs = [prevs[0], prevs[2]]
        return tgt_label_t, tgt_image_t, prevs
    

    def concat_prev(self, prev, now):
        if type(prev) == list:
            return [self.concat_prev(p, n) for p, n in zip(prev, now)]
        
        if prev is None:
            prev = fluid.layers.unsqueeze(now, 1)
            prev = fluid.layers.expand(prev, (1, cfg.MODELS.GENERATOR.N_FRAME, 1, 1, 1))
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


if __name__ == '__main__':
    import numpy as np
    from paddle.fluid.dygraph import to_variable

    with fluid.dygraph.guard():
        tgt_labels = to_variable(np.zeros((1, 4, 1, 128, 256)).astype('float32'))
        tgt_images = to_variable(np.zeros((1, 4, 3, 128, 256)).astype('float32'))
        ref_labels = to_variable(np.zeros((1, 2, 1, 128, 256)).astype('float32'))
        ref_images = to_variable(np.zeros((1, 2, 3, 128, 256)).astype('float32'))
        flow_gt = to_variable(np.zeros((1, 3, 2, 128, 256)).astype('float32'))
        model = Vid2VidModel()
        [fake_image, fake_raw_image, warped_image, flow, flow_mask], [fg_mask, ref_fg_mask], \
            prevs_new, atn_score = model.generate_images(tgt_labels, tgt_images, ref_labels, ref_images)

        print(fake_image.shape)
        print(fake_raw_image.shape)
        print(warped_image.shape)
        print(flow.shape)
        print(flow_mask.shape)
        print(prevs_new[0].shape)
        print(prevs_new[1].shape)
        print(prevs_new[2].shape)

        # loss_list, [fake_image, fake_raw_image, 
        #     warped_image, flow, flow_mask, atn_score], prevs_new = \
        #     model.forward_generator(tgt_labels, tgt_images, ref_labels, ref_images, flow_gt=flow_gt)
        
        # print(loss_list[0].shape)
        # print(loss_list[1].shape)
        # print(loss_list[2].shape)
        # print(loss_list[3].shape)
        # print(loss_list[4].shape)

        loss_temp, loss_indv = model.forward_discriminator(tgt_labels, tgt_images, ref_labels, ref_images)
        print(loss_temp[0].shape)
        print(loss_temp[1].shape)
        print(loss_indv[0].shape)
        print(loss_indv[1].shape)







