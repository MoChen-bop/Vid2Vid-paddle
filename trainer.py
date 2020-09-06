import os
import sys
sys.path.append('/home/aistudio')
import numpy as np 
from datetime import datetime

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from vid2vid.models.few_shot_model import FewShotModel
from vid2vid.configs.fewshot_config import cfg
from vid2vid.utils.summary import AverageMeter, LogSummary
from vid2vid.utils.timer import Timer
from vid2vid.utils.visualize import save_image, save_flow_image


class Trainer():

    def __init__(self, data_reader):
        self.data_reader = data_reader
        self.model = FewShotModel(cfg.model)
        self.model.train()

        self.optimizer_G = fluid.optimizer.AdamOptimizer(learning_rate=cfg.train.G.lr, 
            beta1=cfg.train.beta1, beta2=cfg.train.beta2,
            parameter_list=self.model.netG.parameters())
        self.optimizer_D = fluid.optimizer.AdamOptimizer(learning_rate=cfg.train.D.lr, 
            beta1=cfg.train.beta1, beta2=cfg.train.beta2,
            parameter_list=self.model.netD.parameters())
        self.optimizer_DT = fluid.optimizer.AdamOptimizer(learning_rate=cfg.train.D.lr, 
            beta1=cfg.train.beta1, beta2=cfg.train.beta2,
            parameter_list=self.model.netDT.parameters())
        
        log_dir = os.path.join(cfg.train.log_dir, cfg.exp_name, datetime.now().strftime('%b%d_%H-%M-%S_'))
        self.logger = LogSummary(log_dir)
        self.global_G_loss = AverageMeter()
        self.global_GT_loss = AverageMeter()
        self.global_G_flow_loss = AverageMeter()
        self.global_G_warp_loss = AverageMeter()
        self.global_G_mask_loss = AverageMeter()
        self.global_G_vgg_loss = AverageMeter()

        self.global_D_real_loss = AverageMeter()
        self.global_D_fake_loss = AverageMeter()
        self.global_DT_real_loss = AverageMeter()
        self.global_DT_fake_loss = AverageMeter()

        self.timer = Timer()
        self.timer.start()


    def read_data(self,):
        for step, data_list in enumerate(self.data_reader()):
            tgt_labels = np.array([x['tgt_label'] for x in data_list]).astype('float32')
            tgt_images = np.array([x['tgt_image'] for x in data_list]).astype('float32')
            ref_labels = np.array([x['ref_label'] for x in data_list]).astype('float32')
            ref_images = np.array([x['ref_image'] for x in data_list]).astype('float32')
            flow_gt = np.array([x['flow_gt'] for x in data_list]).astype('float32')
            tgt_labels = to_variable(tgt_labels)
            tgt_images = to_variable(tgt_images)
            ref_labels = to_variable(ref_labels)
            ref_images = to_variable(ref_images)
            flow_gt = to_variable(flow_gt)
            data_list = {'tgt_labels': tgt_labels, 'tgt_images': tgt_images, 
                'ref_labels': ref_labels, 'ref_images': ref_images, 'flow_gt': flow_gt}
            
            yield step, data_list
    

    def forward_generator(self, data_list, prevs, t):
        tgt_labels, tgt_images, ref_labels, ref_images, flow_gt = self.get_data_t(data_list, t)

        loss_list, generated_images, prevs = \
            self.model.forward_generator(tgt_labels, tgt_images, ref_labels, ref_images,
            prevs=prevs, flow_gt=flow_gt)

        prevs = [to_variable(p.numpy()) for p in prevs]
        return loss_list, generated_images, prevs
    

    def forward_discriminator(self, data_list, prevs, t):
        tgt_labels, tgt_images, ref_labels, ref_images, flow_gt = self.get_data_t(data_list, t)

        loss_list, prevs = self.model.forward_discriminator(tgt_labels, tgt_images, ref_labels, ref_images, prevs)

        prevs = [to_variable(p.numpy()) for p in prevs]
        return loss_list, prevs
    

    def backward_generator(self, loss_list):
        loss_G_GAN, loss_GT_GAN, loss_F_Flow, loss_F_Warp, loss_F_Mask, loss_VGG = loss_list
        loss = loss_G_GAN + loss_GT_GAN + loss_F_Flow + loss_F_Warp + loss_F_Mask + loss_VGG
        # print(loss_G_GAN)
        # print(loss_GT_GAN)
        # print(loss_F_Flow)
        # print(loss_F_Warp)
        # print(loss_F_Mask)
        # print(loss_VGG)
        
        # loss = loss_F_Flow
        loss.backward()
        self.optimizer_G.minimize(loss)
        self.model.netG.clear_gradients()
    

    def backward_discriminator(self, loss_list):
        [loss_real_temp, loss_fake_temp], [loss_real_indv, loss_fake_indv] = loss_list
        loss = loss_real_temp + loss_fake_temp
        loss_t = loss_real_indv + loss_fake_indv
        loss.backward()
        loss_t.backward()
        self.optimizer_D.minimize(loss)
        self.optimizer_DT.minimize(loss_t)
        self.model.netD.clear_gradients()
        self.model.netDT.clear_gradients()

    
    def update_logger(self, epoch, step, t, loss_G_list, loss_D_list):
        global_step = epoch * cfg.train.max_step * cfg.train.max_t + \
            step * cfg.train.max_t + t
        
        loss_G_GAN, loss_GT_GAN, loss_F_Flow, loss_F_Warp, loss_F_Mask, loss_VGG = loss_G_list
        [loss_real_temp, loss_fake_temp], [loss_real_indv, loss_fake_indv] = loss_D_list

        self.global_G_loss.update(loss_G_GAN.numpy()[0])
        self.global_GT_loss.update(loss_GT_GAN.numpy()[0])
        self.global_G_flow_loss.update(loss_F_Flow.numpy()[0])
        self.global_G_warp_loss.update(loss_F_Warp.numpy()[0])
        self.global_G_mask_loss.update(loss_F_Mask.numpy()[0])
        self.global_G_vgg_loss.update(loss_VGG.numpy()[0])

        self.global_D_real_loss.update(loss_real_temp.numpy()[0])
        self.global_D_fake_loss.update(loss_fake_temp.numpy()[0])
        self.global_DT_real_loss.update(loss_real_indv.numpy()[0])
        self.global_DT_fake_loss.update(loss_fake_indv.numpy()[0])
        
        save_dict = {
            'G_loss_avg': self.global_G_loss.avg,
            'GT_loss_avg': self.global_GT_loss.avg,
            'G_flow_loss_avg': self.global_G_flow_loss.avg,
            'G_warp_loss_avg': self.global_G_warp_loss.avg,
            'G_mask_loss_avg': self.global_G_mask_loss.avg,
            'G_VGG_loss_avg': self.global_G_vgg_loss.avg,
            'D_real_loss_avg': self.global_D_real_loss.avg,
            'D_fake_loss_avg': self.global_D_fake_loss.avg,
            'DT_real_loss_avg': self.global_DT_real_loss.avg,
            'DT_fake_loss_avg': self.global_DT_fake_loss.avg,
            'G_loss': loss_G_GAN.numpy()[0],
            'GT_loss': loss_GT_GAN.numpy()[0],
            'G_flow_loss': loss_F_Flow.numpy()[0],
            'G_warp_loss': loss_F_Warp.numpy()[0],
            'G_mask_loss': loss_F_Mask.numpy()[0],
            'G_VGG_loss': loss_VGG.numpy()[0],
            'DT_real_loss': loss_real_temp.numpy()[0],
            'DT_fake_loss': loss_fake_temp.numpy()[0],
            'D_real_loss': loss_real_indv.numpy()[0],
            'D_fake_loss': loss_fake_indv.numpy()[0]
        }
        self.logger.write_scalars(save_dict, tag='train', n_iter=global_step)

        print(("Epoch: {}, Step: {}, ").format(epoch, global_step), end='')
        for k, v in save_dict.items():
            print((k + ': {:.2f}, ').format(v), end='')
        print('Speed: {:.2f} step / second'.format(1 / self.timer.elapsed_time()))
        self.timer.restart()


    def save_models(self, ):
        save_path = os.path.join(cfg.train.save_dir, cfg.exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fluid.save_dygraph(self.model.netG.state_dict(), os.path.join(save_path, 'G'))
        fluid.save_dygraph(self.model.netD.state_dict(), os.path.join(save_path, 'D'))
        fluid.save_dygraph(self.model.netDT.state_dict(), os.path.join(save_path, 'DT'))
        fluid.save_dygraph(self.optimizer_G.state_dict(), os.path.join(save_path, 'G'))
        fluid.save_dygraph(self.optimizer_D.state_dict(), os.path.join(save_path, 'D'))
        fluid.save_dygraph(self.optimizer_DT.state_dict(), os.path.join(save_path, 'DT'))


    def load_models(self,):
        save_path = os.path.join(cfg.train.save_dir, cfg.exp_name)
        if not os.path.exists(os.path.join(save_path) + 'G'):
            print("Pretrained models not found, train from scrach!")
            return

        param_dict, opt_dict = fluid.dygraph.load_dygraph(os.path.join(save_path, 'G'))
        self.model.netG.load_dict(param_dict)
        self.optimizer_G.load_dict(opt_dict)
        param_dict, opt_dict = fluid.dygraph.load_dygraph(os.path.join(save_path, 'D'))
        self.model.netD.load_dict(param_dict)
        self.optimizer_D.load_dict(opt_dict)
        param_dict, opt_dict = fluid.dygraph.load_dygraph(os.path.join(save_path, 'DT'))
        self.model.netDT.load_dict(param_dict)
        self.optimizer_DT.load_dict(opt_dict)
    

    def visualize(self, data_list, generated_images):
        tgt_labels, tgt_images, flow_gt = data_list['tgt_labels'], data_list['tgt_images'], data_list['flow_gt']
        fake_images, fake_raw_images, warped_images, flows, flow_masks, attn_vis = generated_images
        b, n, _, _, _ = tgt_labels.shape
        # print(tgt_labels.shape)
        # print(tgt_images.shape)
        # print(flow_gt.shape)
        # print(fake_images.shape)
        # print(fake_raw_images.shape)
        # print(flows.shape)
        # print(flow_masks.shape)
        for i in range(b):
            save_dir = os.path.join(cfg.train.vis_dir, 'train_results', str(i), 'image')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for j in range(n - 1):
                (tgt_image, fake_image, fake_raw_image, warped_image) = \
                tgt_images[i][j + 1], fake_images[i][j + 1], fake_raw_images[i][j + 1], warped_images[i][j]
                save_image(j, save_dir, tgt_image, fake_image, fake_raw_image, warped_image)

            save_dir = os.path.join(cfg.train.vis_dir, 'train_results', str(i), 'flow')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for j in range(n - 1):
                flow_g, flow, flow_mask, tgt_label = \
                    flow_gt[i][j], flows[i][j], flow_masks[i][j], tgt_labels[i][j + 1]
                save_flow_image(j, save_dir, flow_g, flow, flow_mask, tgt_label)


    def get_data_t(self, data_list, t):
        tgt_labels = data_list['tgt_labels'][:, t : t + cfg.model.n_frames]
        tgt_images = data_list['tgt_images'][:, t : t + cfg.model.n_frames]
        ref_labels = data_list['ref_labels']
        ref_images = data_list['ref_images']
        if t == 0:
            flow_gt = data_list['flow_gt'][:, 0 : cfg.model.n_frames - 1]
        else:
            flow_gt = data_list['flow_gt'][:, t - 1: t - 1 + cfg.model.n_frames]

        return tgt_labels, tgt_images, ref_labels, ref_images, flow_gt
