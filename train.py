import os
import sys
sys.path.append('/home/aistudio')

import paddle
import paddle.fluid as fluid

from vid2vid.readers.street_dataset import StreetReader
# from vid2vid.models.solver import Solver
from vid2vid.trainer import Trainer
from vid2vid.configs.fewshot_config import cfg


def train():
    with fluid.dygraph.guard():
        batch_reader = StreetReader(**cfg.dataset.street).batch_reader(cfg.train.batch_size)
        trainer = Trainer(batch_reader)
        trainer.load_models()
        for epoch in range(cfg.train.start_epoch, cfg.train.max_epoch):
            for step, data_list in trainer.read_data():
                prevs_d = prevs_g = [None] * 3
                for t in range(0, cfg.dataset.street.n_frames_total, cfg.model.n_frames):
                    loss_d_list, prevs_d = trainer.forward_discriminator(data_list, prevs_d, t)
                    trainer.backward_discriminator(loss_d_list)

                    loss_g_list, generated_images, prevs_g = trainer.forward_generator(data_list, prevs_g, t)
                    trainer.backward_generator(loss_g_list)

                    trainer.update_logger(epoch, step, t, loss_g_list, loss_d_list)

            if epoch % cfg.train.vis_interval == 0:
                trainer.visualize(data_list, generated_images)
            if epoch % cfg.train.save_interval == 0:
                trainer.save_models()


if __name__ == '__main__':
    train()