from __future__ import print_function
from __future__ import unicode_literals
import sys 
sys.path.append('/home/aistudio')

from vid2vid.utils.collect import Config


cfg = Config()

cfg.exp_name = 'test_street'

cfg.reader.max_t_step = 4
cfg.reader.dataset_mode = 'pose'
cfg.reader.n_shot = 2 # reference number
cfg.reader.load_size = 256
cfg.reader.fine_size = 512
cfg.reader.aspect_ratio = 2
cfg.reader.resize = False
cfg.reader.scale_width = False
cfg.reader.random_scale = True
cfg.reader.crop = True
cfg.reader.ref_img_id = "0 0 0"
cfg.reader.eval_n_frames_total = 4

cfg.dataset.street.n_shot = 2
cfg.dataset.street.n_frames_total = 8
cfg.dataset.street.data_dir = '/home/aistudio/vid2vid/data/dataset/street'
cfg.dataset.street.seq_path = '/home/aistudio/vid2vid/data/dataset/street/test_images/01'
cfg.dataset.street.ref_img_path = '/home/aistudio/vid2vid/data/dataset/street/test_images/02'
cfg.dataset.street.is_train = True
cfg.dataset.street.fine_size = 512
cfg.dataset.street.aspect_ratio = 2
cfg.dataset.street.num_threads = 4
cfg.dataset.street.buf_size = 1024

cfg.dataset.pose.n_shot = 2
cfg.dataset.pose.n_frames_total = 4
cfg.dataset.pose.data_dir = '/home/aistudio/vid2vid/data/dataset/pose'
cfg.dataset.pose.seq_path = '/home/aistudio/vid2vid/data/dataset/pose/test_images/01'
cfg.dataset.pose.ref_img_path = '/home/aistudio/vid2vid/data/dataset/pose/test_images/02'
cfg.dataset.pose.is_train = True
cfg.dataset.pose.fine_size = 256
cfg.dataset.pose.aspect_ratio = 0.5
cfg.dataset.pose.basic_point_only = True
cfg.dataset.pose.remove_face_labels = True
cfg.dataset.pose.num_threads = 4
cfg.dataset.pose.buf_size = 1024

cfg.dataset.face.n_shot = 2
cfg.dataset.face.n_frames_total = 4
cfg.dataset.face.data_dir = '/home/aistudio/vid2vid/data/dataset/face'
cfg.dataset.face.seq_path = '/home/aistudio/vid2vid/data/dataset/face/test_images/0001'
cfg.dataset.face.ref_img_path = '/home/aistudio/vid2vid/data/dataset/face/test_images/0002'
cfg.dataset.face.is_train = True
cfg.dataset.face.fine_size = 256
cfg.dataset.face.aspect_ratio = 1
cfg.dataset.face.no_upper_face = True
cfg.dataset.face.how_many = 300
cfg.dataset.face.num_threads = 4
cfg.dataset.face.buf_size = 1024

cfg.model.n_frames = 8
cfg.model.prev_frames_n = 2
cfg.model.use_true_prev = False

cfg.model.generator.ngf = 32
cfg.model.generator.image_channel = 3
cfg.model.generator.label_channel = 1
cfg.model.generator.generator_norm_type = 'batch'
cfg.model.generator.reference_n = 2
cfg.model.generator.prev_n = 2

cfg.model.discriminator.nc = 3
cfg.model.discriminator.ndf = 64
cfg.model.discriminator.n_layers = 3
cfg.model.discriminator.stride = 2
cfg.model.discriminator.subarch = 'n_layers'
cfg.model.discriminator.num_D = 3
cfg.model.discriminator.get_intern_feat = False

cfg.model.temporal_discriminator.nc = 3 * 8
cfg.model.temporal_discriminator.ndf = 64
cfg.model.temporal_discriminator.n_layers = 3
cfg.model.temporal_discriminator.stride = 2
cfg.model.temporal_discriminator.subarch = 'n_layers'
cfg.model.temporal_discriminator.num_D = 3
cfg.model.temporal_discriminator.get_intern_feat = False

cfg.model.loss.lambda_mask = 10
cfg.model.loss.lambda_vgg = 10
cfg.model.loss.lambda_flow = 10

cfg.train.start_epoch = 0
cfg.train.max_epoch = 100
cfg.train.vis_interval = 4
cfg.train.save_interval = 4
cfg.train.beta1=0.5
cfg.train.beta2=0.999
cfg.train.G.lr = 0.0002
cfg.train.D.lr = 0.0002
cfg.train.batch_size = 1
cfg.train.log_dir = '/home/aistudio/vid2vid/logs'
cfg.train.save_dir = '/home/aistudio/vid2vid/saved_models'
cfg.train.vis_dir = '/home/aistudio/vid2vid/visualize'
cfg.train.max_step = 1
cfg.train.max_t = 1


