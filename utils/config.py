from __future__ import print_function
from __future__ import unicode_literals
import sys 
sys.path.append('/home/aistudio')

from vid2vid.utils.collect import Config


cfg = Config()

cfg.EXP.NAME = 'train_street'

cfg.READER.NUM_THREADS = 8
cfg.READER.BUF_SIZE = 1024 * 8
cfg.READER.N_FRAMES_TOTAL = 10

cfg.READER.AUG.LOAD_SIZE = 256
cfg.READER.AUG.FINE_SIZE = 256
cfg.READER.AUG.ASPECT_RATIO = 2
cfg.READER.AUG.RESIZE = False
cfg.READER.AUG.CROP = False
cfg.READER.AUG.SCALE_WIDTH = True
cfg.READER.AUG.RANDOM_SCALE = False
cfg.READER.AUG.CROP = False


cfg.DATASET.LABEL_NC = 20
cfg.DATASET.IMAGE_NC = 3
cfg.DATASET.LABEL_CHANNEL = 1
cfg.DATASET.N_REFERENCE = 2

cfg.DATASET.STREET.DATA_DIR = '/home/aistudio/vid2vid/data/dataset/street'
cfg.DATASET.FACE.DATA_DIR = '/home/aistudio/vid2vid/data/dataset/face'
cfg.DATASET.POSE.DATA_DIR = '/home/aistudio/vid2vid/data/dataset/pose'
cfg.DATASET.STREET.INFER_SEQ_PATH = '/home/aistudio/vid2vid/data/dataset/street/test_images/01/'
cfg.DATASET.STREET.INFER_REF_IMAGE_PATH = '/home/aistudio/vid2vid/data/dataset/street/test_images/02/'


cfg.MODELS.GENERATOR.N_DOWNSAMPLE_G = 5
cfg.MODELS.GENERATOR.N_DOWNSAMPLE_A = 5
cfg.MODELS.GENERATOR.N_DOWNSAMPLE_F = 5
cfg.MODELS.GENERATOR.N_ADAPTIVE_LAYERS = 5
cfg.MODELS.GENERATOR.N_FRAME = 2

cfg.MODELS.GENERATOR.NGF = 32

cfg.MODEL.DISCRIMINATOR.NETD_INPUT_NC = cfg.DATASET.IMAGE_NC
cfg.MODEL.DISCRIMINATOR.NETDT_INPUT_NC = 10 * 3

cfg.TRAIN.N_FRAMES = 10
cfg.TRAIN.N_TOTAL_FRAMES = 10
cfg.TRAIN.LAMBDA_FEAT = 10.0
cfg.TRAIN.LAMBDA_FLOW = 10.0
cfg.TRAIN.LAMBDA_MASK = 10.0
cfg.TRAIN.LOG_DIR = '/home/aistudio/vid2vid/logs'
cfg.TRAIN.SAVE_DIR = '/home/aistudio/vid2vid/saved_models'
cfg.TRAIN.VIS_DIR = '/home/aistudio/vid2vid/vis'
cfg.TRAIN.MAX_STEP = 4
cfg.TRAIN.MAX_T = 1
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.MAX_EPOCH = 200
cfg.TRAIN.VIS_INTERVAL = 2
cfg.TRAIN.SAVE_INTERVAL = 2

cfg.SOLVER.BATCH_SIZE = 1
cfg.SOLVER.TRAIN = True
cfg.SOLVER.MAX_T_STEP = 4
cfg.SOLVER.N_SHOT = 2
cfg.SOLVER.DATASET_MODE = 'street'
cfg.SOLVER.BETA1=0.5
cfg.SOLVER.BETA2=0.999
cfg.SOLVER.G.LR = 0.0004
cfg.SOLVER.D.LR = 0.0001

def update_config():
    pass