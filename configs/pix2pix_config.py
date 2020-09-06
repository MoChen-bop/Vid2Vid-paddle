from __future__ import print_function
from __future__ import unicode_literals
import sys 
sys.path.append('/home/aistudio')

from vid2vid.utils.collect import Config


cfg = Config()

cfg.model.ref_encoder.image_channel = 3
cfg.model.ref_encoder.label_channel = 1
cfg.model.ref_encoder.nf = 64
cfg.model.ref_encoder.reference_n = 4

cfg.model.generator.ngf = 64
cfg.model.generator.semantic_nc = 1
cfg.model.generator.norm_type = 'instance'
cfg.model.generator.use_spectral_norm = True

