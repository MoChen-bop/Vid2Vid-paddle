import os
import sys
sys.path.append('/home/aistudio')
from PIL import Image
import random
import numpy as np

from vid2vid.configs.fewshot_config import cfg


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp',
    '.txt', '.json',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images
    
    if recursive:
        make_dataset_rec(dir, images)
    else:
        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    
    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)
    
    return images


def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))

        if len(paths) > 0:
            images.append(paths)
    return images


def check_path_valid(A_paths, B_paths):
    if len(A_paths) != len(B_paths):
        print("%s not equal to %s" % (A_paths[0], B_paths[0]))
    assert(len(A_paths) == len(B_paths))

    if isinstance(A_paths[0], list):
        for a, b in zip(A_paths, B_paths):
            if len(a) != len(b):
                print('%s not equal to %s' % (a[0], b[0]))
            assert(len(a) == len(b))


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_video_params(n_frames_total, cur_seq_len, index, is_train=True):
    if is_train:
        n_frames_total = min(cur_seq_len, n_frames_total)
        max_t_step = min(cfg.reader.max_t_step, 
            (cur_seq_len - 1) // max(1, (n_frames_total - 1)))
        t_step = random.randrange(max_t_step) + 1

        offset_max = max(1, cur_seq_len - (n_frames_total - 1) * t_step)
        if 'pose' in cfg.reader.dataset_mode:
            start_idx = index % offset_max
            max_range, min_range = 60, 14
        else:
            start_idx = random.randrange(offset_max)
            max_range, min_range = 300, 14
        
        ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
            + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
        ref_indices = random.sample(ref_range, cfg.reader.n_shot)

    else:
        n_frames_total = cfg.reader.eval_n_frames_total
        start_idx = index
        t_step = 1
        ref_indices = cfg.reader.ref_img_id.split(' ')
        ref_indices = [int(i) for i in ref_indices]

    return n_frames_total, start_idx, t_step, ref_indices


def get_img_params(size, is_train=True):
    w, h = size
    new_w, new_h = w, h

    if cfg.reader.resize:
        new_h = new_w = cfg.reader.load_size
    elif cfg.reader.scale_width:
        new_w = cfg.reader.load_size
        new_h = int(new_w * h) // w
    elif cfg.reader.random_scale:
        new_w = random.randrange(int(cfg.reader.fine_size), int(1.2 * cfg.reader.fine_size))
        new_h = int(new_w * h) // w

    if not cfg.reader.crop:
        new_h = int(new_w // cfg.reader.aspect_ratio)

    new_w = new_w // 4 * 4  
    new_h = new_h // 4 * 4

    size_x = min(cfg.reader.load_size, cfg.reader.fine_size)
    size_y = size_x // cfg.reader.aspect_ratio

    if not is_train:
        pos_x = (new_w - size_x) // 2
        pos_y = (new_h - size_y) // 2
    else:
        pos_x = random.randrange(np.maximum(1, new_w - size_x))
        pos_y = random.randrange(np.maximum(1, new_h - size_y))
    
    h_b = random.uniform(-30, 30)
    s_a = random.uniform(0.8, 1.2)
    s_b = random.uniform(-10, 10)
    v_a = random.uniform(0.8, 1.2)
    v_b = random.uniform(-10, 10)

    flip = random.random() > 0.5
    return {'new_size': (new_w, new_h), 'crop_pos': (pos_x, pos_y), 
        'crop_size': (size_x, size_y), 'flip': flip, 
        'color_aug': (h_b, s_a, s_b, v_a, v_b),
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}

