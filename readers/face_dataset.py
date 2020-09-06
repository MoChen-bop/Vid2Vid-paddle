import os
import os.path as path
import sys
sys.path.append('/home/aistudio')
import random
import numpy as np 
import json
from PIL import Image
import cv2

import paddle
import paddle.fluid as fluid


from vid2vid.readers.dataset_utils import make_dataset, check_path_valid
from vid2vid.readers.dataset_utils import make_grouped_dataset
from vid2vid.readers.dataset_utils import get_video_params, get_img_params
from vid2vid.readers.keypoint2img import interp_points, draw_edge
from vid2vid.readers.augmentation import BaseTransform, Augmentation


class FaceDataset():

    def __init__(self, 
        n_shot=4,
        n_frames_total=16,
        fine_size=256,
        aspect_ratio=2,
        data_dir='/home/aistudio/vid2vid/data/dataset/face', 
        seq_path='/home/aistudio/vid2vid/data/dataset/face/test_images/0001',
        ref_img_path='/home/aistudio/vid2vid/data/dataset/face/test_images/0002',
        is_train=True,
        no_upper_face=True,
        how_many=300,
        num_threads=4,
        buf_size=1024):

        self.n_shot = n_shot
        self.n_frames_total = n_frames_total
        self.is_train = is_train
        self.fine_size = fine_size
        self.aspect_ratio = aspect_ratio
        self.how_many = how_many
        self.num_threads = num_threads
        self.buf_size = buf_size

        if is_train:
            self.I_paths = sorted(make_grouped_dataset(path.join(data_dir, 'train_images')))
            self.L_paths = sorted(make_grouped_dataset(path.join(data_dir, 'train_keypoints')))
            check_path_valid(self.L_paths, self.I_paths)

            self.transform_I = Augmentation(method=Image.BICUBIC, is_label=False)
            self.transform_L = Augmentation(method=Image.NEAREST, is_label=True)
        else:
            self.I_paths = sorted(make_dataset(seq_path))
            self.L_paths = sorted(make_dataset(seq_path.replace('images', 'keypoints')))

            self.ref_I_paths = sorted(make_dataset(ref_img_path))
            self.ref_L_paths = sorted(make_dataset(ref_img_path.replace('images', 'keypoints')))
            
            self.transform_I = BaseTransform(method=Image.BICUBIC, is_label=False)
            self.transform_L = BaseTransform(method=Image.NEAREST, is_label=True)
        
        self.n_of_seqs = len(self.I_paths) if self.is_train else 1

        self.add_upper_face = not no_upper_face
        self.part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if self.add_upper_face else [])],
            [range(17, 22)],
            [range(22, 27)],
            [[28, 31], range(31, 36), [35, 28]],
            [[36, 37, 38, 39], [39, 40, 41, 36]],
            [[42, 43, 44, 45], [45, 46, 47, 42]],
            [range(48, 55), [54, 55, 56, 57, 58, 59, 48], range(60, 65), [64, 65, 66, 67, 60]],
            ]
        self.ref_dist_x, self.ref_dist_y = [None] * 83, [None] * 83
        self.dist_scale_x, self.dist_scale_y = [None] * 83, [None] * 83
        self.fix_crop_pos = True
        self.L = self.I = self.Lr = self.Ir = self.F = None


    def create_reader(self, ):
        def reader():
            for count in range(self.n_of_seqs):
                yield count
        
        return paddle.reader.xmap_readers(self.load_frames, reader, self.num_threads, self.buf_size)


    def batch_reader(self, batch_size):
        reader = self.create_reader()

        def _batch_reader():
            batch_out = []
            for data_list in reader():
                if data_list is None:
                    continue
                batch_out.append(data_list)
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
        return _batch_reader


    def load_frames(self, count):

        if self.is_train:
            seq_idx = random.randrange(self.n_of_seqs)
            L_paths = self.L_paths[seq_idx]
            I_paths = self.I_paths[seq_idx]
            ref_L_paths, ref_I_paths = L_paths, I_paths
        else:
            L_paths, I_paths = self.L_paths, self.I_paths
            ref_L_paths, ref_I_paths = self.ref_L_paths, self.ref_I_paths
        
        n_frames_total, start_idx, t_step, ref_indices = get_video_params(self.n_frames_total, len(I_paths), count)
        w, h = self.fine_size, int(self.fine_size / self.aspect_ratio)
        img_params = get_img_params((w, h), is_train=self.is_train)
        is_first_frame = self.is_train or count == 0

        Lr, Ir = self.Lr, self.Ir
        if is_first_frame:
            keypoints = np.loadtxt(ref_L_paths[ref_indices[0]], delimiter=',')
            ref_crop_coords = self.get_crop_coords(keypoints, for_ref=True)
            self.bw = max(1, (ref_crop_coords[1] - ref_crop_coords[0]) // 256)

            ref_L_paths = [ref_L_paths[idx] for idx in ref_indices]
            all_keypoints = self.read_all_keypoints(ref_L_paths, ref_crop_coords, is_ref=True)

            for i, idx in enumerate(ref_indices):
                keypoints = all_keypoints[i]
                ref_img = self.crop(Image.open(ref_I_paths[idx]), ref_crop_coords)
                Li = self.get_face_image(keypoints, img_params, ref_img.size)
                Ii = self.transform_I(ref_img, img_params)
                Lr = self.concat_frame(Lr, Li[np.newaxis,:])
                Ir = self.concat_frame(Ir, Ii[np.newaxis,:])
            
            if not self.is_train:
                self.Lr, self.Ir = Lr, Ir
        
        if is_first_frame:
            keypoints = np.loadtxt(L_paths[start_idx], delimiter=',')
            crop_coords = self.get_crop_coords(keypoints)
            if not self.is_train:
                if self.fix_crop_pos:
                    self.crop_coords = crop_coords
                else:
                    self.crop_size = crop_coords[1] - crop_coords[0], crop_coords[3] - crop_coords[2]
            
            self.bw = max(1, (crop_coords[1] - crop_coords[0]) // 256)
            end_idx = (start_idx + n_frames_total * t_step) if self.is_train else (start_idx + self.how_many)
            L_paths = L_paths[start_idx : end_idx : t_step]
            all_keypoints = self.read_all_keypoints(L_paths, crop_coords if self.fix_crop_pos else None, is_ref=False)
            if not self.is_train:
                self.all_keypoints = all_keypoints
        else:
            if self.fix_crop_pos:
                crop_coords = self.crop_coords
            else:
                keypoints = np.loadtxt(L_paths[start_idx], delimiter=',')
                crop_coords = self.get_crop_coords(keypoints, self.crop_size)
            all_keypoints = self.all_keypoints
        
        L, I = self.L, self.I
        for t in range(n_frames_total):
            ti = t if self.is_train else start_idx + t
            keypoints = all_keypoints[ti]
            I_path = I_paths[start_idx + t * t_step]
            img = self.crop(Image.open(I_path), crop_coords)
            Lt = self.get_face_image(keypoints, img_params, img.size)
            It = self.transform_I(img, img_params)
            L = self.concat_frame(L, Lt[np.newaxis,:])
            I = self.concat_frame(I, It[np.newaxis,:])

        F = self.F
        if self.is_train:
            for t in range(1, n_frames_total):
                idx = start_idx + t * t_step
                f = self.get_flow_tensor(I_paths[idx - t_step], I_paths[idx], crop_coords, img_params)
                F = self.concat_frame(F, f[np.newaxis,:])
        
        if not self.is_train:
            self.L, self.I = L, I
        return_list = {'tgt_label': L, 'tgt_image': I, 
            'ref_label': Lr, 'ref_image': Ir, 'flow_gt': F}

        return return_list
    

    def read_all_keypoints(self, L_paths, crop_coords, is_ref):
        all_keypoints = [self.read_keypoints(L_path, crop_coords) for L_path in L_paths]
        if not self.is_train or self.n_frames_total > 4:
            self.normalize_faces(all_keypoints, is_ref=is_ref)
        
        return all_keypoints
    

    def read_keypoints(self, L_path, crop_coords):
        keypoints = np.loadtxt(L_path, delimiter=',')

        if crop_coords is None:
            crop_coords = self.get_crop_coords(keypoints)
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]

        if self.add_upper_face:
            pts = keypoints[:17, :].astype(np.int32)
            baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
            upper_pts = pts[1: -1, :].copy()
            upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
            keypoints = np.vstack((keypoints, upper_pts[::-1, :]))
        
        return keypoints
    

    def get_crop_coords(self, keypoints, crop_size=None, for_ref=False):
        min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
        min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
        w = h = (max_x - min_x)
        if crop_size is not None:
            h, w = crop_size[0] / 2, crop_size[1] / 2
        if self.is_train and self.fix_crop_pos:
            offset_max = 0.2
            offset = [random.uniform(-offset_max, offset_max), 
                      random.uniform(-offset_max, offset_max)]
            if for_ref:
                scale_max = 0.2
                self.scale = [random.uniform(1 - scale_max, 1 + scale_max), 
                              random.uniform(1 - scale_max, 1 + scale_max)]
            w *= self.scale[0]
            h *= self.scale[1]
            x_cen += int(offset[0]*w)
            y_cen += int(offset[1]*h)
                        
        min_x = x_cen - w
        min_y = y_cen - h*1.25
        max_x = min_x + w*2        
        max_y = min_y + h*2

        return int(min_y), int(max_y), int(min_x), int(max_x)
    

    def normalize_faces(self, all_keypoints, is_ref=False):        
        central_keypoints = [8]
        face_centers = [np.mean(keypoints[central_keypoints,:], axis=0) for keypoints in all_keypoints]        
        compute_mean = not is_ref
        if compute_mean:
            if self.opt.isTrain:
                img_scale = 1
            else:
                img_scale = self.img_scale / (all_keypoints[0][:,0].max() - all_keypoints[0][:,0].min())

        part_list = [[0,16], [1,15], [2,14], [3,13], [4,12], [5,11], [6,10], [7,9, 8], # face 17
                     [17,26], [18,25], [19,24], [20,23], [21,22], # eyebrows 10
                     [27], [28], [29], [30], [31,35], [32,34], [33], # nose 9
                     [36,45], [37,44], [38,43], [39,42], [40,47], [41,46], # eyes 12
                     [48,54], [49,53], [50,52], [51], [55,59], [56,58], [57], # mouth 12
                     [60,64], [61,63], [62], [65,67], [66], # tongue 8                     
                    ]
        if self.add_upper_face:
            part_list += [[68,82], [69,81], [70,80], [71,79], [72,78], [73,77], [74,76, 75]] # upper face 15

        for i, pts_idx in enumerate(part_list):            
            if compute_mean or is_ref:                
                mean_dists_x, mean_dists_y = [], []
                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]                    
                    for p, pt in enumerate(pts):                        
                        mean_dists_x.append(np.linalg.norm(pt - pts_cen))                        
                        mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
                mean_dist_x = sum(mean_dists_x) / len(mean_dists_x) + 1e-3                
                mean_dist_y = sum(mean_dists_y) / len(mean_dists_y) + 1e-3                
            if is_ref:
                self.ref_dist_x[i] = mean_dist_x
                self.ref_dist_y[i] = mean_dist_y
                self.img_scale = all_keypoints[0][:,0].max() - all_keypoints[0][:,0].min()
            else:
                if compute_mean:                    
                    self.dist_scale_x[i] = self.ref_dist_x[i] / mean_dist_x / img_scale
                    self.dist_scale_y[i] = self.ref_dist_y[i] / mean_dist_y / img_scale                    

                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]                    
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]                    
                    pts = (pts - pts_cen) * self.dist_scale_x[i] + (pts_cen - face_cen) * self.dist_scale_y[i] + face_cen                    
                    all_keypoints[k][pts_idx] = pts


    def get_face_image(self, keypoints, img_patams, size):   
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interp_points(x, y) # interp keypoints to get the curve shape
                    draw_edge(im_edges, curve_x, curve_y, bw=self.bw)
        input_tensor = self.transform_I(Image.fromarray(im_edges), img_patams)
        return input_tensor


    def get_flow_tensor(self, prev_img_path, cur_img_path, crop_coords, img_params):
        prev_gray = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
        curr_gray = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 30, 3, 5, 1.2, 0)
        flow = (flow + 30) * (255.0 / (2 * 30))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow < 0] = 0
        flow = Image.fromarray(np.array(flow).astype('uint8'))
        flow = self.crop(flow, crop_coords)
        flow_tensor = self.transform_L(flow, img_params)
        flow_tensor = flow_tensor / 255.0 * (2 * 30) - 30
        flow_tensor = flow_tensor.transpose((2, 0, 1))
        return flow_tensor


    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))


    def __len__(self):
        return self.n_of_seqs


    def concat_frame(self, A, Ai, n=100):
        if A is None or Ai.shape[0] >= n:
            return Ai[-n:]
        else:
            return np.concatenate([A, Ai])[-n:]


if __name__ == '__main__':
    from vid2vid.configs.fewshot_config import cfg
    from vid2vid.utils.visualize import visualize_dataset_image
    from vid2vid.utils.visualize import visualize_dataset_label
    from vid2vid.utils.visualize import visualize_dataset_flow


    reader = FaceDataset(**cfg.dataset.face).create_reader()

    for i, data_list in enumerate(reader()):
        print(data_list['tgt_label'].shape)
        print(data_list['tgt_image'].shape)
        print(data_list['ref_label'].shape)
        print(data_list['ref_image'].shape)
        print(data_list['flow_gt'].shape)

        save_root = '/home/aistudio/vid2vid/visualize/dataset/face/' + str(i)
        save_dir = os.path.join(save_root, 'tgt_images')
        visualize_dataset_image(data_list['tgt_image'], save_dir)
        
        save_dir = os.path.join(save_root, 'ref_images')
        visualize_dataset_image(data_list['ref_image'], save_dir)

        save_dir = os.path.join(save_root, 'tgt_labels')
        visualize_dataset_label(data_list['tgt_label'], save_dir)
        
        save_dir = os.path.join(save_root, 'ref_labels')
        visualize_dataset_label(data_list['ref_label'], save_dir)
        
        save_dir = os.path.join(save_root, 'flow_gts')
        visualize_dataset_flow(data_list['flow_gt'], save_dir)

        if i > 4:
            break









