import os
import sys
sys.path.append('/home/aistudio')

import numpy as np
import cv2


def save_image(index, save_dir, tgt_image, fake_image, fake_raw_image, warped_image):
    tgt_image = tgt_image.numpy()
    fake_image = fake_image.numpy()
    fake_raw_image = fake_raw_image.numpy()
    warped_image = warped_image.numpy()

    tgt_image = tgt_image.transpose((1, 2, 0))
    tgt_image = tgt_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    tgt_image = (tgt_image * 255).astype(np.uint8)
    tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_RGB2BGR)
    
    fake_image = fake_image.transpose((1, 2, 0))
    fake_image = fake_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    fake_image = (fake_image * 255).astype(np.uint8)
    fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
    
    fake_raw_image = fake_raw_image.transpose((1, 2, 0))
    fake_raw_image = fake_raw_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    fake_raw_image = (fake_raw_image * 255).astype(np.uint8)
    fake_raw_image = cv2.cvtColor(fake_raw_image, cv2.COLOR_RGB2BGR)
    
    warped_image = warped_image.transpose((1, 2, 0))
    warped_image = warped_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    warped_image = (warped_image * 255).astype(np.uint8)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)

    show = np.concatenate([
        np.concatenate([tgt_image, fake_image], axis=1),
        np.concatenate([fake_raw_image, warped_image], axis=1)
    ], axis=0)
    path = os.path.join(save_dir, '{}.png'.format(index))
    cv2.imwrite(path, show)


def save_flow_image(index, save_dir, flow_gt, flow, flow_mask, tgt_label):
    flow_gt = flow_gt.numpy()
    flow = flow.numpy()
    flow_mask = flow_mask.numpy()
    tgt_label = tgt_label.numpy()

    flow_gt_u = flow_gt[0]
    flow_gt_v = flow_gt[1]
    flow_u = flow[0]
    flow_v = flow[1]
    flow_mask = flow_mask[0]
    tgt_label = tgt_label[0]

    flow_gt_u = (flow_gt_u + 30) * 255.0 / (30 * 2)
    flow_gt_v = (flow_gt_v + 30) * 255.0 / (30 * 2)
    flow_u = (flow_u + 30) * 255.0 / (30 * 2)
    flow_v = (flow_v + 30) * 255.0 / (30 * 2)
    tgt_label = tgt_label / 255.0 * 50
    flow_mask = flow_mask * 255.0

    flow_gt_u = flow_gt_u.astype('uint8')
    flow_gt_v = flow_gt_v.astype('uint8')
    flow_u = flow_u.astype('uint8')
    flow_v = flow_v.astype('uint8')
    flow_mask = flow_mask.astype('uint8')
    tgt_label = tgt_label.astype('uint8')

    show = np.concatenate(
        [
            np.concatenate([flow_gt_u, flow_u], axis=1),
            np.concatenate([flow_gt_v, flow_v], axis=1),
            np.concatenate([tgt_label, flow_mask], axis=1),
        ],
        axis=0
    )
    path = os.path.join(save_dir, '{}.png'.format(index))
    cv2.imwrite(path, show)


def save_attention_image(attn_vis, save_dir):
    pass


def visualize_dataset_image(images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n, c, h, w = images.shape

    for i in range(n):
        show = np.transpose(images[i], (1, 2, 0))
        show = show * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        show = (show * 255).astype(np.uint8)
        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        path = os.path.join(save_dir, '{}.png'.format(i))
        cv2.imwrite(path, show)


def visualize_dataset_label(labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n, c, h, w = labels.shape

    for i in range(n):
        if c == 1:
            show = labels[i][0] / 255.0 * 50
            show = show.astype(np.uint8)
            path = os.path.join(save_dir, '{}.png'.format(i))
            cv2.imwrite(path, show)
        elif c == 6:
            show_O = labels[i, :3]
            show_O = np.transpose(show_O, (1, 2, 0))
            show_O = show_O.astype(np.uint8)
            path = os.path.join(save_dir, '{}_O.png'.format(i))
            cv2.imwrite(path, show_O)
            show_D = labels[i, 3:]
            show_D = np.transpose(show_D, (1, 2, 0))
            show_D[:,:,2] = ((show_D[:,:,2] * 0.5 + 0.5) * 24 / 255 - 0.5) / 0.5
            show_D = show_D.astype(np.uint8)
            path = os.path.join(save_dir, '{}_D.png'.format(i))
            cv2.imwrite(path, show_D)
        elif c == 3:
            show = labels[i]
            show = show.transpose((1, 2, 0))
            show = show.astype(np.uint8)
            path = os.path.join(save_dir, '{}.png'.format(i))
            cv2.imwrite(path, show)



def visualize_dataset_flow(flows, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n, c, h, w = flows.shape

    for i in range(n):
        show = flows[i]
        show = (show + 30) * 255.0 / (30 * 2)
        show = show.astype(np.uint8)
        if show.shape[0] > show.shape[1]:
            show = np.concatenate([show[0], show[1]], 0)
        else:
            show = np.concatenate([show[0], show[1]], 1)
        path = os.path.join(save_dir, '{}.png'.format(i))
        cv2.imwrite(path, show)
