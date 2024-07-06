import torch
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
# import trimesh
import os
import configargparse
import open3d as o3d
import cv2
import imageio


'''
Setup
'''

# set cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
PointCloud reconstruction
'''

###################################################################################################
# Usage Example
###################################################################################################

# python pc_reconstruction.py --datadir /data/ForPlane-main/ForPlane-main/exps/endonerf_cutting_tissues_twice_32k_gt_depth --depth_smoother

###################################################################################################


def reconstruct_pointcloud(rgb_np, depth_np, vis_rgbd=False, depth_filter=None, crop_left_size=0):
    
    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]
        depth_np = depth_np[:, crop_left_size:]

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

    rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth_np.astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)

    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(hwf[1],hwf[0], hwf[2], hwf[2], hwf[1] / 2, hwf[0] / 2)
    )

    return pcd



if __name__ == '__main__':
    cfg_parser = configargparse.ArgumentParser()
    cfg_parser.add_argument('--config_file', type=str, 
                        help='config file path')
    cfg_parser.add_argument('--reload_ckpt', type=str, default='',
                        help='model ckpt to reload')
    cfg_parser.add_argument("--no_pc_saved", action='store_true',
                        help='donot save reconstructed point clouds?')
    cfg_parser.add_argument('--out_postfix', type=str, default='',
                        help='the postfix append to the output directory name')
    cfg_parser.add_argument("--vis_rgbd", action='store_true', 
                        help='visualize RGBD output from NeRF?')
    cfg_parser.add_argument("--start_t", type=float, default=0.0,
                        help='time of start frame')
    cfg_parser.add_argument("--end_t", type=float, default=1.0,
                        help='time of end frame')
    cfg_parser.add_argument("--n_frames", type=int, default=1,
                        help='num of frames')
    cfg_parser.add_argument("--depth_smoother", action='store_true',
                        help='apply bilateral filtering on depth maps?')
    cfg_parser.add_argument("--depth_smoother_d", type=int, default=28,
                        help='diameter of bilateral filter for depth maps')
    cfg_parser.add_argument("--depth_smoother_sv", type=float, default=64,
                        help='The greater the value, the depth farther to each other will start to get mixed')
    cfg_parser.add_argument("--depth_smoother_sr", type=float, default=32,
                        help='The greater its value, the more further pixels will mix together')
    cfg_parser.add_argument("--crop_left_size", type=int, default=75,
                        help='the size of pixels to crop')
    cfg_parser.add_argument("--datadir", type=str,
                        help='path of the result')

    cfg = cfg_parser.parse_args()

    # set render params for DaVinci endoscopic
    hwf = [512, 640, 569.46820041]

    # output directory
    if not cfg.no_pc_saved:
        out_dir = os.path.join(cfg.datadir, 'ply_dnfplane_pulling')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # build depth filter
    if cfg.depth_smoother:
        depth_smoother = (cfg.depth_smoother_d, cfg.depth_smoother_sv, cfg.depth_smoother_sr)
    else:
        depth_smoother = None
        
    image_paths = os.path.join(cfg.datadir, 'endonerf_32k', 'estm', 'rgb')
    # image_paths = "Endonerf_full_datasets/Hamlyn_20_1/images/"
    image = [imageio.imread(os.path.join(image_paths, path)) for path in os.listdir(image_paths)]
    image = np.array(image)
    
    depth_paths = os.path.join(cfg.datadir, 'endonerf_32k', 'estm', 'depth')
    # depth_paths = "Endonerf_full_datasets/Hamlyn_20_1/depth/"
    depth = [imageio.imread(os.path.join(depth_paths, path)) for path in os.listdir(depth_paths)]
    #depth = (255-np.array(depth,dtype = np.float32)) / 255
    depth = np.array(depth,dtype = np.float32) / 255


    # reconstruct pointclouds
    print('Reconstructing point clouds...')

    pcds = []
    for i in range(len(image)):
        print('>>> i=', i, " /", len(image))
        pcd = reconstruct_pointcloud(image[i], depth[i], cfg.vis_rgbd, depth_filter=depth_smoother, crop_left_size=cfg.crop_left_size)
        pcds.append(pcd)

    if not cfg.no_pc_saved:
        print('Saving point clouds...')

        for i, pcd in enumerate(pcds):
            
            fn = os.path.join(out_dir, f"frame_{i:06d}_pc.ply")
            o3d.io.write_point_cloud(fn, pcd)
        
        print('Point clouds saved to', out_dir)