import os
import sys
import json
import math
import numpy as np
import magnum as mn
import open3d as o3d
from imageio import imread
from mat4py import loadmat
from typing import Any, Dict, List, Optional, Tuple

class Projector():
    def __init__(self,
                 intrinsics: Dict
                 ):
        
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.img_h = intrinsics['img_h']
        self.img_w = intrinsics['img_w']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']

        self.x_scale, self.y_scale, self.ones = self._compute_scaling_params()

    
    def _compute_scaling_params(self):
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy

        x = np.arange(start=0, stop=self.img_w)
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, self.img_h, axis=0)
        x = x.astype(np.float)

        y = np.arange(start=0, stop=self.img_h)
        y = np.expand_dims(y, axis=1)
        y = np.repeat(y, self.img_w, axis=1)
        y = y.astype(np.float)

        # 0.5 is so points are projected through the center of pixels
        x_scale = (x - cx - 0.5) / fx#; print(x_scale[0,0,:])
        y_scale = (y - cy - 0.5) / fy#; print(y_scale[0,:,0]); stop
        ones = np.ones((self.img_h, self.img_w), dtype=np.float)

        return x_scale, y_scale, ones

    def _point_cloud(self, depth, depth_scaling=1.0):
        z = depth / float(depth_scaling)
        x = z * self.x_scale
        y = z * self.y_scale
        ones = self.ones

        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        z = np.expand_dims(z, axis=2)
        ones = np.expand_dims(ones, axis=2)
        
        xyz1 = np.concatenate((x,y,z,ones), axis=2)
        return xyz1
    
    def _transform_y_up(self,xyz1):
        R_yup = np.array([[1, 0, 0, 0],
                          [0, np.cos(np.pi), -np.sin(np.pi), 0],
                          [0, np.sin(np.pi), np.cos(np.pi), 0],
                          [0, 0, 0, 1],
                        ])
        return np.matmul(R_yup, xyz1)

    def _transform_camera_to_world(self, xyz1, T):
        return np.matmul(T, xyz1)
    
    def unproject_depth(self, depth, T):
        no_depth_mask = depth == 0
    
        xyz1 = self._point_cloud(depth)
   
        # shape: (height * width, 4)
        xyz1 = np.reshape(xyz1, (xyz1.shape[0] * xyz1.shape[1], 4))

        # shape: (4, height * width)
        xyz1_t = np.transpose(xyz1, (1,0))  # [4,HxW]
        
        # Transformed points from CV camera coordinate system to Robotic
        # coordinate system  # y is up:
        # shape: xyz1_w(4, height * width)
        xyz1_t = self._transform_y_up(xyz1_t)

        # Transformed points from camera coordinate system to world coordinate system  # GEO:
        # shape: xyz1_w(4, height * width)
        xyz1_w = self._transform_camera_to_world(xyz1_t, T)

        # shape: (height * width, 3)
        world_xyz = xyz1_w.transpose(1,0)[:, :3]

        # shape: (batch_size, height, width, 3)
        point_cloud = np.reshape(world_xyz,(depth.shape[0],
                                            depth.shape[1],
                                            3,))

        return point_cloud, no_depth_mask



class AVDParser():

    def _get_scale(self, filename: str) -> float:
        data = loadmat(filename)
        scale = data['scale']
        return scale

    def _load_positions(self, filename: str) -> Dict:
        data = loadmat(filename)
        scale = data['scale']
        world_pose = data['image_structs']['world_pos']
        directions = data['image_structs']['direction']
        image_filenames = data['image_structs']['image_name']

        poses = {}
        for t,d,n in zip(world_pose, directions, image_filenames):
            if (len(t) > 0) and (len(d) > 0):
                p = [x[0] for x in t]
                p = np.array(p) * scale

                d = [x[0] for x in d]
                d = np.array(d)

                target = p + d* scale
                up = np.array([0,-1,0], dtype=np.float)

                a = mn.Matrix4()
                a = a.look_at(
                    mn.Vector3(p),
                    mn.Vector3(target),
                    mn.Vector3(up),
                )
                T = np.array(a)
                poses[n] = T
            else:
                poses[n] = None
        return poses

    def _load_intrinsics(self, filename: str) -> Dict:
        with open(filename, 'r') as f:
            for _ in range(4):
                line = f.readline()
        elements = line.split(' ')
        intrinsics = {
            'img_w': int(elements[2]),
            'img_h': int(elements[3]),
            'fx': float(elements[4]),
            'fy': float(elements[5]),
            'cx': float(elements[6]),
            'cy': float(elements[7]),
            'k1': float(elements[8]),
            'k2': float(elements[9]),
            'k3': float(elements[10]),
            'k4': float(elements[11]),
        }
        return intrinsics


    def compute_point_cloud_for_house(self, dirname: str, resolution: float = 30) -> o3d.geometry.PointCloud:
        """
        Reconstructs the point cloud for a house in the AVD dataset
        dirname: provides the root dir of a house (eq Home_001_1)
        resolution: point cloud resolution in milimeters

        NOTE: the point cloud is first built in millimeters
        Convert to meters happens in the end.
        """


        # load intrinsics
        intrinsics_filename = os.path.join(dirname, 'cameras.txt')
        intrinsics = self._load_intrinsics(intrinsics_filename)

        # init Projector
        # NOTE: current projector does not take into account distortion
        projector = Projector(intrinsics)

        # load positions
        pose_filename = os.path.join(dirname, 'image_structs.mat')
        poses = self._load_positions(pose_filename)

        # loop through images and unprojects pixels
        rgbdir = os.path.join(dirname, 'jpg_rgb')
        depthdir = os.path.join(dirname, 'high_res_depth')

        point_cloud = o3d.geometry.PointCloud()

        image_filenames = os.listdir(rgbdir)
        for image_filename in image_filenames:
            image_name = image_filename.split('.')[0]
            T = poses[image_filename]

            if T is not None:

                rgb = imread(os.path.join(rgbdir, image_filename))
                rgb = rgb.astype(np.float)
                rgb = rgb / 255.0

                depth = imread(os.path.join(depthdir, image_name[:-2]+'03.png'))
                depth = depth.astype(np.float32)
                depth[depth>3000] = 0 # discard pixels projected further than 3m. Helps reducing noise.

                pc, mask_outliers = projector.unproject_depth(depth, T)

                mask_inliers = ~mask_outliers

                rgb = rgb[mask_inliers]

                pc = pc[mask_inliers]

                pc[:,1] *= -1.0 # to y is up

                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(pc)
                tmp_pcd.colors = o3d.utility.Vector3dVector(rgb)

                point_cloud += tmp_pcd

                point_cloud=point_cloud.voxel_down_sample(voxel_size=resolution)

        # convert PC to meters
        points = np.array(point_cloud.points) / 1000.0
        point_cloud.points = o3d.utility.Vector3dVector(points)

        return point_cloud


if __name__=='__main__':
    rootdir = 'data/avd/src'

    avd_parser = AVDParser()

    house_name = 'Home_001_1'
    dirname = os.path.join(rootdir, house_name)
    point_cloud = avd_parser.compute_point_cloud_for_house(dirname)
    print(point_cloud.shape)