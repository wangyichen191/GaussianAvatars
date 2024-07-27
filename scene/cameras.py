#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image_height, image_path, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep

        # try:
        #     self.data_device = torch.device(data_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #     self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera_IMAvatar(nn.Module):
    def __init__(self, colmap_id, R, T, bg, image_path,
                 image_name, uid, timestep,
                 FoVx=None, FoVy=None,
                 w=None, h=None, fx=None, fy=None, cx=None, cy=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera_IMAvatar, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.bg = bg
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = w
        self.image_height = h
        self.timestep = timestep

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]

        # if gt_alpha_mask is not None:
        #     self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            
        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 5.0
        self.znear = 0.5

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)

        if FoVx is not None:
            self.FoVy =  FoVy
            self.FoVx =  FoVx
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        else:
            self.FoVy =  2 * np.arctan(h / (2.0 * fy))
            self.FoVx =  2 * np.arctan(w / (2.0 * fx))
            self.projection_matrix = getProjectionMatrix2(w, h, fx, fy, cx, cy, self.znear, self.zfar).transpose(0,1).to(self.data_device)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def cuda(self):
        cam = copy.deepcopy(self)
        for key in dir(cam):
            if isinstance(getattr(cam, key), torch.Tensor):
                setattr(cam, key, getattr(cam, key).cuda())
        return cam

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep

