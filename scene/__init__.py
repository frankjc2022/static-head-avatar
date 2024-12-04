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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch import save as torch_save
from torch import load as torch_load

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], init_random_points=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, masks_dir=args.masks)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            #### try to load pre-trained mlp
            load_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            for filename in os.listdir(load_path):
                if filename.endswith("_mlp_model.pt"):
                    mlp_type = filename.split("_mlp_model.pt")[0]
                    if mlp_type.startswith("direct_fullinput_"):
                        mlp_type = mlp_type[len("direct_fullinput_"):]
                        self.gaussians.direct_mlp = True
                        self.gaussians.mlp_full_input = True
                    elif mlp_type.startswith("direct_"):
                        mlp_type = mlp_type[len("direct_"):]
                        self.gaussians.direct_mlp = True
                        self.gaussians.mlp_full_input = False
                    elif mlp_type.startswith("fullinput_"):
                        mlp_type = mlp_type[len("fullinput_"):]
                        self.gaussians.direct_mlp = False
                        self.gaussians.mlp_full_input = True
                    else:
                        self.gaussians.direct_mlp = False
                        self.gaussians.mlp_full_input = False
                    self.gaussians.mlp_type = mlp_type
                    self.gaussians.mlp = torch_load(os.path.join(load_path, filename))
                    print(f"Loaded MLP model: {filename}")
                    break

        else:
            ####### handle random points initialization
            if not args.random_init_points:
                # original initialization from point cloud
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            else:
                # generate random points
                center = -scene_info.nerf_normalization["translate"]
                radius = 1.5#scene_info.nerf_normalization["radius"]
                self.gaussians.create_from_random(center, radius, radius, init_points=init_random_points, train_cameras=self.getTrainCameras())

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.mlp_type:
            mlp_savename = f"{self.gaussians.mlp_type}_mlp_model.pt"
            if self.gaussians.direct_mlp and self.gaussians.mlp_full_input:
                mlp_savename = f"direct_fullinput_{mlp_savename}"
            elif self.gaussians.direct_mlp and not self.gaussians.mlp_full_input:
                mlp_savename = f"direct_{mlp_savename}"
            elif not self.gaussians.direct_mlp and self.gaussians.mlp_full_input:
                mlp_savename = f"fullinput_{mlp_savename}"
            torch_save(self.gaussians.mlp, os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), f"{mlp_savename}"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]