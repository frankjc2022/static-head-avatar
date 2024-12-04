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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        # sh_degree=pc.max_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    ########## for mlp
    # modify the shs and/or opacity by mlp
    if pc.mlp_type and pc.mlp:
        # get view direction per point
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

        if pc.mlp_type == "color":
            if pc.direct_mlp:
                if not pc.mlp_full_input:
                    colors_precomp = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=True).to(torch.float32)
                else:
                    colors_precomp = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=False, scales=scales, rotations=rotations).to(torch.float32)
                shs = None
            else:
                if not pc.mlp_full_input:
                    modifier = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=True)
                else:
                    modifier = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=False, scales=scales, rotations=rotations)
                shs = torch.reshape(modifier, (-1, 16, 3)) * shs
        elif pc.mlp_type == "opacity":
            if pc.direct_mlp:
                if not pc.mlp_full_input:
                    opacity = pc.mlp(means3D, is_xyz_only=True).to(torch.float32)
                else:
                    opacity = pc.mlp(means3D,is_xyz_only=False, scales=scales, rotations=rotations).to(torch.float32)
            else:
                if not pc.mlp_full_input:
                    modifier = pc.mlp(means3D, is_xyz_only=True)
                else:
                    modifier = pc.mlp(means3D, is_xyz_only=False, scales=scales, rotations=rotations)
                opacity = pc.opacity_activation(modifier * pc._opacity)
        elif pc.mlp_type == "color_opacity":
            if pc.direct_mlp:
                if not pc.mlp_full_input:
                    opacity, rgb = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=True)
                else:
                    opacity, rgb = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=False, scales=scales, rotations=rotations)
                opacity = opacity.to(torch.float32)
                colors_precomp = rgb.to(torch.float32)
                shs = None
            else:
                if not pc.mlp_full_input:
                    modifier_opacity, modifier_shs = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=True)
                else:
                    modifier_opacity, modifier_shs = pc.mlp(means3D, dir_pp_normalized, is_xyz_only=False, scales=scales, rotations=rotations)
                shs = torch.reshape(modifier_shs, (-1, 16, 3)) * shs
                opacity = pc.opacity_activation(modifier_opacity * pc._opacity)



    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
