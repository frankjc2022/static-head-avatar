# Use tinycudann, it's fast and lightweight, it also has hash encoding for spatial data and better for rendering.
# https://github.com/NVlabs/tiny-cuda-nn
# https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md

import torch
import torch.nn as nn
# import tinycudann as tcnn
import copy

# mlp_setworks_config.json in the repo is the config data, for details checkout the documentation:
# https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
# some experiments of the encoding:
# https://github.com/NVlabs/instant-ngp/issues/140


# input is 3D points, or with rotation and scale
# output is 1 dimension modifier for the opacity
class OpacityModel(nn.Module):
    def __init__(self, position_encoding_config, network_config, is_xyz_only=True):
        super().__init__()

        # encoding for position xyz
        self.xyz_encoding = tcnn.Encoding(3, encoding_config=position_encoding_config)

        # output size 1, for opacity
        output_size = 1

        rotation_size = 4
        scale_size = 3

        if is_xyz_only:
            input_size = self.xyz_encoding.n_output_dims
        else:
            input_size = self.xyz_encoding.n_output_dims + rotation_size + scale_size

        self.mlp = tcnn.Network(input_size, output_size, network_config=network_config)

    def forward(self, xyz, is_xyz_only=True, scales=None, rotations=None):
        xyz_encoded = self.xyz_encoding(xyz)
        if is_xyz_only:
            concat_input = torch.cat([xyz_encoded], dim=-1)
        else:
            concat_input = torch.cat([xyz_encoded, scales, rotations], dim=-1)
        opacity = self.mlp(concat_input)
        return opacity


# input is 3D points, or with rotation and scale. And view direction, which has shape 3, checkout the code in gaussian_renderer/__init__.py
# for how to compute the view direction
# output the modifier for SH coefficients, for the default max degree is 3, the output has 48 coefficients
class SHCoefficientModel(nn.Module):

    def __init__(self, position_encoding_config, network_config, color_size, is_xyz_only=True):
        super().__init__()

        # encoding for position xyz
        self.xyz_encoding = tcnn.Encoding(3, encoding_config=position_encoding_config)

        dir_pp_size = 3
        rotation_size = 4
        scale_size = 3

        if is_xyz_only:
            input_size = self.xyz_encoding.n_output_dims + dir_pp_size
        else:
            input_size = self.xyz_encoding.n_output_dims + dir_pp_size + rotation_size + scale_size

        self.mlp = tcnn.Network(input_size, color_size, network_config=network_config)

    def forward(self, xyz, viewdirs, is_xyz_only=True, scales=None, rotations=None):
        xyz_encoded = self.xyz_encoding(xyz)
        if is_xyz_only:
            concat_input = torch.cat([xyz_encoded, viewdirs], dim=-1)
        else:
            concat_input = torch.cat([xyz_encoded, viewdirs, scales, rotations], dim=-1)
        rgb = self.mlp(concat_input)
        return rgb


# similar to the nerf network, it first use 3D points to predict a feature vector and 1 for opacity, then use the
# feature factor concat with the view direction, input to another network and predict the color.
# input is 3D points, or with rotation and scale, and view direction
# output is tuple, modifier for opacity and sh coefficients
class OpacitySHCoefficientModel(nn.Module):

    def __init__(self, position_encoding_config, opacity_network_config, color_network_config, feature_vector_size, color_size, is_xyz_only=True):
        super().__init__()

        # first network: first input xyz and predict a feature vector and the opacity
        self.xyz_encoding = tcnn.Encoding(3, encoding_config=position_encoding_config)

        # for the feature vector, we dont want to apply activation function
        first_network_config = copy.deepcopy(opacity_network_config)
        first_network_config["output_activation"] = "None"

        self.opacity_activation_type = opacity_network_config["output_activation"]

        rotation_size = 4
        scale_size = 3

        if is_xyz_only:
            input_size = self.xyz_encoding.n_output_dims
        else:
            input_size = self.xyz_encoding.n_output_dims + rotation_size + scale_size
        # +1 is for the opacity
        self.opacity_mlp = tcnn.Network(input_size, feature_vector_size + 1, network_config=first_network_config)

        # second network: predict the SH coefficients, input is feature vector + viewing direction
        dir_pp_size = 3
        self.sh_mlp = tcnn.Network(feature_vector_size + dir_pp_size, color_size, network_config=color_network_config)

    def apply_opacity_activation(self, x):
        if self.opacity_activation_type == "Sigmoid":
            return torch.sigmoid(x)
        elif self.opacity_activation_type == "ReLU":
            return torch.relu(x)
        elif self.opacity_activation_type == "Tanh":
            return torch.tanh(x)
        elif self.opacity_activation_type == "None":
            return x  # No activation
        else:
            raise ValueError(f"Unsupported output activation: {self.opacity_activation_type}")

    def forward(self, xyz, viewdirs, is_xyz_only=True, scales=None, rotations=None):

        # first network
        xyz_encoded = self.xyz_encoding(xyz)
        if is_xyz_only:
            opacity_and_features = self.opacity_mlp(xyz_encoded)
        else:
            opacity_and_features = self.opacity_mlp(torch.cat([xyz_encoded, scales, rotations], dim=-1))
        # split into opacity and features
        opacity_raw, features = opacity_and_features[:, :1], opacity_and_features[:, 1:]
        opacity = self.apply_opacity_activation(opacity_raw)

        # second network
        # viewdirs_encoded = self.viewdir_encoding(viewdirs)
        concat_input = torch.cat([features, viewdirs], dim=-1)
        rgb = self.sh_mlp(concat_input)

        return opacity, rgb

