
import torch
from scene import Scene
from gaussian_renderer import render
import torchvision
from tqdm import tqdm
import os
from os import makedirs
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
import numpy as np
from scene.cameras import Camera
import time
import shutil

def calculate_center_point(gaussian_model):
    points = gaussian_model.get_xyz
    center = torch.mean(points, dim=0)
    return center


def create_circular_cameras_around_center(center, base_camera, radius, num_cameras):

    # https://github.com/RenderMe-360/RenderMe-360-Benchmark/blob/4c61ddeba15ff35d61c4eae147b2341bf44ff5f6/case_specific_nvs/neuralvolumes/eval/cameras/rotate.py
    cameras = []
    center_np = center.cpu().numpy()

    for idx in range(num_cameras):

        angle = idx * 2. * np.pi / num_cameras
        x = np.cos(angle) * radius
        y = 0
        z = np.sin(angle) * radius


        campos = np.array([x, y, z], dtype=np.float32)

        lookat = center_np
        up = np.array([0., -1., 0.], dtype=np.float32)
        forward = lookat - campos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        camrot = np.array([right, -up, forward], dtype=np.float32)

        camT = -camrot @ campos

        # Create cameras
        R_torch = torch.tensor(camrot, dtype=torch.float32, device="cuda")
        T_torch = torch.tensor(camT, dtype=torch.float32, device="cuda")

        camera = Camera(
            colmap_id=idx,
            R=R_torch.cpu().numpy(),
            T=T_torch.cpu().numpy(),
            FoVx=base_camera.FoVx,
            FoVy=base_camera.FoVy,
            image=torch.zeros((3, base_camera.image_height, base_camera.image_width), device="cuda"),
            mask=None,
            image_name=f"rotating_camera_{idx}",
            uid=f"rotating_camera_{idx}",
            data_device="cuda",
        )
        cameras.append(camera)

    return cameras


def render_scene_with_rotating_cameras(gaussian_model, pipeline, center, radius, num_cameras, model_path, base_camera, iteration):

    render_path = os.path.join(model_path, "rotating_cameras", "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    # Create rotating cameras
    cameras = create_circular_cameras_around_center(center, base_camera, radius, num_cameras)

    total_time = 0.0
    progress_bar = tqdm(cameras, desc="Rendering rotating cameras")

    for idx, camera in enumerate(progress_bar):

        start_time = time.time()
        rendering = render(camera, gaussian_model, pipeline, torch.tensor([1., 1., 1.], device="cuda"))["render"]
        total_time += time.time() - start_time

        current_fps = (idx + 1) / total_time
        progress_bar.set_postfix(fps=f"{current_fps:.2f}")

        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))

    fps = len(cameras) / total_time
    print(f"Average render FPS: {fps:.2f}")

    return render_path


def render_with_rotating_cameras(dataset: ModelParams, iteration: int, pipeline: PipelineParams, radius, num_cameras):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        center = calculate_center_point(gaussians)

        base_camera = scene.getTrainCameras()[0]  # Use the first camera as the base

        render_path = render_scene_with_rotating_cameras(
            gaussian_model=gaussians,
            pipeline=pipeline,
            center=center,
            radius=radius,
            num_cameras=num_cameras,
            model_path=dataset.model_path,
            base_camera=base_camera,
            iteration=scene.loaded_iter
        )

        return render_path


if __name__ == "__main__":

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering rotating cameras for " + args.model_path)

    # 5 second, 30 fps = 150 images
    render_path = render_with_rotating_cameras(
        dataset=model.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        radius=5.0,
        num_cameras=150
    )

    time.sleep(5)

    upper_level_render_path = os.path.dirname(render_path)
    os.system(fr'ffmpeg -framerate 30 -i {render_path}/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p {upper_level_render_path}/renders.mp4')