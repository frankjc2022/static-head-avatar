import os
import logging
import shutil
from renderme_360_reader import SMCReader
from pathlib import Path
import cv2
import numpy as np
import sys
import time
import os
import cv2
import numpy as np


# this is for avoiding a Windows error
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start_time = time.time()

# sleep time between each step, avoid racing
sleeptime = 5

colmap_executable = fr"E:\3DGS\gaussian-splatting\datasets\colmap-x64-windows-cuda\bin\colmap.exe"

camera_count = 60

###### STEP 1: extracts data from the renderme-360 dataset file smc (raw and anno)

# step 1 output two folders, input and mask, which output from the dataset directly
def step1(frame_id, reader_raw, reader_anno, outdir_input, outdir_mask):
    # the dataset has totoal 60 cameras
    for i in range(camera_count):
        camera_id = f"{i:02d}"
        image = reader_raw.get_img(camera_id, 'color', frame_id)
        image_outpath = os.path.join(outdir_input, f"{camera_id}.jpg")
        cv2.imwrite(image_outpath, image)

        # the distorted matrix is all 0, no need to undistort the mask
        mask = reader_anno.get_img(camera_id, 'mask', frame_id)
        mask_outpath = os.path.join(outdir_mask, f"{camera_id}.jpg.png") # this naming is for colmap masking
        cv2.imwrite(mask_outpath, mask)

        print(f"Saving image and mask - camera id (id in dataset): {camera_id}")


###### STEP 2: call the 3dgs convert script
# we specify the camera model as PINHOLE, because we checked that the distort matrix in the dataset is all zeros
def step2(source_path):

    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

    # Initialize Feature Extraction
    feat_extracton_cmd = fr'"{colmap_executable}" feature_extractor --database_path {source_path}/distorted/database.db --ImageReader.mask_path {source_path}/mask --image_path {source_path}/input --ImageReader.single_camera 0 --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1'
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    time.sleep(sleeptime)

    ## Feature matching
    # add information to distorted/database.db
    feat_matching_cmd = fr'"{colmap_executable}" exhaustive_matcher --database_path {source_path}/distorted/database.db --SiftMatching.use_gpu 1'
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print(f"COLMAP - Initialize Feature Extraction - done: sleep for {sleeptime} seconds...")
    time.sleep(sleeptime)

    # Bundle adjustment
    # The default Mapper tolerance is unnecessarily large, decreasing it speeds up bundle adjustment steps.
    # add information to distorted/database.db
    # create distorted/sparse/0/cameras.bin
    # create distorted/sparse/0/images.bin
    # create distorted/sparse/0/points3D.bin
    # create distorted/sparse/0/project.ini
    mapper_cmd = fr'"{colmap_executable}" mapper --database_path {source_path}/distorted/database.db --image_path {source_path}/input --output_path {source_path}/distorted/sparse --Mapper.ba_global_function_tolerance=0.000001'
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print(f"COLMAP - Mapper - done: sleep for {sleeptime} seconds...")
    time.sleep(sleeptime)

    # Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    # create images/* (undistorted images)
    # create sparse/cameras.bin
    # create sparse/images.bin
    # create sparse/points3D.bin
    # create stereo/*
    # create run-colmap-geometric.sh
    # create run-colmap-photometric.sh
    img_undist_cmd = fr'"{colmap_executable}" image_undistorter --image_path {source_path}/input --input_path {source_path}/distorted/sparse/0 --output_path {source_path} --output_type COLMAP'
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print(f"COLMAP - undistortion - done: sleep for {sleeptime} seconds...")
    time.sleep(sleeptime)

    print(f"move files to specified folder to match gaussian splatting folder structure...")
    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    print(f"move files to specified folder to match gaussian splatting folder structure... done!")




###### STEP 3: convert cameras.bin to cameras.txt
def step3(outdir):
    print("convert colmap bin files to txt files...")
    sparse_text_outdir = fr"{outdir}/sparse_text"
    Path(sparse_text_outdir).mkdir(parents=True, exist_ok=True)
    os.system(fr"{colmap_executable} model_converter  --input_path {outdir}/sparse/0  --output_path {sparse_text_outdir}  --output_type TXT")
    print("convert colmap bin files to txt files... done!")

###### STEP 4: use the cameras.bin of undistorted information from colmap, to undistorted the mask png
def read_cameras_txt(file_path):
    cameras = {}
    with open(file_path, "r") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def read_images_txt(file_path):
    # Parse images.txt
    image_to_camera = {}
    with open(file_path, 'r') as f:
        line = f.readline().strip()

        while line:
            line = line.strip()
            if line.startswith("#") or not line:
                line = f.readline()
                continue
            parts = line.split()
            if len(parts) >= 10:  # First line of image metadata
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
                image_to_camera[name] = {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "quaternion": (qw, qx, qy, qz),
                    "translation": (tx, ty, tz)
                }
                # Skip the next line (POINTS2D)
                line = f.readline()
            line = f.readline()

    return image_to_camera


# for more settings please see:
# - https://colmap.github.io/cameras.html
# - https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
def get_camera_matrix_and_distortion(camera):
    model = camera["model"]
    params = camera["params"]

    if model == "PINHOLE":
        fx, fy, cx, cy = params
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))  # No distortion for PINHOLE
    elif model == "RADIAL":
        fx, cx, cy, k1, k2 = params
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)  # Add tangential if needed
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return camera_matrix, dist_coeffs


def undistort_mask(mask_path, output_mask_path, camera_matrix, dist_coeffs, original_dim, undistorted_dim):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Fix orientation if dimensions are swapped
    if mask.shape[:2] != (original_dim[1], original_dim[0]):  # Check for mismatch
        print(f"Fixing orientation for {mask_path}")
        mask = cv2.transpose(mask)  # Swap width and height

    # Compute the undistortion maps
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, original_dim, 1, undistorted_dim)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, undistorted_dim, cv2.CV_16SC2)

    # Undistort the mask
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)

    # Save the undistorted mask
    cv2.imwrite(output_mask_path, undistorted_mask)
    print(f"Undistorted mask saved to {output_mask_path}")


def process_all_masks(outdir, outdir_input, outdir_mask, outdir_undistorted_mask):

    undistorted_folder = fr"{outdir}/images"
    masked_image_folder = fr"{outdir}/masked_images"
    cameras_txt_path = fr"{outdir}/sparse_text/cameras.txt"
    images_txt_path = fr"{outdir}/sparse_text/images.txt"

    # we need the images.txt to find the match camera_id of given image name
    cameras = read_cameras_txt(cameras_txt_path)
    images = read_images_txt((images_txt_path))

    # Ensure output folder exists
    os.makedirs(masked_image_folder, exist_ok=True)

    for filename in os.listdir(outdir_input):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = filename #os.path.splitext(filename)

            # Paths
            original_image_path = os.path.join(outdir_input, filename)
            undistorted_image_path = os.path.join(undistorted_folder, filename)
            mask_path = os.path.join(outdir_mask, f"{base_name}.png")
            output_mask_path = os.path.join(outdir_undistorted_mask, f"{base_name}.png")
            masked_image_path = os.path.join(masked_image_folder, f"{base_name}")

            if not os.path.exists(mask_path):
                print(mask_path)
                print(f"Mask not found for {filename}, skipping.")
                continue
            if not os.path.exists(undistorted_image_path):
                print(f"Undistorted image not found for {filename}, skipping.")
                continue

            # Get dimensions
            original_image = cv2.imread(original_image_path)
            undistorted_image = cv2.imread(undistorted_image_path)
            original_dim = (original_image.shape[1], original_image.shape[0])  # (width, height)
            undistorted_dim = (undistorted_image.shape[1], undistorted_image.shape[0])  # (width, height)

            # Get camera parameters
            camera_id = images[base_name]["camera_id"] #1  # undistorted txt file should only contain 1 pinhole camera
            if camera_id not in cameras:
                print(f"Camera ID {camera_id} not found in cameras.txt")
                continue

            camera = cameras[camera_id]
            camera_matrix, dist_coeffs = get_camera_matrix_and_distortion(camera)

            # Undistort the mask
            # print(f"Processing {filename}...")
            undistort_mask(mask_path, output_mask_path, camera_matrix, dist_coeffs, original_dim, undistorted_dim)
            apply_mask_and_save(original_image_path, output_mask_path, masked_image_path)


def apply_mask_and_save(image_path, mask_path, output_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    white_background = np.ones_like(image) * 255

    # inside mask keep the original image, outside set to white
    masked_image = np.where(binary_mask[:, :, None] == 255, image, white_background)

    cv2.imwrite(output_path, masked_image)
    print(f"Masked image saved to {output_path}")


if __name__ == '__main__':


    # https://renderme-360.github.io/inner-benchmark-nvs.html#NVS
    # above benchmark videos use:
    # - e10, e11
    # - 0262, 0295
    # for 0295, some images are ignored in colmap, due to not enough key points


    all_ids = ["0026", "0041", "0048", "0094", "0100", "0116", "0156", "0168", "0175", "0189", "0195", "0232", "0250", "0253", "0259", "0262", "0278", "0290", "0295", "0297"]
    expression_ids = ["e0", "e10", "e11"] # 0 - nothing, 10 - smile, 11 -open mouse
    # expression_ids = ["e8", "e2", "e4"] # 8 - tongue out, 2 - teeth, 4 move mouth


    for id_ in ids:
        for expression_id in expression_ids:
            subject_id = id_#"0295"
            fileprefix = expression_id#"e11"  # e: expression
            frame_id = 60
            dataset_path_prefix = fr'E:\datasets\3dgs\renderme_360\20ids'

            reader_raw = SMCReader(fr'{dataset_path_prefix}\raw\{subject_id}\{subject_id}_{fileprefix}_raw.smc')
            reader_anno = SMCReader(fr'{dataset_path_prefix}\anno\{subject_id}\{subject_id}_{fileprefix}_anno.smc')

            outdir = fr'F:/datasets/{subject_id}_{fileprefix}_{frame_id}'  ## <------ output here
            outdir_mask = fr'{outdir}/mask'
            outdir_undistorted_mask = fr'{outdir}/undistorted_mask'
            outdir_input = fr'{outdir}/input'
            Path(outdir_input).mkdir(parents=True, exist_ok=True)
            Path(outdir_mask).mkdir(parents=True, exist_ok=True)
            Path(outdir_undistorted_mask).mkdir(parents=True, exist_ok=True)

            source_path = outdir

            step1(frame_id, reader_raw, reader_anno, outdir_input, outdir_mask)
            print(f"Sleeping for {sleeptime} seconds...")
            time.sleep(sleeptime)

            step2(source_path)
            print(f"Sleeping for {sleeptime} seconds...")
            time.sleep(sleeptime)

            step3(outdir)
            print(f"Sleeping for {sleeptime} seconds...")
            time.sleep(sleeptime)

            # # Process all masks
            process_all_masks(outdir, outdir_input, outdir_mask, outdir_undistorted_mask)

            print(f"Done: total {time.time() - start_time:.2f} seconds.")
