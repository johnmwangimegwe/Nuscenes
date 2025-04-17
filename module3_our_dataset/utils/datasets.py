import numpy as np
import sys 
import os


def load_calib(filename: str) -> np.array:
    """
    return a list [fx, cx, fy, cy, k1, k2, t1, t2, k3]
    where fx, fy, cx, cy are elements in projection_matrix
    and k1, k2, k3, t1, t2 are camera distortion_coefficients
    """
    trans_x, trans_y, trans_z = [-0.07, -0.05, 0]  
    with open(filename, "r") as f:
        y = yaml.load(f)
        camera_matrix = np.resize(y["camera_matrix"]["data"], (3, 3))
        distortion = np.array(y["distortion_coefficients"]["data"])
        calib_param = np.array([camera_matrix[0, 0], camera_matrix[0, 2], \
                                camera_matrix[1, 1], camera_matrix[1, 2], \
                                *distortion, *[trans_x, trans_y, trans_z]]) 
    return calib_param



def from_3d_to_2d(points: np.array, calib_param: np.array) -> np.array:
    """
    3d to 2d projection
    input: 3d radar coordinate -> np.array with shape (4, n)
    output: image coordinate -> np.array: (n, 2) and (n, 4)
    camera coordinate: u, v, r = radar coordinate: x, -z, y
    """
    # radar coordinate -> image coordinate
    x, y, z, velocity = points[0], -points[2], points[1], points[3]
    u, v = projection_xyr_to_uv([x, y, z], calib_param)

    # outputs xyzV (in camera coordinate)
    trans_x, trans_y, trans_z = calib_param[-3:]    # actually, trans_z=0, so actually no translation applied
    uv = np.array([*zip(u, v)]).astype(np.int64)
    xyzV = np.array([*zip(x, y, z+trans_z, velocity)])
    return uv, xyzV


def load_radar_data(match_list, cur_frame_idx, overlay_num, max_depth, min_velocity):
    matched_radar_idx = match_list[cur_frame_idx]
    idx_range = range(matched_radar_idx[0], matched_radar_idx[0] - overlay_num, -1)
    
    x, y, z, v = [], [], [], []
    for i in idx_range:
        x += point_data[i]["Data"]["x"]
        y += point_data[i]["Data"]["y"]
        z += point_data[i]["Data"]["z"]
        v += point_data[i]["Data"]["velocity"]
    
    points_3d = np.array([x, y, z, v])
    uv, xyzV = from_3d_to_2d(points_3d, calib_param)
    
    filtering = [(0 <= i[0] < 640) and (0 <= i[1] < 480) and (j[2] < max_depth) and (abs(j[3]) >= min_velocity) 
                 for i, j in zip(uv, xyzV)]
    uv, xyzV = uv[filtering], xyzV[filtering]
    
    return np.concatenate((uv, xyzV[..., 2:]), axis=-1)

print("**********************************************************")

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class NuScenesDataset(Dataset):
    """
    Custom dataset class for loading NuScenes data.
    """

    def __init__(self, root_dir, split="train", transform=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir (str): Path to the NuScenes dataset directory.
        - split (str): "train", "val", or "test".
        - transform (callable, optional): Optional transform to apply to the data.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load annotations or metadata
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        Load annotations or metadata for the dataset.
        Replace this with actual logic for loading NuScenes annotations.
        """
        # Example: Load a list of file paths or bounding box annotations
        annotations = []
        split_dir = os.path.join(self.root_dir, self.split)
        for filename in os.listdir(split_dir):
            if filename.endswith(".jpg"):  # Example: Load image files
                annotations.append(os.path.join(split_dir, filename))
        return annotations

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - dict: A dictionary containing the image, LiDAR data, radar data, and labels.
        """
        # Example: Load an image
        img_path = self.annotations[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Replace with actual logic for loading LiDAR, radar, and labels
        lidar_data = np.random.rand(100, 3)  # Dummy LiDAR data
        radar_data = np.random.rand(100, 4)  # Dummy radar data
        label = 0  # Dummy label

        return {
            "image": image,
            "lidar_data": lidar_data,
            "radar_data": radar_data,
            "label": label,
        }