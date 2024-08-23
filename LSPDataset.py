import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import math


class LSPDataset(Dataset):
    def __init__(
        self, image_path, image_indexes, joints_path, joints_indexes, transforms=None
    ):
        self.image_path = image_path
        self.joints_indexes = joints_indexes
        self.image_indexes = image_indexes
        self.transforms = transforms
        self.joints = loadmat(joints_path)["joints"]
        self.joints = self.joints.transpose(2, 0, 1)
        # self.joints = self.joints[joints_indexes]

        assert len(self.joints_indexes) == len(
            self.image_indexes
        )  # Ensure the lengths match

    def __len__(self):
        return len(self.image_indexes)

    def __getitem__(self, idx):
        image_idx = self.image_indexes[idx]
        image = self.load_image(image_idx)
        joints_idx = self.joints_indexes[idx]
        joints = self.joints[joints_idx]

        # Apply transformations if provided
        if self.transforms:
            image, joints, joint_args = self.transforms(image, joints)

        return image, joints, joint_args

    def to_five_digit_string(self, number):
        return f"{number:05}"

    def load_image(self, index):
        file_name = (
            self.image_path + "/im" + self.to_five_digit_string(index + 1) + ".jpg"
        )
        image = Image.open(file_name)
        return np.array(image).astype(float)  # Convert the image to a numpy array

    def add_transformation(self, transform):
        self.transforms = transform
