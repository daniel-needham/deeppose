import numpy as np
import torch
import cv2
from torchvision.transforms import v2


class LSPTransforms:
    def __init__(
        self,
        image_size,
        image_mean=None,
        image_std=None,
        joints_mean=None,
        joints_std=None,
    ):
        self.image_size = image_size
        self.joints_mean = joints_mean
        self.joints_std = joints_std
        self.image_mean = image_mean
        self.image_std = image_std

    def create_bounding_box_coords(self, joints):
        joints = np.copy(joints)

        x = joints[joints[:, 2] == 1][:, 0]  # in view x
        y = joints[joints[:, 2] == 1][:, 1]  # in view y

        return {
            "min_X": np.min(x),
            "max_X": np.max(x),
            "min_Y": np.min(y),
            "max_Y": np.max(y),
        }

    def crop_to_bounding_box(self, image_np, joints):
        bb = self.create_bounding_box_coords(joints)

        width = bb["max_X"] - bb["min_X"]
        height = bb["max_Y"] - bb["min_Y"]

        width *= 1.5
        height *= 1.5

        # Calculate the new minimum X and Y
        min_X = bb["min_X"] - (width - (bb["max_X"] - bb["min_X"])) / 2
        min_Y = bb["min_Y"] - (height - (bb["max_Y"] - bb["min_Y"])) / 2

        # Make sure the new bounding box is within the image's dimensions
        min_X = max(0, min_X)
        min_Y = max(0, min_Y)
        max_X = min(image_np.shape[1], min_X + width)
        max_Y = min(image_np.shape[0], min_Y + height)

        cropped_image = image_np[int(min_Y) : int(max_Y), int(min_X) : int(max_X)]

        # Normalize the joints' x and y coordinates
        recalc_joints = []
        for joint in joints:
            recalc_x = joint[0] - min_X
            recalc_y = joint[1] - min_Y
            recalc_joints.append([recalc_x, recalc_y, joint[2]])

        return cropped_image, np.array(recalc_joints), {"min_X": min_X, "min_Y": min_Y}

    def resize(self, img, joints, size=220, joint_args=None):
        ## resize
        h, w = img.shape[:2]
        ratio = w / h
        if ratio > 1:
            new_w = size
            new_h = int(size / ratio)
        else:
            new_h = size
            new_w = int(size * ratio)

        w_change = new_w / w
        h_change = new_h / h

        resized_joints = []
        for joint in joints:
            resized_x = joint[0] * w_change
            resized_y = joint[1] * h_change
            resized_joints.append([resized_x, resized_y, joint[2]])

        resized_image = cv2.resize(img, (new_w, new_h))

        ## pad
        pad_top = (size - new_h) // 2
        pad_bottom = size - new_h - pad_top
        pad_left = (size - new_w) // 2
        pad_right = size - new_w - pad_left

        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        padded_joints = []
        for joint in resized_joints:
            padded_x = joint[0] + pad_left
            padded_y = joint[1] + pad_top
            padded_joints.append([padded_x, padded_y, joint[2]])

        args = {}
        if joint_args:
            args = {**joint_args}

        args.update(
            {
                "w_change": w_change,
                "h_change": h_change,
                "pad_left": pad_left,
                "pad_top": pad_top,
            }
        )

        return padded_image, np.array(padded_joints), args

    def normalize_joints(self, joints):
        mean_x, mean_y = self.joints_mean
        std_x, std_y = self.joints_std

        joints = np.copy(joints)
        joints[:, 0] = (joints[:, 0] - mean_x) / std_x
        joints[:, 1] = (joints[:, 1] - mean_y) / std_y

        return joints

    def denormalize_joints(self, joints):
        mean_x, mean_y = self.joints_mean
        std_x, std_y = self.joints_std

        joints = np.copy(joints)
        joints[:, 0] = (joints[:, 0] * std_x) + mean_x
        joints[:, 1] = (joints[:, 1] * std_y) + mean_y

        return joints

    def normalize_image(self, image):
        image = np.copy(image)
        image = (image - self.image_mean) / self.image_std
        return image

    # def denormalize_image(self, image): # TODO: not working right may delete
    #     image *= self.image_std
    #     image += self.image_mean
    #     # image *= 255.0
    #     return image

    def __call__(self, image, joints):
        image, joints, joint_args = self.crop_to_bounding_box(image, joints)
        image, joints, joint_args = self.resize(image, joints, joint_args=joint_args)

        if all(
            [
                tensor is not None
                for tensor in [
                    self.image_mean,
                    self.image_std,
                    self.joints_mean,
                    self.joints_std,
                ]
            ]
        ):
            joints = self.normalize_joints(joints)
            image = self.normalize_image(image)

        image = torch.from_numpy(image).float()
        joints = torch.from_numpy(joints).float()

        return image, joints, joint_args
