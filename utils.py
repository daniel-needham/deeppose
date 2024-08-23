from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

body_parts = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
]

connections = [
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("right_hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_wrist", "right_elbow"),
    ("right_elbow", "right_shoulder"),
    ("right_shoulder", "neck"),
    ("left_shoulder", "neck"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "head_top"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
]


connections_indices = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (8, 12),
    (9, 12),
    (9, 10),
    (10, 11),
    (12, 13),
    (8, 2),
    (9, 3),
]


from PIL import Image
import matplotlib.patches as patches
import math


def to_five_digit_string(number):
    return f"{number:05}"


def create_bounding_box_coords(joints):
    joints = np.copy(joints)

    x = joints[joints[:, 2] == 1][:, 0]  # in view x
    y = joints[joints[:, 2] == 1][:, 1]  # in view y

    return {
        "min_X": np.min(x),
        "max_X": np.max(x),
        "min_Y": np.min(y),
        "max_Y": np.max(y),
    }


def load_image(index):
    file_name = "lsp/images/im" + to_five_digit_string(index + 1) + ".jpg"
    image = Image.open(file_name)
    return np.array(image)  # Convert the image to a numpy array


def display_image_with_pose(image_np, joints, bb=False):
    plt.imshow(image_np)

    # load pose x,y
    for x, y, o in joints:
        if o == 1:
            plt.scatter(x, y, c="blue", s=20)

    cmap = plt.get_cmap("rainbow", len(connections))

    for index, connection in enumerate(connections_indices):
        joints1_idx, joints2_idx = connection
        if joints[joints1_idx][2] == joints[joints2_idx][2] == 1:
            x = [joints[joints1_idx][0], joints[joints2_idx][0]]
            y = [joints[joints1_idx][1], joints[joints2_idx][1]]

            plt.plot(x, y, color=cmap(index), linewidth=2)

    if bb:
        # display bounding box
        bb = create_bounding_box_coords(joints)

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
        max_X = min(image_np.shape[1] - 1, min_X + width)
        max_Y = min(image_np.shape[0] - 1, min_Y + height)

        rect = patches.Rectangle(
            (min_X, min_Y),
            max_X - min_X,
            max_Y - min_Y,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    # Display the image using matplotlib

    plt.axis("off")  # Hide the axis
    plt.show()
