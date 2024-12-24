import cv2
import numpy as np


STATIC_OBJECT_NAME_TO_COLOR = {
    "DoomPlayer": (255, 255, 255),     # White

    # Weapons
    "Pistol": (64, 64, 64),            # Dark Gray
    "Shotgun": (139, 69, 19),          # Brown
    "SuperShotgun": (160, 82, 45),     # Darker Brown
    "Chaingun": (0, 0, 128),           # Navy Blue
    "RocketLauncher": (128, 0, 0),     # Maroon
    "PlasmaRifle": (0, 128, 128),      # Teal
    "BFG9000": (0, 255, 255),          # Cyan

    # Bonuses and Items
    "ArmorBonus": (0, 255, 0),         # Bright Green
    "BlurSphere": (255, 0, 255),       # Magenta
    "Allmap": (255, 255, 0),           # Yellow
    "Backpack": (153, 102, 51),        # Dark Beige
    "RadSuit": (0, 102, 102),          # Dark Teal
    "Infrared": (255, 165, 0),         # Orange

    # Enemies
    "CacoDemon": (255, 0, 0),          # Bright Red
    "LostSoul": (255, 255, 255),       # White
    "ZombieMan": (128, 128, 128),      # Gray
    "ShotgunGuy": (64, 0, 64),         # Purple
    "ChaingunGuy": (0, 64, 128),       # Dark Cyan
    "DoomImp": (139, 69, 19),          # Saddle Brown
    "Demon": (255, 105, 180),          # Pink
    "Revenant": (210, 180, 140),       # Tan

    # Environmental Hazards
    "ExplosiveBarrel": (255, 0, 0),    # Red

    # Health Items
    "HealthBonus": (0, 255, 0),        # Green
    "Stimpack": (192, 0, 0),           # Crimson
    "Medikit": (255, 0, 0),            # Red

    # Ammo
    "Shell": (255, 255, 0),            # Yellow
    "Cell": (0, 0, 255),               # Blue
    "RocketAmmo": (128, 0, 0),         # Maroon

    # Decorations and Environmental Objects
    "Stalagtite": (128, 128, 128),     # Light Gray
    "TorchTree": (0, 255, 150),        # Light Green
    "BigTree": (0, 200, 100),          # Dark Green
    "Gibs": (100, 200, 150),           # Light Brownish Green
    "BrainStem": (100, 0, 200),        # Purple
    "HeartColumn": (200, 0, 50),       # Deep Red
    "TechPillar": (150, 150, 255),     # Light Blue
    "ShortRedColumn": (255, 50, 50),   # Bright Red
    "ShortGreenColumn": (50, 255, 50), # Bright Green
    "RocketSmokeTrail": (128, 128, 128),# Smoke Gray
    "Rocket": (200, 0, 0),             # Bright Red
    "BulletPuff": (220, 220, 220),     # Light Gray
    "Blood": (150, 0, 0),              # Blood Red
    "GibbedMarine": (180, 50, 50),     # Dark Red
    "TechLamp": (100, 100, 255),       # Light Blue
    "SmallBloodPool": (80, 0, 0),      # Dark Blood Red
}

OBJECT_ID_TO_COLOR = {
    0: (0, 0, 0),
    1: (128, 128, 128),
    255: (255, 255, 255),
}

DEFAULT_COLOR = (200, 200, 200)

# This dict merges static + dynamic assignments
object_color_map = dict(STATIC_OBJECT_NAME_TO_COLOR)

unique_label_names = set()

def generate_unique_color() -> tuple:
    """Generate a new unique color not already in use by any object."""
    while True:
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        if color not in object_color_map.values():
            return color

def segment_and_draw_boxes(
        obs_rgb,
        state,
        skip_ids=(0, 1, 255),
        do_segment=True,
        do_boxes=True
):
    """
    Convert the original obs (RGB) to a BGR image with each object colored
    and optionally bounding boxes drawn. Displays object names, not just IDs.

    Args:
        obs_rgb (np.ndarray): Original observation in RGB format (H x W x 3).
        state: The current game state (has labels, labels_buffer).
        skip_ids (tuple): Label values to skip (e.g., background, agent).
        do_segment (bool): Whether to apply segmentation coloring.
        do_boxes (bool): Whether to draw bounding boxes around detected objects.

    Returns:
        final_bgr (np.ndarray): BGR image. Segmented if do_segment=True,
                                bounding boxes if do_boxes=True.
    """
    # 1) Convert RGB->BGR for OpenCV
    obs_bgr = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)

    # 2) Extract label data from the state
    label_buffer = state.labels_buffer.astype(np.uint8)
    labels = state.labels  # list of label objects with .object_name, .value, etc.

    # 3) Build a map from label value -> object name
    value_to_object_name = {}
    for label in labels:
        value_to_object_name[label.value] = label.object_name
        # If this object name is new, assign a unique color
        if label.object_name not in object_color_map:
            object_color_map[label.object_name] = generate_unique_color()

    # Depending on whether we want segmentation,
    # we'll work on a segmented version or just use obs_bgr as-is.
    if do_segment:
        # 4) Create a blank image to hold segmented colors
        segmented_bgr = np.zeros_like(obs_bgr)

        # 5) Find all unique label values
        unique_vals = np.unique(label_buffer)

        # 6) Colorize each unique label
        for val in unique_vals:
            obj_name = value_to_object_name.get(val, None)
            if obj_name is not None:
                # Known object name -> use object_color_map
                color = object_color_map[obj_name]
            else:
                # Possibly a floor/ceiling/agent ID or unknown
                color = OBJECT_ID_TO_COLOR.get(val, DEFAULT_COLOR)

            mask = (label_buffer == val).astype(np.uint8)

            # Apply color channel by channel
            for c in range(3):
                segmented_bgr[:, :, c] += (mask * color[c])

        final_bgr = segmented_bgr
    else:
        # No segmentation, work directly with the original BGR
        final_bgr = obs_bgr.copy()

    if do_boxes:
        # 7) Draw bounding boxes for each object (excluding skip_ids)
        unique_vals = np.unique(label_buffer)
        for val in unique_vals:
            if val in skip_ids:
                continue  # skip background, agent, etc.

            obj_name = value_to_object_name.get(val, "Unknown")
            # Create mask for the object
            mask = (label_buffer == val).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw rectangle
                cv2.rectangle(final_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put object name instead of ID
                cv2.putText(
                    final_bgr,
                    f"{obj_name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

    return final_bgr
