import cv2
import numpy as np


def add_bounding_boxes(self, screen, state):
    label_buffer = state.labels_buffer  # Per-pixel object ID
    # Normalize label buffer for processing
    label_buffer = label_buffer.astype(np.uint8)
    # Find unique object IDs in the label buffer
    unique_ids = np.unique(label_buffer)
    # Create a mask for each object and find contours
    for obj_id in unique_ids:
        if obj_id in [0, 1, 255]:  # Skip background, agent
            continue

        # Create a binary mask for the current object
        mask = (label_buffer == obj_id).astype(np.uint8)

        # Find contours of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the object
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

            # Optional: Add object ID as text
            cv2.putText(screen, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)