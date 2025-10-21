#!/usr/bin/env python3
"""Debug script to visualize the detection process."""

import cv2
import numpy as np
from pathlib import Path

def debug_detection(image_path):
    """Visualize the detection steps."""
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if needed
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Save binary image
    cv2.imwrite('debug_binary.jpg', binary)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite('debug_morph.jpg', morph)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours")

    # Draw all contours
    debug_img = image.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)

    # Analyze each contour
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.005
    max_area = img_area * 0.8

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        print(f"Contour {i}: area={area:.0f} ({area/img_area*100:.2f}%), "
              f"bbox=({x},{y},{w},{h}), aspect={aspect_ratio:.2f}")

        if min_area <= area <= max_area and 0.3 <= aspect_ratio <= 3.5 and w >= 50 and h >= 50:
            print(f"  -> ACCEPTED")
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        else:
            print(f"  -> REJECTED")

    cv2.imwrite('debug_contours.jpg', debug_img)
    print(f"\nDebug images saved: debug_binary.jpg, debug_morph.jpg, debug_contours.jpg")

if __name__ == "__main__":
    debug_detection(r"C:\Users\Remus\Downloads\Poze\scan-1.jpg")
