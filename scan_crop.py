#!/usr/bin/env python3
"""
scan-crop: Extract individual photos from scanned images containing multiple photos.

Usage:
    python scan_crop.py input.jpg [--output destination_folder]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys


def find_photo_contours(image):
    """
    Detect individual photos in the scanned image using contour detection.

    Args:
        image: OpenCV image (BGR format)

    Returns:
        List of bounding rectangles (x, y, w, h) for detected photos
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to separate photos from background
    # This works better for scanned photos with borders
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if needed (photos should be white/light in binary image)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (use RETR_TREE to get nested contours)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and aspect ratio
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.01   # Minimum 1% of image area
    max_area = img_area * 0.5    # Maximum 50% of image area
    bounding_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by aspect ratio (photos are typically between 1:2 and 2:1)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue

        # Filter by minimum dimensions (at least 50x50 pixels)
        if w < 50 or h < 50:
            continue

        bounding_boxes.append((x, y, w, h))

    return bounding_boxes


def crop_and_save_photos(input_path, output_dir):
    """
    Process scanned image and save individual cropped photos.

    Args:
        input_path: Path to input JPG file
        output_dir: Directory to save cropped photos
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Validate input
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Processing: {input_path}")
    image = cv2.imread(str(input_path))

    if image is None:
        print(f"Error: Could not read image '{input_path}'.", file=sys.stderr)
        return False

    # Detect photos
    print("Detecting individual photos...")
    bounding_boxes = find_photo_contours(image)

    if not bounding_boxes:
        print("Warning: No photos detected in the image.", file=sys.stderr)
        return False

    print(f"Found {len(bounding_boxes)} photo(s)")

    # Crop and save each photo
    base_name = input_path.stem
    for idx, (x, y, w, h) in enumerate(bounding_boxes, start=1):
        # Crop the photo
        cropped = image[y:y+h, x:x+w]

        # Save using Pillow for better JPEG quality control
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)

        output_path = output_dir / f"{base_name}_photo_{idx:02d}.jpg"
        pil_image.save(output_path, "JPEG", quality=95)
        print(f"  Saved: {output_path}")

    print(f"Successfully extracted {len(bounding_boxes)} photo(s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual photos from scanned images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scan_crop.py scan001.jpg
  python scan_crop.py scan001.jpg --output ./photos

For batch processing, use shell loops:
  for %f in (*.jpg) do python scan_crop.py "%f"  (Windows cmd)
  Get-ChildItem *.jpg | ForEach-Object { python scan_crop.py $_.Name }  (PowerShell)
        """
    )

    parser.add_argument(
        "input",
        help="Input JPG file containing scanned photos"
    )

    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for cropped photos (default: ./output)"
    )

    args = parser.parse_args()

    # Process the image
    success = crop_and_save_photos(args.input, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
