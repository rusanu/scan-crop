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
import logging
from datetime import datetime
import traceback
import os


def setup_error_logging():
    """
    Setup logging to capture runtime errors and diagnostic information.
    Creates a log file in the same directory as the executable/script.
    """
    # Determine log file location (same directory as executable)
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        app_dir = Path(sys.executable).parent
    else:
        # Running as script
        app_dir = Path(__file__).parent

    log_file = app_dir / f"scan-crop-error-{datetime.now().strftime('%Y%m%d')}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stderr)
        ]
    )

    return logging.getLogger(__name__)


def log_opencv_diagnostics(logger):
    """
    Log OpenCV version and DLL location information for debugging.
    This helps diagnose DLL loading issues on different machines.
    """
    logger.info("=" * 60)
    logger.info("OpenCV Diagnostic Information")
    logger.info("=" * 60)

    # OpenCV version
    logger.info(f"OpenCV Version: {cv2.__version__}")

    # OpenCV file location
    try:
        opencv_path = cv2.__file__
        logger.info(f"OpenCV Module Path: {opencv_path}")
        logger.info(f"OpenCV Module Exists: {os.path.exists(opencv_path)}")
    except Exception as e:
        logger.error(f"Could not determine OpenCV path: {e}")

    # Check for DLL dependencies (Windows)
    if sys.platform == 'win32':
        try:
            opencv_dir = Path(cv2.__file__).parent
            dll_files = list(opencv_dir.glob('*.dll'))
            logger.info(f"DLL files in OpenCV directory: {len(dll_files)}")
            for dll in dll_files[:10]:  # Log first 10 DLLs
                logger.info(f"  - {dll.name} ({dll.stat().st_size} bytes)")
            if len(dll_files) > 10:
                logger.info(f"  ... and {len(dll_files) - 10} more")
        except Exception as e:
            logger.error(f"Could not enumerate DLLs: {e}")

    # Build information
    try:
        build_info = cv2.getBuildInformation()
        # Log key sections
        for line in build_info.split('\n'):
            if any(keyword in line.lower() for keyword in ['version', 'platform', 'compiler', 'python']):
                logger.info(f"  {line.strip()}")
    except Exception as e:
        logger.error(f"Could not get build information: {e}")

    # System information
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Frozen (PyInstaller): {getattr(sys, 'frozen', False)}")
    if getattr(sys, 'frozen', False):
        logger.info(f"Executable Path: {sys.executable}")
        logger.info(f"MEIPASS: {getattr(sys, '_MEIPASS', 'Not set')}")

    # Environment variables that might affect DLL loading
    logger.info("Relevant Environment Variables:")
    for var in ['PATH', 'PYTHONPATH', 'OPENCV_DIR']:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' and value != 'Not set':
            # Log PATH entries separately for readability
            logger.info(f"  {var}:")
            for path_entry in value.split(os.pathsep)[:5]:
                logger.info(f"    - {path_entry}")
            logger.info(f"    ... ({len(value.split(os.pathsep))} total entries)")
        else:
            logger.info(f"  {var}: {value}")

    logger.info("=" * 60)


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
    logger = logging.getLogger(__name__)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Validate input
    if not input_path.exists():
        logger.error(f"Input file '{input_path}' not found")
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Processing: {input_path}")
    logger.info(f"Loading image: {input_path}")

    try:
        image = cv2.imread(str(input_path))

        if image is None:
            logger.error(f"cv2.imread returned None for '{input_path}'")
            logger.info(f"File exists: {input_path.exists()}, Size: {input_path.stat().st_size} bytes")
            print(f"Error: Could not read image '{input_path}'.", file=sys.stderr)
            return False

        logger.info(f"Image loaded successfully: shape={image.shape}, dtype={image.dtype}")

    except Exception as e:
        logger.error(f"Exception during cv2.imread: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: Could not read image '{input_path}': {e}", file=sys.stderr)
        return False

    # Detect photos
    print("Detecting individual photos...")
    logger.info("Starting photo detection")

    try:
        bounding_boxes = find_photo_contours(image)
        logger.info(f"Detection complete: found {len(bounding_boxes)} bounding boxes")

    except Exception as e:
        logger.error(f"Exception during photo detection: {e}")
        logger.error(traceback.format_exc())
        print(f"Error during photo detection: {e}", file=sys.stderr)
        return False

    if not bounding_boxes:
        logger.warning("No photos detected in the image")
        print("Warning: No photos detected in the image.", file=sys.stderr)
        return False

    print(f"Found {len(bounding_boxes)} photo(s)")

    # Crop and save each photo
    base_name = input_path.stem
    for idx, (x, y, w, h) in enumerate(bounding_boxes, start=1):
        try:
            logger.info(f"Processing photo {idx}/{len(bounding_boxes)}: bbox=({x},{y},{w},{h})")

            # Crop the photo
            cropped = image[y:y+h, x:x+w]

            # Save using Pillow for better JPEG quality control
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)

            output_path = output_dir / f"{base_name}_photo_{idx:02d}.jpg"
            pil_image.save(output_path, "JPEG", quality=95)
            logger.info(f"Saved: {output_path}")
            print(f"  Saved: {output_path}")

        except Exception as e:
            logger.error(f"Exception while processing photo {idx}: {e}")
            logger.error(traceback.format_exc())
            print(f"  Error saving photo {idx}: {e}", file=sys.stderr)
            # Continue processing other photos

    print(f"Successfully extracted {len(bounding_boxes)} photo(s)")
    return True


def main():
    # Setup logging first
    logger = setup_error_logging()

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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with OpenCV diagnostics"
    )

    args = parser.parse_args()

    try:
        # Log diagnostic information if debug mode or on first run
        if args.debug:
            log_opencv_diagnostics(logger)

        logger.info(f"Starting scan-crop processing: {args.input}")

        # Process the image
        success = crop_and_save_photos(args.input, args.output)

        if success:
            logger.info("Processing completed successfully")
        else:
            logger.warning("Processing completed with warnings or errors")

        sys.exit(0 if success else 1)

    except Exception as e:
        # Log any uncaught exceptions with full diagnostics
        logger.error("=" * 60)
        logger.error("FATAL ERROR OCCURRED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error("Stack Trace:")
        logger.error(traceback.format_exc())

        # Always log OpenCV diagnostics on fatal errors
        logger.error("\nLogging OpenCV diagnostics due to fatal error:")
        try:
            log_opencv_diagnostics(logger)
        except Exception as diag_error:
            logger.error(f"Could not log diagnostics: {diag_error}")

        print(f"\nFATAL ERROR: {str(e)}", file=sys.stderr)
        print(f"Error details have been logged. Please check the log file.", file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
