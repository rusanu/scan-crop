# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scan-crop is a Python CLI tool that processes scanned JPG images containing multiple physical photos and automatically crops out individual photos as separate files.

**Target Platform**: Windows
**Target Users**: Moderately technical users with no programming background

## Development Commands

### Setup Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows cmd
.\venv\Scripts\Activate.ps1  # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Running the Tool
```bash
# Basic usage
python scan_crop.py input.jpg

# With custom output directory
python scan_crop.py input.jpg --output ./photos

# Batch processing (PowerShell)
Get-ChildItem *.jpg | ForEach-Object { python scan_crop.py $_.Name }
```

### Testing
```bash
# Run with test images (create test_images/ folder with sample scans)
python scan_crop.py test_images/sample.jpg --output test_output
```

### Building Standalone Executable
```bash
# Install PyInstaller
pip install pyinstaller

# Build single-file executable
pyinstaller --onefile --name scan-crop scan_crop.py

# Output will be in dist/scan-crop.exe
```

## Architecture

### Core Components

**scan_crop.py** - Single-file CLI application with three main functions:

1. **find_photo_contours(image)**:
   - Uses OpenCV binary thresholding and contour finding
   - Applies Otsu's method for automatic threshold detection
   - Uses morphological operations (close/open) to clean up binary image
   - Filters contours by area (1-50% of image), aspect ratio (0.3-3.5), and minimum size (50x50px)
   - Returns bounding boxes for detected photos
   - Algorithm: Grayscale → Otsu Binary Threshold → Morphological Ops → Contour Detection (RETR_TREE)

2. **crop_and_save_photos(input_path, output_dir)**:
   - Orchestrates the full processing pipeline
   - Loads image with OpenCV (cv2)
   - Calls detection function
   - Crops each detected region
   - Saves using Pillow for quality control (95% JPEG quality)

3. **main()**:
   - Argument parsing with argparse
   - Default output directory: ./output
   - Exit codes: 0 for success, 1 for errors

### Image Processing Pipeline

```
Input JPG → OpenCV Load → Grayscale Conversion
    ↓
Otsu Binary Threshold → Auto-invert (if needed)
    ↓
Morphological Close (5x5, 2 iterations) → Morphological Open (5x5, 1 iteration)
    ↓
Find Contours (RETR_TREE) → Filter by Area/Aspect Ratio/Size
    ↓
Extract Bounding Boxes → Crop Each Region → Convert BGR to RGB → Save with Pillow
```

### Key Libraries

- **opencv-python (cv2)**: Image loading, binary thresholding, morphological operations, contour finding
- **Pillow (PIL)**: High-quality JPEG saving
- **numpy**: Array operations (dependency of OpenCV)

## Design Decisions

1. **Single-file CLI**: Deliberately simple to keep deployment straightforward
2. **No preview mode**: Users rely on shell commands for batch processing
3. **Binary threshold over edge detection**: Otsu's method automatically adapts to different scan qualities
4. **Hierarchical contours (RETR_TREE)**: Captures nested contours to find individual photos within scanned pages
5. **Fixed quality**: 95% JPEG quality balances size and fidelity
6. **Relative area thresholds**: Uses percentages (1-50% of image) rather than fixed pixel counts for better adaptability

## Development Guidelines

### When Modifying Detection Algorithm

- Adjust area percentages (`min_area = img_area * 0.01`) to change sensitivity
- Modify aspect ratio range (currently 0.3-3.5) to accept wider/narrower shapes
- Tune morphological kernel size (5x5) and iterations for different noise levels
- Adjust minimum dimensions (currently 50x50px) for very small or large photos

### When Adding Features

- Keep CLI interface simple and focused
- Maintain single-file structure for easy distribution
- Test with various scan qualities (different DPI, lighting, orientations)
- Ensure Windows path handling (use pathlib.Path)

### Error Handling

- Validate file existence before processing
- Check if image loads successfully
- Warn if no photos detected (don't fail silently)
- Use sys.stderr for error messages
- Return proper exit codes for scripting

### Distribution

- PyInstaller creates standalone .exe (no Python installation required)
- Include requirements.txt for development setup
- README.md should focus on end-user installation, not development
