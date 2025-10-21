# scan-crop

[![Build and Release](https://github.com/rusanu/scan-crop/actions/workflows/build.yml/badge.svg)](https://github.com/rusanu/scan-crop/actions/workflows/build.yml)
[![Latest Release](https://img.shields.io/github/v/release/rusanu/scan-crop?label=latest%20release)](https://github.com/rusanu/scan-crop/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/rusanu/scan-crop/total?label=downloads)](https://github.com/rusanu/scan-crop/releases)

Interactive GUI application for extracting individual photos from scanned images.

If you've scanned multiple physical photos on a flatbed scanner in one pass, scan-crop will automatically detect each photo and let you adjust the crop regions, apply rotations, and export them as separate high-quality JPEG files.

## Features

- **Automatic photo detection** using computer vision
- **Interactive crop adjustment** with mouse drag and resize
- **Manual region creation** by dragging on the canvas
- **Rotation controls** (90° increments) for each photo
- **Live preview** of selected photo with transformations
- **Keyboard shortcuts** for power users
- **Batch export** of all photos at once
- **Command-line support** for drag-and-drop workflow

## Installation

### Download Standalone Executable (Recommended)

**No Python installation required!**

1. Download the latest `scan-crop.exe` from [Releases](https://github.com/rusanu/scan-crop/releases/latest)
2. Run it directly - no installation needed

### Run from Source

**Requirements:** Python 3.11 or higher

1. Clone this repository:
```cmd
git clone https://github.com/rusanu/scan-crop.git
cd scan-crop
```

2. Create a virtual environment and install dependencies:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
```cmd
python scan_crop_gui.py
```

## Usage

### Launch the Application

**Using the standalone executable:**
- Double-click `scan-crop.exe` to launch
- Or drag a JPEG image file onto `scan-crop.exe`
- Or run from command line: `scan-crop.exe path\to\scan.jpg`

**Running from source:**
```cmd
python scan_crop_gui.py              # Launch with file browser
python scan_crop_gui.py scan.jpg     # Open specific image
```

### Using the Interface

1. **Load an image** - Click "Open Image..." or launch with a file path
2. **Review detected regions** - Green boxes show automatically detected photos
3. **Adjust crop regions** - Click and drag regions to move, drag corners/edges to resize
4. **Create manual regions** - Drag on empty areas to create new crop regions
5. **Select a region** - Click any region to select it (shows in right preview panel)
6. **Rotate photos** - Use toolbar buttons or click region and use rotation controls
7. **Delete regions** - Select unwanted regions and click "Delete Region" or press Delete
8. **Export all** - Click "Export All Photos..." to save all regions to a folder

### Keyboard Shortcuts

- **Tab / Shift+Tab** - Cycle through detected regions
- **Arrow keys** - Move selected region (1 pixel)
- **Shift+Arrow keys** - Resize selected region (1 pixel)
- **Delete** - Delete selected region

For the complete shortcuts list, see **Help → Keyboard Shortcuts** in the application.

## Tips for Best Results

1. **Scan Quality**: Use 300 DPI or higher for best detection
2. **Spacing**: Leave space between photos on the scanner bed
3. **Contrast**: Ensure good contrast between photos and scanner background
4. **Lighting**: Avoid shadows and reflections on the scanner glass
5. **Orientation**: Photos can be in any orientation - adjust rotation in the app

## How It Works

scan-crop uses OpenCV computer vision to:
1. Convert the scanned image to grayscale
2. Apply binary thresholding using Otsu's method
3. Clean up noise with morphological operations
4. Find contours (outlines) of objects in the image
5. Filter contours by size, aspect ratio, and dimensions
6. Present detected regions in an interactive GUI for review and adjustment
7. Apply rotations and export each photo as a high-quality JPEG (95% quality)

## License

This project is free to use and modify.
