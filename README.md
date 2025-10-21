# scan-crop

Automatically extract individual photos from scanned images containing multiple photos.

If you've scanned multiple physical photos on a flatbed scanner in one pass, this tool will automatically detect and crop each photo into separate files.

## Installation

### Option 1: Download Pre-built Executable (Easiest)

1. Download `scan-crop.exe` from the releases page
2. Place it in a folder of your choice (e.g., `C:\Tools\scan-crop\`)
3. You're ready to go!

### Option 2: Install Python Version

If you have Python 3.11+ installed:

1. Download or clone this repository
2. Open Command Prompt or PowerShell in the project folder
3. Run the following commands:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a single scanned image:

```cmd
scan-crop.exe scan001.jpg
```

This will create an `output` folder and save the cropped photos there with names like:
- `scan001_photo_01.jpg`
- `scan001_photo_02.jpg`
- etc.

### Specify Output Folder

Save cropped photos to a specific folder:

```cmd
scan-crop.exe scan001.jpg --output C:\Photos\MyScans
```

### Process Multiple Files

**Using Command Prompt (cmd.exe):**
```cmd
for %f in (*.jpg) do scan-crop.exe "%f"
```

**Using PowerShell:**
```powershell
Get-ChildItem *.jpg | ForEach-Object { scan-crop.exe $_.Name }
```

**Using File Explorer:**
1. Open the folder containing your scanned images
2. Hold Shift and right-click in the folder
3. Select "Open PowerShell window here" or "Open Command window here"
4. Run the appropriate command above

## Tips for Best Results

1. **Scan Quality**: Higher resolution scans (300 DPI or more) work best
2. **Spacing**: Leave some space between photos on the scanner bed
3. **Contrast**: Ensure good contrast between photos and scanner background
4. **Orientation**: Photos can be in any orientation - the tool will detect them
5. **Overlapping**: Avoid overlapping photos if possible

## Troubleshooting

**"No photos detected"**
- Check that your scan has good contrast with the background
- Try increasing the scan resolution
- Ensure photos aren't too small (minimum ~100x100 pixels)

**Photos not cropped correctly**
- Ensure adequate spacing between photos
- Check for shadows or reflections on the scanner glass
- Clean the scanner glass before scanning

**"Could not read image"**
- Verify the file is a valid JPG image
- Check that the file path doesn't contain special characters
- Try converting the image to JPG if it's in another format

## How It Works

scan-crop uses computer vision to:
1. Convert the image to grayscale
2. Detect edges using the Canny algorithm
3. Find contours (outlines) of objects in the image
4. Filter contours by size to identify photos
5. Crop and save each detected photo as a separate file

## Python Version Usage

If you installed the Python version instead of the executable:

```cmd
python scan_crop.py scan001.jpg
python scan_crop.py scan001.jpg --output ./photos
```

## License

This project is free to use and modify.
