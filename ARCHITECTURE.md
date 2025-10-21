# scan-crop GUI Application Architecture

**Version**: 1.0
**Date**: 2025-10-21
**Framework**: tkinter

## Overview

This document defines the architecture for the scan-crop GUI application, transforming the CLI tool into an interactive photo cropper with preview, adjustable crops, rotation controls, and export functionality.

---

## Core Data Structures

### PhotoRegion (Data Class)

Represents a single detected or manually-created photo crop region.

```python
@dataclass
class PhotoRegion:
    """
    Represents a crop region with position, size, and rotation.
    All coordinates are in IMAGE space (original image pixels).
    """
    x: int              # Top-left X coordinate (image pixels)
    y: int              # Top-left Y coordinate (image pixels)
    width: int          # Region width (image pixels)
    height: int         # Region height (image pixels)
    rotation: int = 0   # Rotation angle: 0, 90, 180, 270
    is_manual: bool = False  # True if user-created, False if auto-detected
    region_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_bbox(self):
        """Returns (x, y, w, h) tuple"""
        return (self.x, self.y, self.width, self.height)

    def contains_point(self, px, py):
        """Check if point is inside region (used for selection)"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)
```

### ApplicationState

Central state manager for the entire application.

```python
class ApplicationState:
    """
    Manages all application state including image, regions, selection.
    Single source of truth for the application.
    """
    def __init__(self):
        self.image_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None  # PIL Image
        self.photo_regions: List[PhotoRegion] = []
        self.selected_region_id: Optional[str] = None
        self.output_directory: Path = Path("./output")

    def add_region(self, region: PhotoRegion):
        """Add a new photo region"""
        self.photo_regions.append(region)

    def remove_region(self, region_id: str):
        """Delete a photo region"""
        self.photo_regions = [r for r in self.photo_regions
                             if r.region_id != region_id]

    def get_selected_region(self) -> Optional[PhotoRegion]:
        """Get the currently selected region"""
        if not self.selected_region_id:
            return None
        for region in self.photo_regions:
            if region.region_id == self.selected_region_id:
                return region
        return None

    def select_region_at_point(self, x, y) -> Optional[PhotoRegion]:
        """Select region containing point (x, y in image coords)"""
        for region in reversed(self.photo_regions):  # Top to bottom
            if region.contains_point(x, y):
                self.selected_region_id = region.region_id
                return region
        self.selected_region_id = None
        return None
```

---

## Component Architecture

### 1. Main Application (PhotoCropperApp)

**Responsibilities:**
- Create main window and menu bar
- Coordinate between components
- Handle high-level application lifecycle

**Key Methods:**
- `__init__()` - Initialize UI and components
- `setup_menu()` - Create File/Edit/Help menus
- `run()` - Start main event loop

---

### 2. ImageCanvas (Custom tk.Canvas)

**Responsibilities:**
- Display scaled image
- Render photo region overlays
- Handle all mouse interactions
- Manage coordinate transformations

**State:**
- `app_state: ApplicationState` - Reference to global state
- `scale_factor: float` - Current image scaling
- `image_offset: (int, int)` - Offset for centering image

**Key Methods:**
- `load_and_display_image(image_path)` - Load image and auto-detect regions
- `refresh()` - Redraw canvas (image + regions)
- `image_to_canvas(x, y)` - Convert image coords → canvas coords
- `canvas_to_image(x, y)` - Convert canvas coords → image coords
- `draw_regions()` - Render all PhotoRegion overlays
- `on_mouse_down(event)` - Handle click (selection/drag start)
- `on_mouse_drag(event)` - Handle drag (move/resize)
- `on_mouse_up(event)` - Handle release (finalize edit)

**Mouse Interaction States:**
```python
class InteractionMode(Enum):
    NONE = 0           # No interaction
    DRAGGING = 1       # Moving entire region
    RESIZING_NW = 2    # Resizing from northwest corner
    RESIZING_NE = 3    # Resizing from northeast corner
    RESIZING_SE = 4    # Resizing from southeast corner
    RESIZING_SW = 5    # Resizing from southwest corner
    DRAWING_NEW = 6    # Drawing new manual region
```

---

### 3. DetectionEngine (Static Module)

**Responsibilities:**
- Run OpenCV contour detection (existing algorithm)
- Convert detection results to PhotoRegion objects

**Key Functions:**
```python
def detect_photos(image: Image.Image) -> List[PhotoRegion]:
    """
    Run detection algorithm on PIL Image.
    Returns list of PhotoRegion objects (auto-detected).

    Implementation:
    1. Convert PIL → OpenCV (BGR)
    2. Run find_photo_contours() (existing code)
    3. Convert bounding boxes → PhotoRegion(is_manual=False)
    4. Return list
    """
```

*Note: Reuses existing `find_photo_contours()` from scan_crop.py*

---

### 4. RegionEditor (UI Component)

**Responsibilities:**
- Render selection handles on selected region
- Calculate resize handle positions
- Determine interaction mode from mouse position

**Key Methods:**
```python
def draw_selection_handles(canvas, region, scale):
    """
    Draw 8 resize handles (corners + midpoints) for selected region.
    Handles are small rectangles (e.g., 8x8 pixels).
    """

def get_interaction_mode(mouse_x, mouse_y, region, scale):
    """
    Determine what interaction is happening based on mouse position:
    - On corner handle? → RESIZING_XX
    - Inside region? → DRAGGING
    - Outside? → NONE

    Returns InteractionMode enum.
    """
```

---

### 5. RotationController (UI Component)

**Responsibilities:**
- Provide rotation UI for selected region
- Update PhotoRegion.rotation state

**UI Options (Choose One):**

**Option A: Toolbar Buttons**
```python
# Add buttons to toolbar when region selected
rotate_cw_btn = tk.Button(toolbar, text="⟳ 90°", command=self.rotate_cw)
rotate_ccw_btn = tk.Button(toolbar, text="⟲ 90°", command=self.rotate_ccw)
```

**Option B: Keyboard Shortcuts** (Preferred - simpler UI)
```python
# Bind keys globally
root.bind("r", lambda e: self.rotate_selected_region(90))
root.bind("R", lambda e: self.rotate_selected_region(-90))  # Shift+R
```

**Rotation Logic:**
```python
def rotate_selected_region(angle_delta):
    """
    Rotate selected region by angle_delta (-90 or 90).
    Clamp result to [0, 90, 180, 270].
    """
    region = app_state.get_selected_region()
    if region:
        region.rotation = (region.rotation + angle_delta) % 360
        canvas.refresh()
```

---

### 6. Exporter (Static Module)

**Responsibilities:**
- Save all photo regions to files
- Apply crops and rotations during export

**Key Functions:**
```python
def export_photos(image: Image.Image, regions: List[PhotoRegion], output_dir: Path):
    """
    Export all photo regions to output_dir.

    For each region:
    1. Crop from original image: image.crop((x, y, x+w, y+h))
    2. Apply rotation if needed: cropped.rotate(region.rotation, expand=True)
    3. Save as JPEG: cropped.save(path, "JPEG", quality=95)

    Filename format: {original_stem}_photo_{idx:02d}.jpg
    """
```

*Note: Reuses PIL rotation and save logic. PIL.Image.rotate() handles rotation correctly.*

---

## Coordinate System Architecture

### Two Coordinate Spaces

1. **Image Space** (Original)
   - Original image dimensions (e.g., 3000x2000 pixels)
   - PhotoRegion coordinates stored in this space
   - Used for export/save operations

2. **Canvas Space** (Display)
   - Scaled to fit window (e.g., 800x600 pixels)
   - Used for rendering and mouse events
   - Changes when window resizes

### Transformation Functions

```python
class ImageCanvas:
    def __init__(self):
        self.scale_factor = 1.0
        self.image_offset = (0, 0)  # (offset_x, offset_y)

    def calculate_scale_and_offset(self):
        """
        Called after image load or window resize.
        Calculates scale_factor and image_offset to fit image in canvas.
        """
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        img_w, img_h = self.app_state.original_image.size

        # Calculate scale (fit to canvas, never upscale)
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale_factor = min(scale_w, scale_h, 1.0)

        # Calculate centering offset
        scaled_w = img_w * self.scale_factor
        scaled_h = img_h * self.scale_factor
        self.image_offset = (
            (canvas_w - scaled_w) // 2,
            (canvas_h - scaled_h) // 2
        )

    def image_to_canvas(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        canvas_x = img_x * self.scale_factor + self.image_offset[0]
        canvas_y = img_y * self.scale_factor + self.image_offset[1]
        return (canvas_x, canvas_y)

    def canvas_to_image(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        img_x = (canvas_x - self.image_offset[0]) / self.scale_factor
        img_y = (canvas_y - self.image_offset[1]) / self.scale_factor
        return (int(img_x), int(img_y))
```

**Critical Rule:**
- **ALL PhotoRegion coordinates are in IMAGE space**
- **Convert to canvas space only for rendering and mouse handling**
- **Convert mouse events to image space immediately for state updates**

---

## File Structure

```
scan-crop/
├── scan_crop.py              # Original CLI (keep for reference)
├── scan_crop_gui.py          # NEW: Main GUI application
│   ├── class PhotoRegion
│   ├── class ApplicationState
│   ├── class ImageCanvas(tk.Canvas)
│   ├── class PhotoCropperApp(tk.Tk)
│   ├── def detect_photos()        # From original
│   ├── def export_photos()
│   └── def main()
│
├── requirements.txt          # Update with any new deps (none expected)
├── ARCHITECTURE.md           # This file
├── FRAMEWORK_DECISION.md
├── poc_tkinter.py            # POC (can delete after implementation)
└── README.md                 # Update for GUI usage

```

**Decision: Single-file application** (maintain simplicity for Windows users)

---

## Event Flow Diagrams

### 1. Application Startup

```
User runs scan_crop_gui.py
  → Create ApplicationState()
  → Create PhotoCropperApp (main window)
    → Setup menu bar
    → Create ImageCanvas
    → Bind mouse events
  → Show window (empty canvas with "File → Open" instruction)
```

### 2. Load Image Workflow

```
User: File → Open Image
  → Open file dialog
  → Load image into ApplicationState.original_image (PIL)
  → ImageCanvas.load_and_display_image()
    → Run detect_photos() → List[PhotoRegion]
    → Store regions in ApplicationState
    → Calculate scale and offset
    → Refresh canvas (draw image + regions)
```

### 3. Mouse Interaction: Move Region

```
Mouse Down on region
  → canvas_to_image() to get image coordinates
  → app_state.select_region_at_point(x, y)
  → Store drag_start position
  → Refresh (show selection handles)

Mouse Drag
  → Calculate delta: (current - drag_start)
  → Update selected_region.x, .y (in image space)
  → Refresh canvas

Mouse Up
  → Clear drag_start
  → Refresh (finalize)
```

### 4. Export Workflow

```
User: File → Save Photos
  → Open directory selection dialog
  → Call export_photos(image, regions, output_dir)
    → For each PhotoRegion:
      → Crop from original image
      → Rotate if rotation != 0
      → Save as JPEG (95% quality)
  → Show success message: "Saved N photos to {dir}"
```

---

## Design Decisions

### 1. Why Single-File Architecture?

**Rationale:** Keep deployment simple for Windows users. Single `.py` file → single `.exe` file.

**Trade-off:** Less modular, but acceptable for ~500-800 lines of code.

### 2. Why Store Coordinates in Image Space?

**Rationale:**
- Image coordinates are stable (don't change with window resize)
- Export operations use original image dimensions
- Canvas scaling is display-only concern

**Alternative (rejected):** Store in canvas space → would need to recalculate all regions on resize.

### 3. Why PIL for Image Operations?

**Rationale:**
- Already using PIL for save (quality control)
- PIL.Image.rotate() is simpler than cv2.rotate()
- OpenCV only needed for detection (which works with numpy arrays)

### 4. Rotation: Preview vs Export-Only?

**Decision: Export-only** (simpler implementation)

**Rationale:**
- Rotating canvas preview requires complex rendering
- Most users care about final output, not preview accuracy
- Can add preview later if needed (future enhancement)

**Implementation:** Show rotation angle as text overlay on region (e.g., "90°")

---

## Technical Constraints

1. **No external dependencies beyond current requirements.txt**
   - tkinter (built-in)
   - OpenCV (already required)
   - Pillow (already required)
   - numpy (already required)

2. **Windows-first, but keep code cross-platform compatible**
   - Use `Path` for all file operations
   - Avoid Windows-specific APIs

3. **PyInstaller must build single-file .exe**
   - No data files or resources
   - All code in single Python file

---

## Next Steps (Implementation Order)

1. **scan-crop-4**: Implement basic GUI shell
   - PhotoCropperApp skeleton
   - Menu bar
   - Empty ImageCanvas
   - File open dialog

2. **scan-crop-5**: Integrate detection
   - Port find_photo_contours()
   - Implement detect_photos()
   - Display regions as overlays

3. **scan-crop-6**: Interactive editing
   - Region selection
   - Drag to move
   - Resize handles

4. **scan-crop-7**: Rotation controls
   - Keyboard shortcuts
   - Update region rotation state
   - Visual indicator

5. **scan-crop-8**: Export workflow
   - Implement export_photos()
   - Directory selection dialog
   - Apply crops and rotations

---

## Open Questions / Future Enhancements

1. **Undo/Redo?** - Not critical for v1, but useful
2. **Zoom in/out?** - Current design assumes "fit to window", could add zoom later
3. **Rotation preview rendering?** - More work, defer to v2
4. **Batch processing?** - Load multiple scans? (defer to v2)
5. **Detection parameter tuning UI?** - Allow user to adjust threshold? (defer)

---

**End of Architecture Document**
