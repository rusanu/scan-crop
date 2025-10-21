# scan-crop GUI Application Architecture

**Version**: 2.0
**Date**: 2025-10-21
**Framework**: tkinter

## Overview

This document defines the architecture for the scan-crop GUI application, transforming the CLI tool into an interactive photo cropper with preview, adjustable crops, rotation controls, and export functionality.

**UX Design**: Two-panel layout with source image (left) and photo list with live previews (right).

---

## Core Data Structures

### PhotoRegion (Data Class)

Represents a single detected or manually-created photo crop region.

```python
@dataclass
class PhotoRegion:
    """
    Represents a crop region with position, size, and processing parameters.
    All coordinates are in IMAGE space (original image pixels).
    """
    # Crop geometry (image space)
    x: int              # Top-left X coordinate (image pixels)
    y: int              # Top-left Y coordinate (image pixels)
    width: int          # Region width (image pixels)
    height: int         # Region height (image pixels)

    # Processing parameters (applied during preview and export)
    rotation: int = 0        # Rotation angle: 0, 90, 180, 270
    brightness: int = 0      # -100 to +100 (future enhancement)
    contrast: int = 0        # -100 to +100 (future enhancement)
    denoise: bool = False    # Noise reduction toggle (future enhancement)

    # Metadata
    is_manual: bool = False  # True if user-created, False if auto-detected
    region_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    list_order: int = 0      # Display order in right panel (for export sequencing)

    def to_bbox(self):
        """Returns (x, y, w, h) tuple for PIL crop"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

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

### Layout Overview

```
+------------------------------------------------------------------+
| PhotoCropperApp (Main Window)                                     |
+------------------------------------------------------------------+
| Menu Bar: File | Edit | Help                                      |
+------------------------------------------------------------------+
| PanedWindow (Horizontal Split)                                    |
|                                    |                              |
|  LEFT PANEL                        |  RIGHT PANEL                 |
|  SourceImageCanvas                 |  PhotoListPanel              |
|  (Original scan + crop overlays)   |  (Scrollable photo cards)    |
|                                    |                              |
+------------------------------------------------------------------+
```

### 1. Main Application (PhotoCropperApp)

**Responsibilities:**
- Create main window with two-panel layout
- Setup menu bar (File/Edit/Help)
- Coordinate between left and right panels
- Handle application lifecycle and file operations

**Layout Structure:**
```python
class PhotoCropperApp(tk.Tk):
    def __init__(self):
        # Main horizontal split
        self.paned_window = tk.PanedWindow(orient=tk.HORIZONTAL)

        # LEFT PANEL: Source image with crop overlays
        self.left_frame = tk.Frame(self.paned_window)
        self.source_canvas = SourceImageCanvas(self.left_frame, self.app_state)

        # RIGHT PANEL: Photo list with previews
        self.right_frame = tk.Frame(self.paned_window)
        self.photo_list = PhotoListPanel(self.right_frame, self.app_state)

        # Add to paned window (60/40 split)
        self.paned_window.add(self.left_frame, width=600)
        self.paned_window.add(self.right_frame, width=400)
```

**Key Methods:**
- `setup_menu()` - File (Open, Export), Edit (Add Region, Delete), Help
- `load_image(path)` - Load scan, run detection, update both panels
- `export_photos()` - Save all photos with transformations applied
- `sync_selection(region_id)` - Sync selection between left/right panels

---

### 2. SourceImageCanvas (Left Panel - Custom tk.Canvas)

**Responsibilities:**
- Display source scanned image (scaled to fit)
- Render crop region rectangles as overlays
- Handle mouse interactions for region manipulation
- Manage coordinate transformations (image space ↔ canvas space)
- Show selection highlights and resize handles

**State:**
- `app_state: ApplicationState` - Reference to global state
- `scale_factor: float` - Current image scaling
- `image_offset: (int, int)` - Offset for centering image
- `interaction_mode: InteractionMode` - Current mouse operation

**Key Methods:**
- `display_image()` - Render source image scaled to canvas
- `refresh()` - Redraw entire canvas (image + all region overlays)
- `draw_regions()` - Draw all crop region rectangles
- `draw_selection_handles()` - Draw resize handles for selected region
- `image_to_canvas(x, y)` - Convert image coords → canvas coords
- `canvas_to_image(x, y)` - Convert canvas coords → image coords
- `on_mouse_down(event)` - Handle click (select/start drag/resize)
- `on_mouse_drag(event)` - Handle drag (move/resize region)
- `on_mouse_up(event)` - Finalize operation, notify right panel to update

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

### 3. PhotoListPanel (Right Panel - Scrollable Container)

**Responsibilities:**
- Display scrollable list of photo cards (one per PhotoRegion)
- Manage PhotoItemWidget instances
- Handle reordering of photos
- Provide "Export All" button at bottom

**Layout:**
```python
class PhotoListPanel(tk.Frame):
    def __init__(self, parent, app_state):
        # Scrollable canvas for photo cards
        self.canvas = tk.Canvas(self)
        self.scrollbar = tk.Scrollbar(self, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Container for PhotoItemWidgets
        self.photo_widgets = []  # List of PhotoItemWidget instances

        # Export button at bottom
        self.export_btn = tk.Button(self, text="Export All Photos...",
                                    command=self.export_all)
```

**Key Methods:**
- `refresh()` - Rebuild list of photo cards from app_state.photo_regions
- `add_photo_card(region)` - Create PhotoItemWidget for region
- `remove_photo_card(region_id)` - Delete PhotoItemWidget
- `move_photo_up(region_id)` - Reorder region up in list
- `move_photo_down(region_id)` - Reorder region down in list
- `on_card_click(region_id)` - Sync selection with left panel

---

### 4. PhotoItemWidget (Individual Photo Card)

**Responsibilities:**
- Display thumbnail preview of one cropped photo
- Show rotation and enhancement controls
- Apply transformations to thumbnail in real-time
- Handle user interactions (rotate, delete, reorder)

**Layout:**
```
+------------------------+
| Photo 1                |
|  [Thumbnail 200x200]   |
|  Rotation: 0°          |
|  [⟲] [⟳]              |
|  Brightness: [slider]  |  (future)
|  Contrast: [slider]    |  (future)
|  [Delete] [↑] [↓]      |
+------------------------+
```

**Implementation:**
```python
class PhotoItemWidget(tk.Frame):
    def __init__(self, parent, region, app_state, photo_list_panel):
        self.region = region
        self.app_state = app_state
        self.photo_list = photo_list_panel

        # Thumbnail canvas
        self.thumbnail_canvas = tk.Canvas(self, width=200, height=200, bg="white")

        # Controls
        self.rotation_label = tk.Label(self, text=f"Rotation: {region.rotation}°")
        self.rotate_ccw_btn = tk.Button(self, text="⟲", command=self.rotate_ccw)
        self.rotate_cw_btn = tk.Button(self, text="⟳", command=self.rotate_cw)

        # Future enhancement controls (initially hidden or disabled)
        self.brightness_slider = tk.Scale(self, from_=-100, to=100,
                                          orient=tk.HORIZONTAL, label="Brightness")
        self.contrast_slider = tk.Scale(self, from_=-100, to=100,
                                        orient=tk.HORIZONTAL, label="Contrast")

        # Action buttons
        self.delete_btn = tk.Button(self, text="Delete", command=self.delete_region)
        self.move_up_btn = tk.Button(self, text="↑", command=self.move_up)
        self.move_down_btn = tk.Button(self, text="↓", command=self.move_down)

    def update_thumbnail(self):
        """
        Render thumbnail with current transformations.
        1. Crop from original image
        2. Apply rotation
        3. Apply brightness/contrast (future)
        4. Scale to thumbnail size
        5. Display in canvas
        """
        # Crop from original
        cropped = self.app_state.original_image.crop(self.region.to_bbox())

        # Apply rotation (PIL handles expand=True automatically)
        if self.region.rotation != 0:
            cropped = cropped.rotate(-self.region.rotation, expand=True)

        # Future: Apply enhancements
        # if self.region.brightness != 0:
        #     cropped = ImageEnhance.Brightness(cropped).enhance(1 + self.region.brightness/100)

        # Create thumbnail (maintains aspect ratio)
        cropped.thumbnail((200, 200), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        self.photo_image = ImageTk.PhotoImage(cropped)
        self.thumbnail_canvas.delete("all")
        self.thumbnail_canvas.create_image(100, 100, image=self.photo_image)

    def rotate_cw(self):
        """Rotate 90° clockwise"""
        self.region.rotation = (self.region.rotation + 90) % 360
        self.rotation_label.config(text=f"Rotation: {self.region.rotation}°")
        self.update_thumbnail()
        # Notify left panel to update rotation indicator
        self.photo_list.app.source_canvas.refresh()

    def rotate_ccw(self):
        """Rotate 90° counter-clockwise"""
        self.region.rotation = (self.region.rotation - 90) % 360
        self.rotation_label.config(text=f"Rotation: {self.region.rotation}°")
        self.update_thumbnail()
        self.photo_list.app.source_canvas.refresh()
```

---

### 5. DetectionEngine (Static Module)

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
    3. Convert bounding boxes → PhotoRegion(is_manual=False, list_order=idx)
    4. Return list
    """
```

*Note: Reuses existing `find_photo_contours()` from scan_crop.py*

---

### 6. RegionEditor (Helper Functions)

**Responsibilities:**
- Render selection handles on selected region (left panel)
- Calculate resize handle positions
- Determine interaction mode from mouse position

**Key Functions:**
```python
def draw_selection_handles(canvas, region, scale, offset):
    """
    Draw 8 resize handles (corners + midpoints) for selected region.
    Handles are small rectangles (e.g., 8x8 pixels) in canvas space.
    """

def get_interaction_mode(mouse_x, mouse_y, region, scale, offset):
    """
    Determine what interaction is happening based on mouse position:
    - On corner/edge handle? → RESIZING_XX
    - Inside region? → DRAGGING
    - Outside? → NONE

    Returns InteractionMode enum.
    """
```

*Note: These are utility functions used by SourceImageCanvas, not a separate component.*

---

### 7. Exporter (Static Module)

**Responsibilities:**
- Save all photo regions to files
- Apply crops, rotations, and enhancements during export
- Sort regions by list_order before export

**Key Functions:**
```python
def export_photos(image: Image.Image, regions: List[PhotoRegion], output_dir: Path, base_name: str):
    """
    Export all photo regions to output_dir.

    Process:
    1. Sort regions by list_order (respects user's reordering in right panel)
    2. For each region:
       a. Crop from original image: image.crop(region.to_bbox())
       b. Apply rotation if needed: cropped.rotate(-region.rotation, expand=True)
       c. Apply brightness/contrast if set (future)
       d. Apply denoise if enabled (future)
       e. Save as JPEG: cropped.save(path, "JPEG", quality=95)

    Filename format: {base_name}_photo_{idx:02d}.jpg
    Returns: List of saved file paths
    """

def apply_enhancements(image: Image.Image, region: PhotoRegion) -> Image.Image:
    """
    Apply enhancement parameters to image (future enhancement).

    Parameters:
    - brightness: -100 to +100
    - contrast: -100 to +100
    - denoise: bool

    Uses PIL ImageEnhance and OpenCV for processing.
    """
```

*Note: Rotation uses PIL.Image.rotate() with negative angle because PIL rotates counter-clockwise by default.*

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
  → Run detect_photos() → List[PhotoRegion]
  → Store regions in ApplicationState (with list_order = 0, 1, 2, ...)
  → LEFT PANEL: SourceImageCanvas displays image + region overlays
  → RIGHT PANEL: PhotoListPanel creates PhotoItemWidget for each region
    → Each widget renders thumbnail with current transformations
```

### 3. Left Panel Interaction: Adjust Crop Region

```
User drags corner of region on LEFT PANEL:
  → Mouse Down on region corner/edge
    → Determine interaction mode (RESIZING_NW/NE/SE/SW)
    → Store drag_start position
  → Mouse Drag
    → Calculate new region dimensions (in image space)
    → Update PhotoRegion.x, .y, .width, .height
    → Refresh left canvas (show updated rectangle)
  → Mouse Up
    → Finalize resize
    → Notify RIGHT PANEL: corresponding PhotoItemWidget.update_thumbnail()
    → Right panel shows updated crop
```

### 4. Right Panel Interaction: Rotate Photo

```
User clicks ⟳ button on Photo 2 card (RIGHT PANEL):
  → PhotoItemWidget.rotate_cw()
    → Update region.rotation = (rotation + 90) % 360
    → Update rotation label: "Rotation: 90°"
    → Call update_thumbnail()
      → Crop from original image
      → Apply rotation (PIL.rotate())
      → Render rotated thumbnail in card
    → Notify LEFT PANEL: SourceImageCanvas.refresh()
      → (Optional) Show rotation indicator on region overlay
```

### 5. Selection Sync Between Panels

```
SCENARIO A: User clicks region on LEFT PANEL
  → SourceImageCanvas.on_mouse_down()
    → app_state.select_region_at_point(x, y)
    → app_state.selected_region_id = clicked_region.id
    → LEFT: Draw selection handles
    → RIGHT: Highlight corresponding PhotoItemWidget (border/background change)

SCENARIO B: User clicks photo card on RIGHT PANEL
  → PhotoItemWidget.on_click()
    → app_state.selected_region_id = self.region.region_id
    → RIGHT: Highlight this card
    → LEFT: Draw selection handles on corresponding region
```

### 6. Reorder Photos (Right Panel)

```
User clicks [↓] on Photo 2 card:
  → PhotoItemWidget.move_down()
    → Swap region.list_order with Photo 3
    → PhotoListPanel.refresh()
      → Rebuild photo card list in new order
  → LEFT PANEL: Regions stay in same positions (not affected)
  → Export: Will save photos in new order (photo_01.jpg = old Photo 3, etc.)
```

### 7. Export Workflow

```
User: Clicks "Export All Photos..." button (RIGHT PANEL)
  → Open directory selection dialog
  → Call export_photos(image, regions, output_dir)
    → Sort regions by list_order (respects user reordering)
    → For each PhotoRegion (in order):
      → Crop from original image
      → Apply rotation if rotation != 0
      → Apply brightness/contrast if set (future)
      → Save as JPEG (95% quality)
    → Return list of saved paths
  → Show success message: "Saved N photos to {dir}"
```

---

## Design Decisions

### 1. Why Two-Panel Layout (LEFT=Source, RIGHT=Photo List)?

**Rationale:**
- **Separation of concerns**: Left = adjust crop geometry, Right = process individual photos
- **Live preview**: Users see rotation/enhancements immediately in thumbnails
- **No overlap issues**: Rotation happens in isolated thumbnails, not on main canvas
- **Natural workflow**: Crop → Process → Export
- **Extensibility**: Right panel can add more enhancement controls without cluttering

**Alternatives Considered:**
- **Modal dialog**: Requires extra click, interrupts workflow
- **In-place rotation**: Would cause bounding boxes to overlap on left canvas
- **Export-only**: No visual confirmation before saving

**Benefits for Vintage Photos:**
Right panel becomes processing workspace for per-photo adjustments (rotation, brightness, contrast, denoise).

---

### 2. Why Single-File Architecture?

**Rationale:** Keep deployment simple for Windows users. Single `.py` file → single `.exe` file.

**Trade-off:** Less modular, but acceptable for ~800-1200 lines of code (increased from original estimate due to two-panel design).

---

### 3. Why Store Coordinates in Image Space?

**Rationale:**
- Image coordinates are stable (don't change with window resize)
- Export operations use original image dimensions
- Canvas scaling is display-only concern
- Thumbnails crop from original image, not scaled canvas

**Alternative (rejected):** Store in canvas space → would need to recalculate all regions on resize.

---

### 4. Why PIL for Image Operations?

**Rationale:**
- Already using PIL for save (quality control)
- PIL.Image.rotate() is simpler than cv2.rotate()
- PIL.ImageEnhance for brightness/contrast (future)
- OpenCV only needed for detection (which works with numpy arrays)

---

### 5. Why Live Thumbnail Previews (vs Export-Only)?

**Decision: Live previews in right panel**

**Rationale:**
- User sees final result before export
- Critical for rotation (users need to know which way is "up")
- Enables experimentation with brightness/contrast
- Better UX for non-technical users

**Implementation:** Each PhotoItemWidget renders thumbnail with all transformations applied (crop → rotate → enhance).

**Performance:** PIL thumbnail operations fast enough for interactive use (~50-100ms per photo).

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
