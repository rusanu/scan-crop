#!/usr/bin/env python3
"""
scan-crop GUI: Interactive photo cropper for scanned images
Extracts individual photos from scans with adjustable crops and rotation controls
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import uuid
import sys
import argparse

from PIL import Image, ImageTk
import cv2
import numpy as np
from enum import Enum

# Import version info
try:
    from version import __version__
except ImportError:
    __version__ = "dev"


# ============================================================================
# Enums and Constants
# ============================================================================

class InteractionMode(Enum):
    """Mouse interaction modes for canvas"""
    NONE = 0
    DRAGGING = 1          # Moving entire region
    RESIZING_NW = 2       # Resizing from northwest corner
    RESIZING_N = 3        # Resizing from north edge
    RESIZING_NE = 4       # Resizing from northeast corner
    RESIZING_E = 5        # Resizing from east edge
    RESIZING_SE = 6       # Resizing from southeast corner
    RESIZING_S = 7        # Resizing from south edge
    RESIZING_SW = 8       # Resizing from southwest corner
    RESIZING_W = 9        # Resizing from west edge
    DRAWING_NEW = 10      # Drawing new region


# ============================================================================
# Detection Engine
# ============================================================================

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


def detect_photos(pil_image: Image.Image) -> List:
    """
    Run detection algorithm on PIL Image.
    Returns list of PhotoRegion objects (auto-detected).

    Args:
        pil_image: PIL Image object

    Returns:
        List of PhotoRegion objects with detected crop regions
    """
    # Convert PIL to OpenCV (BGR)
    img_array = np.array(pil_image)
    # PIL uses RGB, OpenCV uses BGR
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        cv_image = img_array

    # Run detection
    bounding_boxes = find_photo_contours(cv_image)

    # Convert to PhotoRegion objects
    # Import PhotoRegion here to avoid circular import
    from dataclasses import dataclass, field

    regions = []
    for idx, (x, y, w, h) in enumerate(bounding_boxes, start=1):
        region = PhotoRegion(
            x=x, y=y, width=w, height=h,
            rotation=0,
            is_manual=False,
            list_order=idx - 1,
            name=f"Photo {idx}"  # Set default name immediately
        )
        regions.append(region)

    return regions


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PhotoRegion:
    """
    Represents a crop region with position, size, and processing parameters.
    All coordinates are in IMAGE space (original image pixels).
    """
    # Crop geometry (image space)
    x: int
    y: int
    width: int
    height: int

    # Processing parameters (applied during preview and export)
    rotation: int = 0        # Rotation angle: 0, 90, 180, 270
    brightness: int = 0      # -100 to +100 (future enhancement)
    contrast: int = 0        # -100 to +100 (future enhancement)
    denoise: bool = False    # Noise reduction toggle (future enhancement)

    # Metadata
    is_manual: bool = False  # True if user-created, False if auto-detected
    region_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    list_order: int = 0      # Display order in right panel (for export sequencing)
    name: str = ""           # User-editable name for export filename

    def to_bbox(self):
        """Returns (x1, y1, x2, y2) tuple for PIL crop"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def contains_point(self, px, py):
        """Check if point is inside region (used for selection)"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


@dataclass
class AppConfig:
    """Application configuration/preferences"""
    overlay_color: str = "green"  # Color for region overlays: "green" or "red"
    export_format: str = "auto"   # Export format: "auto" (same as source), "jpeg", "png"
    filename_include_source: bool = True  # Include source filename in exports


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
        self.source_format: str = "JPEG"  # Track source image format for export
        self.config: AppConfig = AppConfig()  # Application preferences

    def add_region(self, region: PhotoRegion):
        """Add a new photo region"""
        self.photo_regions.append(region)

    def remove_region(self, region_id: str):
        """Delete a photo region"""
        self.photo_regions = [r for r in self.photo_regions
                             if r.region_id != region_id]
        if self.selected_region_id == region_id:
            self.selected_region_id = None

    def get_selected_region(self) -> Optional[PhotoRegion]:
        """Get the currently selected region"""
        if not self.selected_region_id:
            return None
        for region in self.photo_regions:
            if region.region_id == self.selected_region_id:
                return region
        return None

    def get_region_by_id(self, region_id: str) -> Optional[PhotoRegion]:
        """Get region by ID"""
        for region in self.photo_regions:
            if region.region_id == region_id:
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


# ============================================================================
# Left Panel: Source Image Canvas
# ============================================================================

class SourceImageCanvas(tk.Canvas):
    """
    Left panel: Displays source scanned image with crop region overlays.
    Handles mouse interactions for region manipulation.
    """
    def __init__(self, parent, app_state, app):
        super().__init__(parent, bg="gray", highlightthickness=0)
        self.app_state = app_state
        self.app = app
        self.scale_factor = 1.0
        self.image_offset = (0, 0)
        self.photo_image = None  # Keep reference to prevent GC

        # Interaction state
        self.interaction_mode = InteractionMode.NONE
        self.drag_start = None  # (x, y) in image space
        self.drag_region_start = None  # Original (x, y, w, h) of region being edited

        # Bind events
        self.bind("<Configure>", self.on_resize)
        self.bind("<Button-1>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<Motion>", self.on_mouse_move)

    def on_resize(self, event):
        """Handle canvas resize"""
        if self.app_state.original_image:
            self.refresh()

    def display_image(self):
        """Display the source image scaled to fit canvas"""
        if not self.app_state.original_image:
            return

        # Calculate scale and offset
        self.calculate_scale_and_offset()

        # Scale image
        img_w, img_h = self.app_state.original_image.size
        scaled_w = int(img_w * self.scale_factor)
        scaled_h = int(img_h * self.scale_factor)

        if scaled_w > 0 and scaled_h > 0:
            scaled_image = self.app_state.original_image.resize(
                (scaled_w, scaled_h), Image.Resampling.LANCZOS
            )
            self.photo_image = ImageTk.PhotoImage(scaled_image)

            # Draw image
            self.delete("all")
            self.create_image(
                self.image_offset[0], self.image_offset[1],
                anchor=tk.NW, image=self.photo_image
            )

    def calculate_scale_and_offset(self):
        """Calculate scale factor and offset to fit image in canvas"""
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return  # Canvas not yet sized

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

    def refresh(self):
        """Redraw entire canvas (image + regions)"""
        self.display_image()
        self.draw_regions()

    def on_mouse_down(self, event):
        """Handle mouse click on canvas"""
        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image(event.x, event.y)

        # Check if clicking on a resize handle first
        selected_region = self.app_state.get_selected_region()
        if selected_region:
            mode = self.get_handle_at_point(event.x, event.y, selected_region)
            if mode != InteractionMode.NONE:
                # Starting resize operation
                self.interaction_mode = mode
                self.drag_start = (img_x, img_y)
                self.drag_region_start = (selected_region.x, selected_region.y,
                                         selected_region.width, selected_region.height)
                return

        # Try to select region at this point
        selected = self.app_state.select_region_at_point(img_x, img_y)

        if selected:
            # Start drag operation (might become DRAGGING if user drags)
            self.interaction_mode = InteractionMode.DRAGGING
            self.drag_start = (img_x, img_y)
            self.drag_region_start = (selected.x, selected.y, selected.width, selected.height)
        else:
            # Click outside any region - deselect for now
            # Will switch to DRAWING_NEW if user starts dragging (detected in on_mouse_drag)
            self.interaction_mode = InteractionMode.NONE
            self.drag_start = (img_x, img_y)  # Store start point for potential new region
            self.drag_region_start = None
            self.app_state.selected_region_id = None

        # Refresh display
        self.refresh()

        # Sync with right panel
        if self.app:
            self.app.sync_selection()

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.drag_start:
            return

        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image(event.x, event.y)
        dx = img_x - self.drag_start[0]
        dy = img_y - self.drag_start[1]

        # If mode is NONE, check if user is starting to drag outside regions
        if self.interaction_mode == InteractionMode.NONE:
            # Add drag threshold to avoid accidental small drags (10 pixels)
            drag_distance = (dx**2 + dy**2) ** 0.5
            if drag_distance < 10:
                return  # Ignore small movements

            # User started dragging outside any region - enter DRAWING_NEW mode
            self.interaction_mode = InteractionMode.DRAWING_NEW
            self.new_region_preview = None  # Will be drawn in draw_regions

        if self.interaction_mode == InteractionMode.DRAWING_NEW:
            # Draw preview of new region being created
            # The preview will be rendered in draw_regions()
            self.refresh()
            return

        selected_region = self.app_state.get_selected_region()
        if not selected_region or not self.drag_region_start:
            return

        orig_x, orig_y, orig_w, orig_h = self.drag_region_start

        if self.interaction_mode == InteractionMode.DRAGGING:
            # Move entire region
            selected_region.x = max(0, orig_x + dx)
            selected_region.y = max(0, orig_y + dy)

        else:
            # Resizing - update based on which handle
            self.resize_region(selected_region, orig_x, orig_y, orig_w, orig_h, dx, dy)

        # Refresh display
        self.refresh()

    def on_mouse_up(self, event):
        """Handle mouse release"""
        if self.interaction_mode == InteractionMode.DRAWING_NEW and self.drag_start:
            # Create new region
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            start_x, start_y = self.drag_start

            # Calculate bounding box (handle dragging in any direction)
            x = min(start_x, img_x)
            y = min(start_y, img_y)
            width = abs(img_x - start_x)
            height = abs(img_y - start_y)

            # Only create if dimensions are valid (min 50x50)
            if width >= 50 and height >= 50:
                # Get next list order and photo number
                max_order = max([r.list_order for r in self.app_state.photo_regions], default=-1)
                photo_num = len(self.app_state.photo_regions) + 1

                # Create new PhotoRegion
                new_region = PhotoRegion(
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                    rotation=0,
                    brightness=0,
                    contrast=0,
                    denoise=False,
                    is_manual=True,
                    list_order=max_order + 1,
                    name=f"Photo {photo_num}"  # Set default name
                )

                # Add to state and select it
                self.app_state.photo_regions.append(new_region)
                self.app_state.selected_region_id = new_region.region_id

                # Clear interaction mode BEFORE refreshing
                self.interaction_mode = InteractionMode.NONE
                self.drag_start = None
                self.drag_region_start = None

                # Refresh both panels
                if self.app:
                    self.app.sync_selection()

                self.refresh()
                return  # Exit early to avoid clearing mode twice

        elif self.interaction_mode != InteractionMode.NONE:
            # Update preview in right panel after drag/resize
            if self.app:
                self.app.photo_preview.refresh()

        self.interaction_mode = InteractionMode.NONE
        self.drag_start = None
        self.drag_region_start = None

    def on_mouse_move(self, event):
        """Handle mouse move (update cursor)"""
        if self.interaction_mode != InteractionMode.NONE:
            return  # Don't change cursor during drag

        # Check if hovering over resize handle
        selected_region = self.app_state.get_selected_region()
        if selected_region:
            mode = self.get_handle_at_point(event.x, event.y, selected_region)
            self.update_cursor(mode)
        else:
            self.config(cursor="")

    def draw_regions(self):
        """Draw all crop region rectangles"""
        for region in self.app_state.photo_regions:
            # Convert to canvas coordinates
            x1, y1 = self.image_to_canvas(region.x, region.y)
            x2, y2 = self.image_to_canvas(region.x + region.width,
                                           region.y + region.height)

            # Determine color based on preferences and selection state
            is_selected = (region.region_id == self.app_state.selected_region_id)
            if self.app_state.config.overlay_color == "green":
                # Green for selected, red for unselected
                color = "green" if is_selected else "red"
            else:
                # Red for selected, green for unselected
                color = "red" if is_selected else "green"
            width = 3 if is_selected else 2

            # Draw rectangle
            self.create_rectangle(x1, y1, x2, y2,
                                outline=color, width=width,
                                tags=f"region_{region.region_id}")

            # Draw resize handles for selected region
            if is_selected:
                self.draw_resize_handles(x1, y1, x2, y2)

            # Show name label at top-left of region
            if region.name:
                # Position text just below top border
                text_x = x1 + 5
                text_y = y1 + 12
                # Draw text with background for better visibility
                self.create_text(text_x, text_y, text=region.name,
                               fill=color, font=("Arial", 10, "bold"),
                               anchor=tk.W,
                               tags=f"region_{region.region_id}")

            # Show rotation indicator if rotated (center of region)
            if region.rotation != 0:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self.create_text(cx, cy, text=f"{region.rotation}°",
                               fill=color, font=("Arial", 12, "bold"),
                               tags=f"region_{region.region_id}")

        # Draw preview rectangle for new region being drawn
        if self.interaction_mode == InteractionMode.DRAWING_NEW and self.drag_start:
            # Get current mouse position from the last event
            canvas_start_x, canvas_start_y = self.image_to_canvas(self.drag_start[0], self.drag_start[1])

            # Get current cursor position
            mouse_x = self.winfo_pointerx() - self.winfo_rootx()
            mouse_y = self.winfo_pointery() - self.winfo_rooty()

            # Draw dashed preview rectangle
            self.create_rectangle(canvas_start_x, canvas_start_y, mouse_x, mouse_y,
                                outline="blue", width=2, dash=(5, 5),
                                tags="preview_region")

    def draw_resize_handles(self, x1, y1, x2, y2):
        """Draw 8 resize handles on selected region"""
        handle_size = 8
        handles = [
            (x1, y1),           # NW corner
            ((x1+x2)/2, y1),    # N midpoint
            (x2, y1),           # NE corner
            (x2, (y1+y2)/2),    # E midpoint
            (x2, y2),           # SE corner
            ((x1+x2)/2, y2),    # S midpoint
            (x1, y2),           # SW corner
            (x1, (y1+y2)/2),    # W midpoint
        ]

        for hx, hy in handles:
            self.create_rectangle(
                hx - handle_size/2, hy - handle_size/2,
                hx + handle_size/2, hy + handle_size/2,
                fill="white", outline="green", width=2,
                tags="handle"
            )

    def get_handle_at_point(self, canvas_x, canvas_y, region):
        """Check if point is on a resize handle. Returns InteractionMode."""
        # Convert region to canvas coords
        x1, y1 = self.image_to_canvas(region.x, region.y)
        x2, y2 = self.image_to_canvas(region.x + region.width, region.y + region.height)

        handle_size = 8
        tolerance = handle_size / 2

        # Check each handle position
        handles = [
            ((x1, y1), InteractionMode.RESIZING_NW),
            (((x1+x2)/2, y1), InteractionMode.RESIZING_N),
            ((x2, y1), InteractionMode.RESIZING_NE),
            ((x2, (y1+y2)/2), InteractionMode.RESIZING_E),
            ((x2, y2), InteractionMode.RESIZING_SE),
            (((x1+x2)/2, y2), InteractionMode.RESIZING_S),
            ((x1, y2), InteractionMode.RESIZING_SW),
            ((x1, (y1+y2)/2), InteractionMode.RESIZING_W),
        ]

        for (hx, hy), mode in handles:
            if abs(canvas_x - hx) <= tolerance and abs(canvas_y - hy) <= tolerance:
                return mode

        return InteractionMode.NONE

    def resize_region(self, region, orig_x, orig_y, orig_w, orig_h, dx, dy):
        """Resize region based on interaction mode and delta"""
        mode = self.interaction_mode

        # Corners
        if mode == InteractionMode.RESIZING_NW:
            region.x = max(0, orig_x + dx)
            region.y = max(0, orig_y + dy)
            region.width = max(50, orig_w - dx)
            region.height = max(50, orig_h - dy)
        elif mode == InteractionMode.RESIZING_NE:
            region.y = max(0, orig_y + dy)
            region.width = max(50, orig_w + dx)
            region.height = max(50, orig_h - dy)
        elif mode == InteractionMode.RESIZING_SE:
            region.width = max(50, orig_w + dx)
            region.height = max(50, orig_h + dy)
        elif mode == InteractionMode.RESIZING_SW:
            region.x = max(0, orig_x + dx)
            region.width = max(50, orig_w - dx)
            region.height = max(50, orig_h + dy)

        # Edges
        elif mode == InteractionMode.RESIZING_N:
            region.y = max(0, orig_y + dy)
            region.height = max(50, orig_h - dy)
        elif mode == InteractionMode.RESIZING_S:
            region.height = max(50, orig_h + dy)
        elif mode == InteractionMode.RESIZING_E:
            region.width = max(50, orig_w + dx)
        elif mode == InteractionMode.RESIZING_W:
            region.x = max(0, orig_x + dx)
            region.width = max(50, orig_w - dx)

    def update_cursor(self, mode):
        """Update cursor based on interaction mode"""
        # Use Windows-compatible cursor names
        cursors = {
            InteractionMode.NONE: "",
            InteractionMode.DRAGGING: "fleur",
            InteractionMode.RESIZING_NW: "size_nw_se",
            InteractionMode.RESIZING_N: "sb_v_double_arrow",
            InteractionMode.RESIZING_NE: "size_ne_sw",
            InteractionMode.RESIZING_E: "sb_h_double_arrow",
            InteractionMode.RESIZING_SE: "size_nw_se",
            InteractionMode.RESIZING_S: "sb_v_double_arrow",
            InteractionMode.RESIZING_SW: "size_ne_sw",
            InteractionMode.RESIZING_W: "sb_h_double_arrow",
        }
        self.config(cursor=cursors.get(mode, ""))


# ============================================================================
# Right Panel: Single Photo Preview
# ============================================================================

class PhotoPreviewPanel(tk.Frame):
    """
    Right panel: Shows preview of currently selected photo with transformations applied.
    """
    def __init__(self, parent, app_state, app):
        super().__init__(parent, bg="lightgray")
        self.app_state = app_state
        self.app = app
        self.photo_image = None  # Keep reference to prevent GC

        # Info label (photo number)
        self.info_label = tk.Label(self, text="", font=("Arial", 10, "bold"),
                                   bg="lightgray")
        self.info_label.pack(pady=5)

        # Name editor frame
        name_frame = tk.Frame(self, bg="lightgray")
        name_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(name_frame, text="Name:", bg="lightgray").pack(side=tk.LEFT, padx=(0, 5))
        self.name_entry = tk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Update name only on Enter or focus loss (not on every keystroke)
        self.name_entry.bind("<Return>", self.on_name_changed)
        self.name_entry.bind("<FocusOut>", self.on_name_changed)

        # Preview canvas
        self.preview_canvas = tk.Canvas(self, bg="white", highlightthickness=1)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind resize event
        self.preview_canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        """Handle canvas resize"""
        self.refresh()

    def on_name_changed(self, event):
        """Handle name entry change (on Enter or focus loss)"""
        selected_region = self.app_state.get_selected_region()
        if selected_region:
            selected_region.name = self.name_entry.get()
            # Refresh left panel to update name display (full refresh to clear old text)
            if self.app:
                self.app.source_canvas.refresh()

    def refresh(self):
        """Update preview to show currently selected photo"""
        # Clear canvas
        self.preview_canvas.delete("all")

        # Get selected region
        selected_region = self.app_state.get_selected_region()

        if not selected_region or not self.app_state.original_image:
            # Show placeholder
            self.info_label.config(text="")
            self.name_entry.delete(0, tk.END)
            self.name_entry.config(state=tk.DISABLED)
            canvas_w = self.preview_canvas.winfo_width()
            canvas_h = self.preview_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                self.preview_canvas.create_text(
                    canvas_w / 2, canvas_h / 2,
                    text="Select a photo to preview",
                    font=("Arial", 14),
                    fill="gray"
                )
            return

        # Update info label
        try:
            idx = self.app_state.photo_regions.index(selected_region) + 1
            total = len(self.app_state.photo_regions)
            self.info_label.config(text=f"Photo {idx} of {total}")
        except ValueError:
            self.info_label.config(text="")

        # Update name entry
        self.name_entry.config(state=tk.NORMAL)
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, selected_region.name)

        # Crop from original image
        cropped = self.app_state.original_image.crop(selected_region.to_bbox())

        # Apply rotation
        if selected_region.rotation != 0:
            cropped = cropped.rotate(-selected_region.rotation, expand=True)

        # Scale to fit canvas
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return  # Canvas not yet sized

        # Calculate scale factor (fit to canvas, never upscale)
        img_w, img_h = cropped.size
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        scale_factor = min(scale_w, scale_h, 1.0)

        # Resize image
        if scale_factor < 1.0:
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)
            if new_w > 0 and new_h > 0:
                cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display centered
        self.photo_image = ImageTk.PhotoImage(cropped)
        self.preview_canvas.create_image(
            canvas_w / 2, canvas_h / 2,
            image=self.photo_image
        )


# ============================================================================
# Preferences Dialog
# ============================================================================

class PreferencesDialog(tk.Toplevel):
    """Preferences/Settings dialog"""
    def __init__(self, parent, config: AppConfig):
        super().__init__(parent)
        self.title("Preferences")
        self.config = config
        self.result = None

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Create UI
        self.create_widgets()

        # Center on parent
        self.geometry("400x250")
        self.resizable(False, False)

    def create_widgets(self):
        """Create preference controls"""
        # Main frame with padding
        main_frame = tk.Frame(self, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Overlay Color section
        tk.Label(main_frame, text="Region Overlay Color:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.overlay_var = tk.StringVar(value=self.config.overlay_color)
        tk.Radiobutton(main_frame, text="Green (selected) / Red (unselected)",
                      variable=self.overlay_var, value="green").grid(
            row=1, column=0, sticky=tk.W, padx=20)
        tk.Radiobutton(main_frame, text="Red (selected) / Green (unselected)",
                      variable=self.overlay_var, value="red").grid(
            row=2, column=0, sticky=tk.W, padx=20)

        # Export Format section
        tk.Label(main_frame, text="Export Format:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(15, 5))

        self.format_var = tk.StringVar(value=self.config.export_format)
        tk.Radiobutton(main_frame, text="Auto (same as source image)",
                      variable=self.format_var, value="auto").grid(
            row=4, column=0, sticky=tk.W, padx=20)
        tk.Radiobutton(main_frame, text="Always JPEG",
                      variable=self.format_var, value="jpeg").grid(
            row=5, column=0, sticky=tk.W, padx=20)
        tk.Radiobutton(main_frame, text="Always PNG",
                      variable=self.format_var, value="png").grid(
            row=6, column=0, sticky=tk.W, padx=20)

        # Filename Structure section
        tk.Label(main_frame, text="Export Filename:", font=("Arial", 10, "bold")).grid(
            row=7, column=0, sticky=tk.W, pady=(15, 5))

        self.filename_var = tk.BooleanVar(value=self.config.filename_include_source)
        tk.Checkbutton(main_frame, text="Include source filename (e.g., scan1-photo01.jpg)",
                      variable=self.filename_var).grid(
            row=8, column=0, sticky=tk.W, padx=20)

        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=9, column=0, pady=(20, 0))

        tk.Button(button_frame, text="OK", command=self.on_ok, width=10).pack(
            side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.on_cancel, width=10).pack(
            side=tk.LEFT, padx=5)

    def on_ok(self):
        """Save preferences and close"""
        self.config.overlay_color = self.overlay_var.get()
        self.config.export_format = self.format_var.get()
        self.config.filename_include_source = self.filename_var.get()
        self.result = True
        self.destroy()

    def on_cancel(self):
        """Close without saving"""
        self.result = False
        self.destroy()


# ============================================================================
# Main Application
# ============================================================================

class PhotoCropperApp(tk.Tk):
    """
    Main application window with two-panel layout.
    """
    def __init__(self, initial_image_path: Optional[Path] = None):
        super().__init__()

        self.title("Photo Cropper - scan-crop")
        self.geometry("1200x800")

        # Application state
        self.app_state = ApplicationState()

        # Setup menu
        self.setup_menu()

        # Create toolbar
        self.setup_toolbar()

        # Create two-panel layout (50/50 split)
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                          sashrelief=tk.RAISED, sashwidth=4)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL: Source image canvas
        self.left_frame = tk.Frame(self.paned_window)
        self.source_canvas = SourceImageCanvas(self.left_frame, self.app_state, self)
        self.source_canvas.pack(fill=tk.BOTH, expand=True)

        # RIGHT PANEL: Photo preview
        self.right_frame = tk.Frame(self.paned_window)
        self.photo_preview = PhotoPreviewPanel(self.right_frame, self.app_state, self)
        self.photo_preview.pack(fill=tk.BOTH, expand=True)

        # Add panels to paned window (50/50 split)
        self.paned_window.add(self.left_frame, width=600)
        self.paned_window.add(self.right_frame, width=600)

        # Status bar at bottom
        self.status_bar = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN,
                                   anchor=tk.W, padx=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Load initial image if provided, otherwise show welcome message
        if initial_image_path:
            # Schedule image loading after main window is rendered
            self.after(100, lambda: self.load_image(initial_image_path))
        else:
            self.show_welcome()

    def setup_toolbar(self):
        """Create toolbar with action buttons"""
        self.toolbar = tk.Frame(self, relief=tk.RAISED, borderwidth=2)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # Open Image button (always enabled)
        self.open_btn = tk.Button(
            self.toolbar, text="Open Image...",
            command=self.open_image
        )
        self.open_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray", relief=tk.SUNKEN).pack(
            side=tk.LEFT, fill=tk.Y, padx=5, pady=2
        )

        # Rotation buttons
        self.rotate_ccw_btn = tk.Button(
            self.toolbar, text="⟲ Rotate CCW",
            command=self.rotate_selected_ccw,
            state=tk.DISABLED
        )
        self.rotate_ccw_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.rotate_cw_btn = tk.Button(
            self.toolbar, text="⟳ Rotate CW",
            command=self.rotate_selected_cw,
            state=tk.DISABLED
        )
        self.rotate_cw_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray", relief=tk.SUNKEN).pack(
            side=tk.LEFT, fill=tk.Y, padx=5, pady=2
        )

        # Delete button
        self.delete_btn = tk.Button(
            self.toolbar, text="Delete Region",
            command=self.delete_selected_region,
            state=tk.DISABLED
        )
        self.delete_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # Separator
        tk.Frame(self.toolbar, width=2, bg="gray", relief=tk.SUNKEN).pack(
            side=tk.LEFT, fill=tk.Y, padx=5, pady=2
        )

        # Export Selected Photo button
        self.export_selected_btn = tk.Button(
            self.toolbar, text="Export Selected...",
            command=self.export_selected_photo,
            state=tk.DISABLED
        )
        self.export_selected_btn.pack(side=tk.LEFT, padx=2, pady=2)

        # Export All button (always enabled when photos exist)
        self.export_all_btn = tk.Button(
            self.toolbar, text="Export All Photos...",
            command=self.export_all_photos
        )
        self.export_all_btn.pack(side=tk.LEFT, padx=2, pady=2)

    def open_image(self):
        """Open image file dialog and load image"""
        file_path = filedialog.askopenfilename(
            title="Select scanned image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("WebP files", "*.webp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.load_image(Path(file_path))

    def update_toolbar_state(self):
        """Enable/disable toolbar buttons based on selection"""
        has_selection = self.app_state.selected_region_id is not None
        has_regions = len(self.app_state.photo_regions) > 0

        # Update button states
        state = tk.NORMAL if has_selection else tk.DISABLED
        self.rotate_ccw_btn.config(state=state)
        self.rotate_cw_btn.config(state=state)
        self.delete_btn.config(state=state)
        self.export_selected_btn.config(state=state)

        # Export All button enabled when there are regions
        self.export_all_btn.config(state=tk.NORMAL if has_regions else tk.DISABLED)

    def rotate_selected_cw(self):
        """Rotate selected region 90° clockwise"""
        region = self.app_state.get_selected_region()
        if region:
            region.rotation = (region.rotation + 90) % 360
            self.sync_selection()

    def rotate_selected_ccw(self):
        """Rotate selected region 90° counter-clockwise"""
        region = self.app_state.get_selected_region()
        if region:
            region.rotation = (region.rotation - 90) % 360
            self.sync_selection()

    def delete_selected_region(self):
        """Delete the currently selected region"""
        region = self.app_state.get_selected_region()
        if region and messagebox.askyesno("Delete Photo", "Delete this photo region?"):
            self.app_state.remove_region(region.region_id)
            self.source_canvas.refresh()
            self.sync_selection()

    def get_export_format_and_extension(self):
        """Determine export format based on config and source format"""
        if self.app_state.config.export_format == "auto":
            # Use source format
            format_name = self.app_state.source_format
        elif self.app_state.config.export_format == "jpeg":
            format_name = "JPEG"
        elif self.app_state.config.export_format == "png":
            format_name = "PNG"
        else:
            format_name = "JPEG"  # Default fallback

        # Map format to extension
        ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
        return format_name, ext_map.get(format_name, ".jpg")

    def export_selected_photo(self):
        """Export only the currently selected photo"""
        region = self.app_state.get_selected_region()
        if not region:
            return

        # Determine format and extension based on config
        format_name, file_ext = self.get_export_format_and_extension()

        # Get default filename based on config
        if self.app_state.image_path:
            base_name = self.app_state.image_path.stem
        else:
            base_name = "photo"

        # Build filename according to config
        if region.name:
            safe_name = "".join(c for c in region.name if c.isalnum() or c in (' ', '-', '_')).strip()
            if self.app_state.config.filename_include_source and self.app_state.image_path:
                default_filename = f"{base_name}-{safe_name}{file_ext}"
            else:
                default_filename = f"{safe_name}{file_ext}"
        else:
            if self.app_state.config.filename_include_source:
                default_filename = f"{base_name}_photo{file_ext}"
            else:
                default_filename = f"photo{file_ext}"

        # Ask user where to save
        output_file = filedialog.asksaveasfilename(
            title="Save Photo As",
            initialdir=self.app_state.output_directory,
            initialfile=default_filename,
            defaultextension=file_ext,
            filetypes=[
                ("Image files", f"*{file_ext}"),
                ("All files", "*.*")
            ]
        )

        if not output_file:
            return  # User cancelled

        try:
            # Crop from original
            cropped = self.app_state.original_image.crop(region.to_bbox())

            # Apply rotation
            if region.rotation != 0:
                cropped = cropped.rotate(-region.rotation, expand=True)

            # Save with format-appropriate settings
            if format_name == "JPEG":
                cropped.save(output_file, "JPEG", quality=95)
            elif format_name == "PNG":
                cropped.save(output_file, "PNG", optimize=True)
            elif format_name == "WEBP":
                cropped.save(output_file, "WEBP", quality=95)

            # Update default output directory
            self.app_state.output_directory = Path(output_file).parent

            # Show success in status bar
            self.status_bar.config(text=f"Exported photo to {Path(output_file).name}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export photo: {str(e)}")

    def export_all_photos(self):
        """Export all photos to selected directory"""
        if not self.app_state.photo_regions:
            messagebox.showwarning("No Photos", "No photos to export.")
            return

        # Ask user for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.app_state.output_directory
        )

        if not output_dir:
            return  # User cancelled

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Update default output directory
        self.app_state.output_directory = output_path

        # Get base name from source image
        if self.app_state.image_path:
            base_name = self.app_state.image_path.stem
        else:
            base_name = "photo"

        # Determine format and extension based on config
        format_name, file_ext = self.get_export_format_and_extension()

        # Export each region
        exported_count = 0
        for idx, region in enumerate(self.app_state.photo_regions, start=1):
            try:
                # Crop from original
                cropped = self.app_state.original_image.crop(region.to_bbox())

                # Apply rotation
                if region.rotation != 0:
                    cropped = cropped.rotate(-region.rotation, expand=True)

                # Generate filename according to config
                if region.name:
                    # Sanitize filename (remove invalid characters)
                    safe_name = "".join(c for c in region.name if c.isalnum() or c in (' ', '-', '_')).strip()
                    if self.app_state.config.filename_include_source and self.app_state.image_path:
                        filename = f"{base_name}-{safe_name}{file_ext}"
                    else:
                        filename = f"{safe_name}{file_ext}"
                else:
                    if self.app_state.config.filename_include_source:
                        filename = f"{base_name}_photo_{idx:02d}{file_ext}"
                    else:
                        filename = f"photo_{idx:02d}{file_ext}"

                output_file = output_path / filename

                # Save with format-appropriate settings
                if format_name == "JPEG":
                    cropped.save(output_file, "JPEG", quality=95)
                elif format_name == "PNG":
                    cropped.save(output_file, "PNG", optimize=True)
                elif format_name == "WEBP":
                    cropped.save(output_file, "WEBP", quality=95)
                exported_count += 1

            except Exception as e:
                messagebox.showerror("Export Error",
                                   f"Failed to export photo {idx}: {str(e)}")
                return

        # Show success message in status bar
        self.status_bar.config(
            text=f"Successfully exported {exported_count} photo(s) to: {output_path}"
        )

    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences...", command=self.show_preferences)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Keyboard shortcuts
        self.bind("<Delete>", lambda e: self.delete_selected_region())
        self.bind("<Up>", lambda e: self.move_selected_region(0, -1))
        self.bind("<Down>", lambda e: self.move_selected_region(0, 1))
        self.bind("<Left>", lambda e: self.move_selected_region(-1, 0))
        self.bind("<Right>", lambda e: self.move_selected_region(1, 0))

        # Shift+Arrow for resizing
        self.bind("<Shift-Up>", lambda e: self.resize_selected_region(0, -1))
        self.bind("<Shift-Down>", lambda e: self.resize_selected_region(0, 1))
        self.bind("<Shift-Left>", lambda e: self.resize_selected_region(-1, 0))
        self.bind("<Shift-Right>", lambda e: self.resize_selected_region(1, 0))

        # Tab to cycle through regions
        self.bind("<Tab>", lambda e: self.cycle_selection(1))
        self.bind("<Shift-Tab>", lambda e: self.cycle_selection(-1))

    def show_welcome(self):
        """Show welcome message on empty canvas"""
        self.source_canvas.create_text(
            400, 300,
            text="Welcome to Photo Cropper\n\nDrag an image file onto the window\nor use toolbar to open",
            font=("Arial", 16),
            fill="white"
        )

    def move_selected_region(self, dx, dy):
        """Move selected region by dx, dy pixels"""
        region = self.app_state.get_selected_region()
        if not region or not self.app_state.original_image:
            return

        img_w, img_h = self.app_state.original_image.size

        # Move region, keeping it within image bounds
        new_x = max(0, min(img_w - region.width, region.x + dx))
        new_y = max(0, min(img_h - region.height, region.y + dy))

        region.x = new_x
        region.y = new_y

        # Refresh display
        self.sync_selection()

    def resize_selected_region(self, dw, dh):
        """Resize selected region by dw, dh pixels (anchored at top-left)"""
        region = self.app_state.get_selected_region()
        if not region or not self.app_state.original_image:
            return

        img_w, img_h = self.app_state.original_image.size

        # Resize region, keeping minimum size of 50x50 and within image bounds
        new_width = max(50, min(img_w - region.x, region.width + dw))
        new_height = max(50, min(img_h - region.y, region.height + dh))

        region.width = new_width
        region.height = new_height

        # Refresh display
        self.sync_selection()

    def cycle_selection(self, direction):
        """Cycle through regions (direction: 1 for forward, -1 for backward)"""
        if not self.app_state.photo_regions:
            return

        # Find current selection index
        current_idx = -1
        if self.app_state.selected_region_id:
            for idx, region in enumerate(self.app_state.photo_regions):
                if region.region_id == self.app_state.selected_region_id:
                    current_idx = idx
                    break

        # Calculate next index (with wrap-around)
        next_idx = (current_idx + direction) % len(self.app_state.photo_regions)

        # Select next region
        self.app_state.selected_region_id = self.app_state.photo_regions[next_idx].region_id

        # Refresh display
        self.sync_selection()

        # Prevent Tab from changing focus to other widgets
        return "break"

    def show_shortcuts(self):
        """Show keyboard shortcuts reference"""
        shortcuts_text = """Keyboard Shortcuts:

Navigation:
  Tab                 Select next region
  Shift+Tab           Select previous region

Movement:
  Arrow keys          Move selected region (1 pixel)
  Shift+Arrow keys    Resize selected region (1 pixel)

Actions:
  Delete              Delete selected region

Tip: Hold arrow keys for continuous movement/resizing"""

        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def load_image(self, path: Path):
        """Load image and run detection"""
        try:
            # Load image
            self.app_state.image_path = path
            self.app_state.original_image = Image.open(path)

            # Detect and store source format for format-preserving export
            source_format = self.app_state.original_image.format
            if source_format in ("JPEG", "PNG", "WEBP"):
                self.app_state.source_format = source_format
            else:
                # Default to JPEG for unknown formats
                self.app_state.source_format = "JPEG"

            # Run detection
            print(f"Running detection on {path.name}...")
            detected_regions = detect_photos(self.app_state.original_image)

            # Store detected regions
            self.app_state.photo_regions = detected_regions

            print(f"Found {len(detected_regions)} photo(s)")

            # Update both panels
            self.source_canvas.refresh()
            self.photo_preview.refresh()
            self.update_toolbar_state()

            # Show result in status bar (non-intrusive)
            if detected_regions:
                self.status_bar.config(
                    text=f"Found {len(detected_regions)} photo(s) in {path.name}. "
                         f"Select a photo to preview and edit."
                )
            else:
                self.status_bar.config(
                    text=f"No photos detected in {path.name}. "
                         f"Use Edit → Add Region to create manual regions."
                )
                # Show warning dialog only when nothing detected (important)
                messagebox.showwarning(
                    "No Photos Detected",
                    f"Could not detect any photos in {path.name}\n\n"
                    f"You can manually add regions by dragging on the left canvas."
                )

        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))
            import traceback
            traceback.print_exc()

    def show_preferences(self):
        """Show preferences dialog"""
        dialog = PreferencesDialog(self, self.app_state.config)
        self.wait_window(dialog)

        # If preferences were changed, refresh display
        if dialog.result:
            self.source_canvas.refresh()

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Photo Cropper",
            f"Photo Cropper - scan-crop GUI\n\n"
            f"Extract individual photos from scanned images.\n\n"
            f"Version {__version__}\n"
            f"Built with tkinter, PIL, and OpenCV\n\n"
            f"https://github.com/rusanu/scan-crop"
        )

    def sync_selection(self):
        """Synchronize selection highlighting between left and right panels"""
        # Update toolbar button states
        self.update_toolbar_state()

        # Update left panel (will redraw with green border + handles)
        self.source_canvas.refresh()

        # Update right panel preview
        self.photo_preview.refresh()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with command-line argument support"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Photo Cropper - Extract individual photos from scanned images"
    )
    parser.add_argument(
        "image",
        nargs="?",  # Optional positional argument
        type=str,
        help="Path to image file to open automatically (optional)"
    )

    args = parser.parse_args()

    # Convert image path to Path object if provided
    initial_image = None
    if args.image:
        initial_image = Path(args.image)

        # Validate that the file exists
        if not initial_image.exists():
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)

        # Validate that it's a file (not a directory)
        if not initial_image.is_file():
            print(f"Error: Path is not a file: {args.image}", file=sys.stderr)
            sys.exit(1)

    # Create and run the application
    app = PhotoCropperApp(initial_image_path=initial_image)
    app.mainloop()


if __name__ == "__main__":
    main()
