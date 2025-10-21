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

from PIL import Image, ImageTk
import cv2
import numpy as np
from enum import Enum


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
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        region = PhotoRegion(
            x=x, y=y, width=w, height=h,
            rotation=0,
            is_manual=False,
            list_order=idx
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

    def to_bbox(self):
        """Returns (x1, y1, x2, y2) tuple for PIL crop"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def contains_point(self, px, py):
        """Check if point is inside region (used for selection)"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


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
                # Get next list order
                max_order = max([r.list_order for r in self.app_state.photo_regions], default=-1)

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
                    list_order=max_order + 1
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
                    self.app.photo_list.refresh_list()
                    self.app.sync_selection()

                self.refresh()
                return  # Exit early to avoid clearing mode twice

        elif self.interaction_mode != InteractionMode.NONE:
            # Update thumbnails in right panel after drag/resize
            if self.app:
                for widget in self.app.photo_list.photo_widgets:
                    if widget.region.region_id == self.app_state.selected_region_id:
                        widget.update_thumbnail()
                        break

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

            # Determine color (selected vs unselected)
            is_selected = (region.region_id == self.app_state.selected_region_id)
            color = "green" if is_selected else "red"
            width = 3 if is_selected else 2

            # Draw rectangle
            self.create_rectangle(x1, y1, x2, y2,
                                outline=color, width=width,
                                tags=f"region_{region.region_id}")

            # Draw resize handles for selected region
            if is_selected:
                self.draw_resize_handles(x1, y1, x2, y2)

            # Show rotation indicator if rotated
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
# Right Panel: Photo List
# ============================================================================

class PhotoItemWidget(tk.Frame):
    """
    Individual photo card showing thumbnail and controls.
    """
    def __init__(self, parent, region, app_state, photo_list_panel):
        super().__init__(parent, relief=tk.RAISED, borderwidth=2, padx=5, pady=5)
        self.region = region
        self.app_state = app_state
        self.photo_list = photo_list_panel
        self.photo_image = None  # Keep reference

        # Bind click event to select this card
        self.bind("<Button-1>", self.on_click)

        # Header with photo number
        idx = self.app_state.photo_regions.index(region) + 1
        header = tk.Label(self, text=f"Photo {idx}", font=("Arial", 10, "bold"))
        header.pack()
        header.bind("<Button-1>", self.on_click)  # Propagate clicks

        # Thumbnail canvas
        self.thumbnail_canvas = tk.Canvas(self, width=200, height=200,
                                         bg="white", highlightthickness=1)
        self.thumbnail_canvas.pack(pady=5)
        self.thumbnail_canvas.bind("<Button-1>", self.on_click)  # Allow clicking thumbnail

        # Rotation controls
        rotation_frame = tk.Frame(self)
        rotation_frame.pack()
        rotation_frame.bind("<Button-1>", self.on_click)  # Propagate clicks

        self.rotation_label = tk.Label(rotation_frame,
                                       text=f"Rotation: {region.rotation}°")
        self.rotation_label.pack(side=tk.LEFT, padx=5)
        self.rotation_label.bind("<Button-1>", self.on_click)  # Propagate clicks

        self.rotate_ccw_btn = tk.Button(rotation_frame, text="⟲", width=3,
                 command=self.rotate_ccw)
        self.rotate_ccw_btn.pack(side=tk.LEFT, padx=2)
        self.rotate_ccw_btn.bind("<Button-1>", self.on_click_and_execute, add='+')

        self.rotate_cw_btn = tk.Button(rotation_frame, text="⟳", width=3,
                 command=self.rotate_cw)
        self.rotate_cw_btn.pack(side=tk.LEFT, padx=2)
        self.rotate_cw_btn.bind("<Button-1>", self.on_click_and_execute, add='+')

        # Action buttons
        action_frame = tk.Frame(self)
        action_frame.pack(pady=5)
        action_frame.bind("<Button-1>", self.on_click)  # Propagate clicks

        self.delete_btn = tk.Button(action_frame, text="Delete", width=8,
                 command=self.delete_region)
        self.delete_btn.pack(side=tk.LEFT, padx=2)
        self.delete_btn.bind("<Button-1>", self.on_click_and_execute, add='+')

        self.move_up_btn = tk.Button(action_frame, text="↑", width=3,
                 command=self.move_up)
        self.move_up_btn.pack(side=tk.LEFT, padx=2)
        self.move_up_btn.bind("<Button-1>", self.on_click_and_execute, add='+')

        self.move_down_btn = tk.Button(action_frame, text="↓", width=3,
                 command=self.move_down)
        self.move_down_btn.pack(side=tk.LEFT, padx=2)
        self.move_down_btn.bind("<Button-1>", self.on_click_and_execute, add='+')

        # Render initial thumbnail
        self.update_thumbnail()

    def update_thumbnail(self):
        """Render thumbnail with current transformations"""
        if not self.app_state.original_image:
            return

        # Crop from original
        cropped = self.app_state.original_image.crop(self.region.to_bbox())

        # Apply rotation
        if self.region.rotation != 0:
            cropped = cropped.rotate(-self.region.rotation, expand=True)

        # Create thumbnail (maintains aspect ratio)
        cropped.thumbnail((200, 200), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        self.photo_image = ImageTk.PhotoImage(cropped)
        self.thumbnail_canvas.delete("all")

        # Center the image
        self.thumbnail_canvas.create_image(100, 100, image=self.photo_image)

    def rotate_cw(self):
        """Rotate 90° clockwise"""
        self.region.rotation = (self.region.rotation + 90) % 360
        self.rotation_label.config(text=f"Rotation: {self.region.rotation}°")
        self.update_thumbnail()
        # Notify left panel
        self.photo_list.app.source_canvas.draw_regions()

    def rotate_ccw(self):
        """Rotate 90° counter-clockwise"""
        self.region.rotation = (self.region.rotation - 90) % 360
        self.rotation_label.config(text=f"Rotation: {self.region.rotation}°")
        self.update_thumbnail()
        self.photo_list.app.source_canvas.draw_regions()

    def delete_region(self):
        """Delete this region"""
        if messagebox.askyesno("Delete Photo",
                              f"Delete this photo region?"):
            self.app_state.remove_region(self.region.region_id)
            self.photo_list.refresh()
            self.photo_list.app.source_canvas.refresh()

    def move_up(self):
        """Move this photo up in the list"""
        # TODO: Implement reordering
        pass

    def move_down(self):
        """Move this photo down in the list"""
        # TODO: Implement reordering
        pass

    def on_click(self, event):
        """Handle click on this card"""
        # Select this region
        self.app_state.selected_region_id = self.region.region_id

        # Sync with left panel and update highlights
        if self.photo_list.app:
            self.photo_list.app.sync_selection()

    def on_click_and_execute(self, event):
        """Handle click on button - select card first, then let button command execute"""
        # Select this region first
        self.on_click(event)
        # Button's command will execute automatically after this

    def update_highlight(self):
        """Update visual highlight based on selection state"""
        is_selected = (self.region.region_id == self.app_state.selected_region_id)

        if is_selected:
            self.config(bg="lightyellow", relief=tk.SOLID, borderwidth=3)
        else:
            self.config(bg="SystemButtonFace", relief=tk.RAISED, borderwidth=2)


class PhotoListPanel(tk.Frame):
    """
    Right panel: Scrollable list of photo cards.
    """
    def __init__(self, parent, app_state, app):
        super().__init__(parent)
        self.app_state = app_state
        self.app = app
        self.photo_widgets = []

        # Scrollable canvas for photo cards
        self.canvas = tk.Canvas(self, bg="lightgray")
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL,
                                     command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="lightgray")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame,
                                 anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Export button at bottom (outside scroll area)
        self.export_btn = tk.Button(self, text="Export All Photos...",
                                    command=self.export_all, height=2)
        self.export_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def refresh(self):
        """Rebuild list of photo cards"""
        # Clear existing widgets
        for widget in self.photo_widgets:
            widget.destroy()
        self.photo_widgets.clear()

        # Create new widgets
        for region in self.app_state.photo_regions:
            widget = PhotoItemWidget(self.scrollable_frame, region,
                                    self.app_state, self)
            widget.pack(fill=tk.X, padx=5, pady=5)
            self.photo_widgets.append(widget)

    def export_all(self):
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

        # Export each region
        exported_count = 0
        for idx, region in enumerate(self.app_state.photo_regions, start=1):
            try:
                # Crop from original
                cropped = self.app_state.original_image.crop(region.to_bbox())

                # Apply rotation
                if region.rotation != 0:
                    cropped = cropped.rotate(-region.rotation, expand=True)

                # Generate filename
                filename = f"{base_name}_photo_{idx:02d}.jpg"
                output_file = output_path / filename

                # Save with high quality
                cropped.save(output_file, "JPEG", quality=95)
                exported_count += 1

            except Exception as e:
                messagebox.showerror("Export Error",
                                   f"Failed to export photo {idx}: {str(e)}")
                return

        # Show success message
        messagebox.showinfo("Export Complete",
                          f"Successfully exported {exported_count} photo(s) to:\n{output_path}")


# ============================================================================
# Main Application
# ============================================================================

class PhotoCropperApp(tk.Tk):
    """
    Main application window with two-panel layout.
    """
    def __init__(self):
        super().__init__()

        self.title("Photo Cropper - scan-crop")
        self.geometry("1200x800")

        # Application state
        self.app_state = ApplicationState()

        # Setup menu
        self.setup_menu()

        # Create two-panel layout
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                          sashrelief=tk.RAISED, sashwidth=4)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL: Source image canvas
        self.left_frame = tk.Frame(self.paned_window)
        self.source_canvas = SourceImageCanvas(self.left_frame, self.app_state, self)
        self.source_canvas.pack(fill=tk.BOTH, expand=True)

        # RIGHT PANEL: Photo list
        self.right_frame = tk.Frame(self.paned_window)
        self.photo_list = PhotoListPanel(self.right_frame, self.app_state, self)
        self.photo_list.pack(fill=tk.BOTH, expand=True)

        # Add panels to paned window (60/40 split)
        self.paned_window.add(self.left_frame, width=720)
        self.paned_window.add(self.right_frame, width=480)

        # Show welcome message
        self.show_welcome()

    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.open_image,
                             accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Region Manually...",
                             command=self.add_manual_region)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self.open_image())

    def show_welcome(self):
        """Show welcome message on empty canvas"""
        self.source_canvas.create_text(
            400, 300,
            text="Welcome to Photo Cropper\n\nFile → Open Image to get started",
            font=("Arial", 16),
            fill="white"
        )

    def open_image(self):
        """Open image file dialog and load image"""
        file_path = filedialog.askopenfilename(
            title="Select scanned image",
            filetypes=[
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.load_image(Path(file_path))

    def load_image(self, path: Path):
        """Load image and run detection"""
        try:
            # Load image
            self.app_state.image_path = path
            self.app_state.original_image = Image.open(path)

            # Run detection
            print(f"Running detection on {path.name}...")
            detected_regions = detect_photos(self.app_state.original_image)

            # Store detected regions
            self.app_state.photo_regions = detected_regions

            print(f"Found {len(detected_regions)} photo(s)")

            # Update both panels
            self.source_canvas.refresh()
            self.photo_list.refresh()

            # Show result
            if detected_regions:
                messagebox.showinfo(
                    "Detection Complete",
                    f"Found {len(detected_regions)} photo(s) in {path.name}\n\n"
                    f"Adjust crop regions on the left panel.\n"
                    f"Rotate photos on the right panel."
                )
            else:
                messagebox.showwarning(
                    "No Photos Detected",
                    f"Could not detect any photos in {path.name}\n\n"
                    f"You can manually add regions using Edit → Add Region."
                )

        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))
            import traceback
            traceback.print_exc()

    def add_manual_region(self):
        """Add manual crop region"""
        # TODO: Implement manual region drawing
        messagebox.showinfo("Add Region", "Manual region creation coming soon!")

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Photo Cropper",
            "Photo Cropper - scan-crop GUI\n\n"
            "Extract individual photos from scanned images.\n\n"
            "Version 1.0\n"
            "Built with tkinter and PIL"
        )

    def sync_selection(self):
        """Synchronize selection highlighting between left and right panels"""
        # Update left panel (will redraw with green border + handles)
        self.source_canvas.refresh()

        # Update right panel highlights
        for widget in self.photo_list.photo_widgets:
            widget.update_highlight()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    app = PhotoCropperApp()
    app.mainloop()


if __name__ == "__main__":
    main()
