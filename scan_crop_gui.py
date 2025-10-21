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

        # Bind events
        self.bind("<Configure>", self.on_resize)
        self.bind("<Button-1>", self.on_mouse_down)

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

        # Try to select region at this point
        selected = self.app_state.select_region_at_point(img_x, img_y)

        # Refresh display
        self.refresh()

        # Sync with right panel
        if self.app:
            self.app.sync_selection()

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

        # Rotation controls
        rotation_frame = tk.Frame(self)
        rotation_frame.pack()

        self.rotation_label = tk.Label(rotation_frame,
                                       text=f"Rotation: {region.rotation}°")
        self.rotation_label.pack(side=tk.LEFT, padx=5)

        tk.Button(rotation_frame, text="⟲", width=3,
                 command=self.rotate_ccw).pack(side=tk.LEFT, padx=2)
        tk.Button(rotation_frame, text="⟳", width=3,
                 command=self.rotate_cw).pack(side=tk.LEFT, padx=2)

        # Action buttons
        action_frame = tk.Frame(self)
        action_frame.pack(pady=5)

        tk.Button(action_frame, text="Delete", width=8,
                 command=self.delete_region).pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="↑", width=3,
                 command=self.move_up).pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="↓", width=3,
                 command=self.move_down).pack(side=tk.LEFT, padx=2)

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
        """Export all photos"""
        if not self.app_state.photo_regions:
            messagebox.showwarning("No Photos", "No photos to export.")
            return

        # TODO: Implement export functionality
        messagebox.showinfo("Export", "Export functionality coming soon!")


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
