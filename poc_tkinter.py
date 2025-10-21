#!/usr/bin/env python3
"""
Proof of Concept: tkinter GUI for scan-crop
Tests: image display, rectangle drawing, mouse interaction
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys

class PhotoCropperTkinter:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Cropper - tkinter POC")
        self.root.geometry("800x600")

        # State
        self.image = None
        self.photo_image = None
        self.rectangles = []  # List of (x1, y1, x2, y2, rotation)
        self.selected_rect = None
        self.drag_start = None

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Test: Draw some rectangles
        self.rectangles = [(100, 100, 300, 400, 0), (400, 150, 600, 350, 0)]

    def load_image(self):
        """Load image from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select scanned image",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )

        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        """Display image on canvas with scaling"""
        if not self.image:
            return

        # Scale image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate scaling factor
        img_width, img_height = self.image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale

        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Draw rectangles
        self.draw_rectangles()

    def draw_rectangles(self):
        """Draw all crop rectangles on canvas"""
        for i, (x1, y1, x2, y2, rotation) in enumerate(self.rectangles):
            color = "green" if i == self.selected_rect else "red"
            width = 3 if i == self.selected_rect else 2

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=width,
                tags=f"rect_{i}"
            )

            # Draw rotation indicator (simple text for POC)
            if rotation != 0:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self.canvas.create_text(
                    cx, cy,
                    text=f"{rotation}°",
                    fill=color,
                    font=("Arial", 14, "bold")
                )

    def on_mouse_down(self, event):
        """Handle mouse click"""
        # Check if clicking on a rectangle
        self.selected_rect = None
        for i, (x1, y1, x2, y2, _) in enumerate(self.rectangles):
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.selected_rect = i
                self.drag_start = (event.x, event.y)
                break

        self.draw_rectangles()

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if self.selected_rect is not None and self.drag_start:
            # Calculate drag offset
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]

            # Move rectangle
            x1, y1, x2, y2, rotation = self.rectangles[self.selected_rect]
            self.rectangles[self.selected_rect] = (
                x1 + dx, y1 + dy,
                x2 + dx, y2 + dy,
                rotation
            )

            self.drag_start = (event.x, event.y)
            self.draw_rectangles()

    def on_mouse_up(self, event):
        """Handle mouse release"""
        self.drag_start = None

def main():
    root = tk.Tk()
    app = PhotoCropperTkinter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
