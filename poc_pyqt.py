#!/usr/bin/env python3
"""
Proof of Concept: PyQt5 GUI for scan-crop
Tests: image display, rectangle drawing, mouse interaction
Requires: pip install PyQt5
"""

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView,
                                  QGraphicsScene, QGraphicsRectItem, QFileDialog,
                                  QAction, QGraphicsPixmapItem)
    from PyQt5.QtCore import Qt, QRectF, QPointF
    from PyQt5.QtGui import QPen, QColor, QPixmap, QBrush
    import sys

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not installed. Run: pip install PyQt5")


class CropRectItem(QGraphicsRectItem):
    """Custom rectangle item with rotation state"""
    def __init__(self, rect, rotation=0):
        super().__init__(rect)
        self.rotation_angle = rotation
        self.setPen(QPen(QColor("red"), 2))
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

    def paint(self, painter, option, widget):
        # Change color when selected
        if self.isSelected():
            self.setPen(QPen(QColor("green"), 3))
        else:
            self.setPen(QPen(QColor("red"), 2))
        super().paint(painter, option, widget)


class PhotoCropperPyQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Cropper - PyQt5 POC")
        self.setGeometry(100, 100, 800, 600)

        # State
        self.crop_items = []  # List of CropRectItem

        # Setup UI
        self.setup_ui()

        # Test: Add sample rectangles
        self.add_test_rectangles()

    def setup_ui(self):
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Graphics view and scene
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        self.setCentralWidget(self.view)

    def load_image(self):
        """Load image from file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select scanned image",
            "",
            "JPEG files (*.jpg);;All files (*.*)"
        )

        if file_path:
            pixmap = QPixmap(file_path)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def add_test_rectangles(self):
        """Add sample crop rectangles for testing"""
        rect1 = CropRectItem(QRectF(100, 100, 200, 300), rotation=0)
        rect2 = CropRectItem(QRectF(400, 150, 200, 200), rotation=90)

        self.scene.addItem(rect1)
        self.scene.addItem(rect2)

        self.crop_items = [rect1, rect2]


def main():
    if not PYQT_AVAILABLE:
        sys.exit(1)

    app = QApplication(sys.argv)
    window = PhotoCropperPyQt()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
