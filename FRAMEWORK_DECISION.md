# GUI Framework Selection for scan-crop

**Decision Date**: 2025-10-21
**Selected Framework**: **tkinter**

## Executive Summary

After evaluating tkinter, PyQt5/PySide6, and wxPython, **tkinter** is the best choice for scan-crop's GUI transformation based on deployment simplicity, bundle size, and sufficient feature support.

## Evaluation Criteria

1. **PyInstaller single-file executable support**
2. **Bundle size** (critical for user distribution)
3. **Canvas/interactive drawing capabilities**
4. **Mouse event handling** (drag, resize, select)
5. **Built-in vs external dependency**
6. **Windows compatibility**

## Framework Comparison

### tkinter ✅ SELECTED

**Pros:**
- **Built into Python** - No additional dependencies in requirements.txt
- **Excellent PyInstaller support** - Tested successfully
- **Small bundle size** - 30MB single-file .exe (includes PIL, numpy, OpenCV)
- **Sufficient canvas capabilities** - Canvas widget supports rectangles, mouse events, image display
- **Simple API** - Easy to maintain for non-programmer users who may need to modify
- **Native Windows widgets** - File dialogs work correctly

**Cons:**
- Less "modern" appearance (acceptable for utility tool)
- More manual coding for complex interactions (mitigated by simple requirements)

**Test Results:**
- ✅ POC builds with `pyinstaller --onefile --windowed`
- ✅ Executable size: 30MB
- ✅ Mouse interaction working (drag rectangles)
- ✅ File dialog for image loading
- ✅ Image display with PIL integration

**Code Sample:** See `poc_tkinter.py`

---

### PyQt5/PySide6 ❌ NOT SELECTED

**Pros:**
- Professional, polished UI
- Excellent QGraphicsView framework for interactive canvas
- Strong PyInstaller support
- Rich widget ecosystem

**Cons:**
- **Large external dependency** - Adds 50-100MB to bundle
- **Licensing complexity** - PyQt5 is GPL (forces GPL on scan-crop), PySide6 is LGPL
- **Overkill for this use case** - We don't need the advanced features
- **Steeper learning curve** - Harder for non-technical users to maintain

**Verdict:** Too heavy for a simple photo cropping utility. Would be worth it for complex applications, but tkinter is sufficient here.

---

### wxPython ❌ NOT SELECTED

**Pros:**
- Native Windows look
- Good canvas support

**Cons:**
- **Large bundle size** (similar to PyQt)
- **External dependency** (less popular than PyQt, fewer examples)
- **PyInstaller can be problematic** - Known issues with bundling
- **Less documentation** than tkinter or PyQt

**Verdict:** Middle ground between tkinter and PyQt, but doesn't excel at anything. tkinter is simpler, PyQt is more powerful.

---

## Final Decision Rationale

**tkinter wins because:**

1. **Zero dependency overhead** - Already included with Python
2. **30MB total bundle** - Competitive for a GUI app with image processing
3. **Proven PyInstaller compatibility** - Built and tested successfully
4. **Sufficient for requirements** - Canvas supports all needed interactions:
   - Display images
   - Draw/update rectangles
   - Handle mouse events (click, drag, release)
   - Track selection state
   - Render rotation indicators

5. **Maintainability** - Simple API means non-technical users can understand the code if needed
6. **No licensing concerns** - tkinter is part of Python's standard library

## Implementation Notes

The POC (`poc_tkinter.py`) demonstrates:
- Image loading with file dialog
- Canvas-based image display
- Rectangle drawing and selection
- Mouse drag interaction
- Coordinate system handling

**Next Steps:**
- Refine POC into full architecture (scan-crop-3)
- Integrate existing OpenCV detection algorithm
- Add resize handles and rotation controls
- Implement save workflow

## Alternatives Considered But Not Tested

- **Kivy** - Modern, but overkill and has poor PyInstaller support
- **Dear PyGui** - Game-focused, not suitable for image editing
- **Tkinter + customtkinter** - Could improve appearance, but adds dependency

**Conclusion:** Standard tkinter is the pragmatic choice for this project.
