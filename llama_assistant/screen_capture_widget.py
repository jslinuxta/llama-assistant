from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QPen

from llama_assistant import config
from llama_assistant.ocr_engine import OCREngine
if TYPE_CHECKING:
    from llama_assistant.llama_assistant_app import LlamaAssistantApp


class ScreenCaptureWidget(QWidget):
    def __init__(self, parent: "LlamaAssistantApp"):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.parent = parent
        self.ocr_engine = OCREngine()
        
        # Get screen size
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Set crosshairs cursor
        self.setCursor(Qt.CrossCursor)
        
        # To store the start and end points of the mouse region
        self.start_point = None
        self.end_point = None

        # Buttons to appear after selection
        self.button_widget = QWidget()
        self.ocr_button = QPushButton("OCR", self.button_widget)
        self.ask_button = QPushButton("Ask", self.button_widget)
        self.ocr_button.setCursor(Qt.PointingHandCursor)
        self.ask_button.setCursor(Qt.PointingHandCursor)
        opacity = self.parent.settings.get("transparency", 90) / 100
        base_style = f"""
            border: none;
            border-radius: 20px;
            color: white;
            padding: 10px 15px;
            font-size: 16px;
        """
        button_style = f"""
            QPushButton {{
                {base_style}
                padding: 2.5px 5px;
                border-radius: 5px;
                background-color: rgba{QColor(self.parent.settings["color"]).lighter(120).getRgb()[:3] + (opacity,)};
            }}
        """
        self.ocr_button.setStyleSheet(button_style)
        self.ask_button.setStyleSheet(button_style)
        self.button_widget.hide()

        # Connect button signals
        self.ocr_button.clicked.connect(self.parent.on_ocr_button_clicked)
        self.ask_button.clicked.connect(self.parent.on_ask_with_ocr_context)

    def show(self):
        # remove painting if any
        self.start_point = None
        self.end_point = None
        self.update()

        # Set window opacity to 50%
        self.setWindowOpacity(0.5)
        # self.setAttribute(Qt.WA_TranslucentBackground, True)
        super().show()

    def hide(self):
        self.button_widget.hide()
        super().hide()

        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()  # Capture start position
            self.end_point = event.pos()    # Initialize end point to start position
            print(f"Mouse press at {self.start_point}")
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = event.pos()  # Capture end position
            
            print(f"Mouse release at {self.end_point}")
            
            # Capture the region between start and end points
            if self.start_point and self.end_point:
                self.capture_region(self.start_point, self.end_point)
                
            # Trigger repaint to show the red rectangle
            self.update()

        self.show_buttons()

                
    def mouseMoveEvent(self, event):
        if self.start_point:
            # Update the end_point to the current mouse position as it moves
            self.end_point = event.pos()
            
            # Trigger repaint to update the rectangle
            self.update()
            
    def capture_region(self, start_point, end_point):
        # Convert local widget coordinates to global screen coordinates
        start_global = self.mapToGlobal(start_point)
        end_global = self.mapToGlobal(end_point)
        
        # Create a QRect from the global start and end points
        region_rect = QRect(start_global, end_global)
        
        # Ensure the rectangle is valid (non-negative width/height)
        region_rect = region_rect.normalized()
        
        # Capture the screen region
        screen = QApplication.primaryScreen()
        pixmap = screen.grabWindow(0, region_rect.x(), region_rect.y(), region_rect.width(), region_rect.height())

        # Save the captured region as an image
        pixmap.save(str(config.ocr_tmp_file), "PNG")
        print(f"Captured region saved at '{config.ocr_tmp_file}'.")
    
    def paintEvent(self, event):
        # If the start and end points are set, draw the rectangle
        if self.start_point and self.end_point:
            # Create a painter object
            painter = QPainter(self)
            
            # Set the pen color to red
            pen = QPen(QColor(255, 0, 0))  # Red color
            pen.setWidth(3)  # Set width of the border
            painter.setPen(pen)
            
            # Draw the rectangle from start_point to end_point
            self.region_rect = QRect(self.start_point, self.end_point)
            self.region_rect = self.region_rect.normalized()  # Normalize to ensure correct width/height
            
            painter.drawRect(self.region_rect)  # Draw the rectangle

        super().paintEvent(event)  # Call the base class paintEvent

    def show_buttons(self):
        if self.start_point and self.end_point:
            # Get normalized rectangle
            rect = QRect(self.start_point, self.end_point).normalized()

            # Calculate button positions
            button_y = rect.bottom() + 10  # Place buttons below the rectangle
            button_width = 80
            button_height = 30
            spacing = 10

            print("Showing buttons")

            self.ocr_button.setGeometry(0, 0, button_width, button_height)
            self.ask_button.setGeometry(button_width + spacing, 0, button_width, button_height)

            self.button_widget.setGeometry(rect.left(), button_y, 2 * button_width + spacing, button_height)
            self.button_widget.setAttribute(Qt.WA_TranslucentBackground)
            self.button_widget.setWindowFlags(Qt.FramelessWindowHint)
            self.button_widget.show()

    