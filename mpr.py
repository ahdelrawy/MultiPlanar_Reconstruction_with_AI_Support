"""
3D Slicer-Style Medical Image Viewer with Orientation Classification
Modified: Fourth window with selectable modes (Oblique/Brain Surface)
"""

import sys
import os
import numpy as np
import json
import math

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGridLayout, QLabel, QPushButton, QToolBar, 
                            QStatusBar, QAction, QFileDialog, QMessageBox, 
                            QSlider, QGroupBox, QSpinBox, QComboBox, QDoubleSpinBox,
                            QCheckBox, QDialog, QDialogButtonBox, QTextEdit, QApplication,
                            QButtonGroup, QRadioButton, QProgressDialog, QListWidget,
                            QListWidgetItem, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint, QRect, QPointF, QLineF, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QMouseEvent, QBrush

import google.generativeai as genai
from PIL import Image

# Gemini API Key
API_KEY = "AIzaSyC5bkRcmKsHMyrkYJWErkXwQmb-5dxUSZs"

# Brain Surface DICOM path - MODIFY THIS PATH
BRAIN_SURFACE_DICOM_PATH = r"D:\task2\dicom_outlines"

# Import image loader
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from image_loader import MedicalImageLoader
except ImportError:
    print("Warning: Could not import MedicalImageLoader")
    class MedicalImageLoader:
        def __init__(self):
            self.image_array = None
        def load_file(self, path):
            return False, "Loader not available"
        def load_dicom_series(self, path):
            return False, "Loader not available"
        def get_slice(self, orientation, idx):
            return None
        def get_num_slices(self, orientation):
            return 0

# Check for scipy availability
SCIPY_AVAILABLE = False
try:
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, binary_dilation, label
    SCIPY_AVAILABLE = True
except ImportError:
    print("Info: scipy not available. Some features will be limited.")

# Check for scikit-image
SKIMAGE_AVAILABLE = False
try:
    from skimage import measure
    from skimage.morphology import binary_closing, remove_small_objects
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Info: scikit-image not available.")


# ============================================================================
# ORIENTATION CLASSIFICATION INTEGRATION
# ============================================================================

class OrientationClassificationThread(QThread):
    """Thread for orientation classification"""
    classification_complete = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, dicom_path, classifier):
        super().__init__()
        self.dicom_path = dicom_path
        self.classifier = classifier
    
    def run(self):
        """Run orientation classification"""
        try:
            result = self.classifier.predict_from_dicom(self.dicom_path)
            self.classification_complete.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DicomFileManagerWindow(QDialog):
    """Window for DICOM orientation classification"""
    
    file_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM Orientation Classification")
        self.setModal(False)
        self.resize(600, 500)
        
        self.current_file = None
        self.classifier = None
        
        self.init_ui()
        self.init_classifier()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🧭 DICOM Orientation Classifier")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4a9eff; margin: 10px;")
        layout.addWidget(title)
        
        # File selection
        file_group = QGroupBox("DICOM File Selection")
        file_layout = QVBoxLayout()
        
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #888; padding: 5px; border: 1px solid #555; border-radius: 3px;")
        file_select_layout.addWidget(self.file_path_label)
        
        browse_file_btn = QPushButton("Browse...")
        browse_file_btn.clicked.connect(self.select_dicom_file)
        file_select_layout.addWidget(browse_file_btn)
        
        file_layout.addLayout(file_select_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Classifier status
        status_group = QGroupBox("Classifier Status")
        status_layout = QVBoxLayout()
        
        self.classifier_status = QLabel("⏳ Initializing classifier...")
        self.classifier_status.setStyleSheet("color: #ffaa00; font-weight: bold; padding: 5px;")
        status_layout.addWidget(self.classifier_status)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Classification button
        self.classify_btn = QPushButton("🎯 Classify Orientation")
        self.classify_btn.setStyleSheet("font-weight: bold; padding: 12px; font-size: 12pt; background-color: #4a9eff;")
        self.classify_btn.clicked.connect(self.classify_orientation)
        self.classify_btn.setEnabled(False)
        layout.addWidget(self.classify_btn)
        
        # Results display
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setMinimumHeight(200)
        self.results_display.setPlaceholderText("Classification results will appear here...\n\nPlease select a DICOM file and click 'Classify Orientation'.")
        self.results_display.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt;")
        results_layout.addWidget(self.results_display)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        action_layout.addWidget(close_btn)
        
        layout.addLayout(action_layout)
    
    def select_dicom_file(self):
        """Select a DICOM file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DICOM File", "",
            "DICOM Files (*.dcm);;All Files (*.*)"
        )
        if file_path:
            self.current_file = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setStyleSheet("color: #44ff44; padding: 5px; border: 1px solid #555; border-radius: 3px;")
            
            if self.classifier:
                self.classify_btn.setEnabled(True)
            
            self.results_display.clear()
    
    def init_classifier(self):
        """Initialize the orientation classifier"""
        try:
            # Default paths
            model_path = r"D:\task2\orientation code\mri_orientation_finetuned.pth"
            foundation_path = r"D:\task2\orientation code\mri_foundation-master"
            
            if not os.path.exists(model_path):
                self.classifier_status.setText("❌ Model file not found")
                self.classifier_status.setStyleSheet("color: #ff4444; font-weight: bold; padding: 5px;")
                self.results_display.append(f"ERROR: Model checkpoint not found at:\n{model_path}\n\nPlease ensure the model file exists.")
                return
            
            self.classifier_status.setText("⏳ Loading classifier...")
            self.classifier_status.setStyleSheet("color: #ffaa00; font-weight: bold; padding: 5px;")
            QApplication.processEvents()
            
            # Import required modules
            import torch
            import torch.nn as nn
            from torchvision import transforms
            import pydicom
            
            # Add foundation path to sys.path
            if foundation_path and os.path.exists(foundation_path):
                sys.path.insert(0, foundation_path)
            
            # Define classifier class inline
            class MRIOrientationClassifier:
                def __init__(self, checkpoint_path, device='cpu', image_size=(1024, 1024)):
                    self.checkpoint_path = checkpoint_path
                    self.image_size = image_size
                    self.label_map = {0: 'Axial', 1: 'Coronal', 2: 'Sagittal'}
                    self.device = torch.device(device)
                    
                    # Load model
                    try:
                        from models.sam import sam_model_registry
                        import cfg
                        
                        args = cfg.parse_args()
                        args.if_encoder_adapter = False
                        args.if_mask_decoder_adapter = False
                        args.num_cls = 3
                        args.image_size = image_size[0]
                        
                        if not hasattr(args, 'decoder_adapt_depth'):
                            args.decoder_adapt_depth = 1
                        if not hasattr(args, 'encoder_adapter_stride'):
                            args.encoder_adapter_stride = 16
                        if not hasattr(args, 'encoder_adapter_input_dim'):
                            args.encoder_adapter_input_dim = 768
                        if not hasattr(args, 'encoder_adapter_output_dim'):
                            args.encoder_adapter_output_dim = 256
                        
                        base_model = sam_model_registry["vit_b"](
                            args,
                            checkpoint=None,
                            num_classes=3,
                            image_size=image_size[0],
                            pretrained_sam=False
                        )
                        
                        # Wrapper class
                        class MRISAMClassifier(nn.Module):
                            def __init__(self, base_model, num_classes=3):
                                super().__init__()
                                self.image_encoder = base_model.image_encoder
                                
                                with torch.no_grad():
                                    dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
                                    encoder_out = self.image_encoder(dummy_input)
                                    encoder_dim = encoder_out.shape[1]
                                
                                self.classification_head = nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(encoder_dim, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, num_classes)
                                )
                            
                            def forward(self, x):
                                features = self.image_encoder(x)
                                logits = self.classification_head(features)
                                return logits
                        
                        self.model = MRISAMClassifier(base_model, num_classes=3)
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        self.model.load_state_dict(checkpoint)
                        
                    except Exception as e:
                        print(f"Error loading SAM model: {e}")
                        from torchvision.models import resnet50
                        self.model = resnet50(pretrained=False)
                        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        self.model.load_state_dict(checkpoint, strict=False)
                    
                    self.model.to(self.device)
                    self.model.eval()
                    
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                    ])
                
                def predict_from_dicom(self, dicom_path):
                    dicom_data = pydicom.dcmread(dicom_path)
                    image_array = dicom_data.pixel_array.astype(float)
                    image_array -= image_array.min()
                    image_array /= (image_array.max() + 1e-8)
                    
                    return self.predict_from_array(image_array)
                
                def predict_from_array(self, image_array):
                    from PIL import Image
                    
                    if image_array.dtype == np.uint8:
                        img = Image.fromarray(image_array, mode='L')
                    else:
                        img_uint8 = (image_array * 255).astype(np.uint8)
                        img = Image.fromarray(img_uint8, mode='L')
                    
                    img_rgb = img.convert('RGB')
                    
                    if img_rgb.size != self.image_size:
                        img_rgb = img_rgb.resize(self.image_size, Image.BILINEAR)
                    
                    img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_label = torch.max(probabilities, 1)
                    
                    predicted_label = predicted_label.item()
                    confidence = confidence.item()
                    orientation = self.label_map[predicted_label]
                    
                    return {
                        'orientation': orientation,
                        'label': predicted_label,
                        'confidence': confidence,
                        'all_probabilities': {
                            self.label_map[i]: probabilities[0, i].item()
                            for i in range(3)
                        }
                    }
            
            # Create classifier instance
            self.classifier = MRIOrientationClassifier(
                checkpoint_path=model_path,
                device='cpu'
            )
            
            self.classifier_status.setText("✅ Classifier Ready")
            self.classifier_status.setStyleSheet("color: #44ff44; font-weight: bold; padding: 5px;")
            
            if self.current_file:
                self.classify_btn.setEnabled(True)
            
            self.results_display.append("✓ Orientation classifier loaded successfully!\n")
            self.results_display.append(f"Model: {os.path.basename(model_path)}\n")
            self.results_display.append("Ready to classify DICOM images.\n")
            
        except Exception as e:
            self.classifier_status.setText("❌ Classifier Load Failed")
            self.classifier_status.setStyleSheet("color: #ff4444; font-weight: bold; padding: 5px;")
            self.results_display.append(f"ERROR: Failed to load classifier\n{str(e)}\n")
    
    def classify_orientation(self):
        """Classify the orientation of the selected DICOM file"""
        if not self.current_file:
            self.results_display.append("ERROR: No DICOM file selected.\n")
            return
        
        if not self.classifier:
            self.results_display.append("ERROR: Classifier not loaded.\n")
            return
        
        # Clear previous results
        self.results_display.clear()
        self.results_display.append("="*60)
        self.results_display.append(f"📁 File: {os.path.basename(self.current_file)}")
        self.results_display.append("="*60)
        self.results_display.append("\n⏳ Analyzing image orientation...\n")
        QApplication.processEvents()
        
        try:
            # Run classification
            result = self.classifier.predict_from_dicom(self.current_file)
            
            # Display results
            self.results_display.append("✅ CLASSIFICATION COMPLETE\n")
            self.results_display.append("="*60)
            self.results_display.append(f"\n🎯 PREDICTED ORIENTATION: {result['orientation']}")
            self.results_display.append(f"✓  Confidence: {result['confidence']*100:.2f}%\n")
            self.results_display.append("="*60)
            self.results_display.append("\n📊 Probability Distribution:\n")
            
            for orient, prob in result['all_probabilities'].items():
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                self.results_display.append(f"  {orient:10s}: [{bar}] {prob*100:5.2f}%")
            
            self.results_display.append("\n" + "="*60)
            
            # Emit signal to potentially load in main viewer
            self.file_selected.emit(self.current_file)
            
        except Exception as e:
            self.results_display.append(f"\n❌ ERROR: Classification failed\n")
            self.results_display.append(f"Error details: {str(e)}\n")
            self.results_display.append("="*60)


# ============================================================================
# EXISTING CLASSES (GeminiDetectionThread, etc.)
# ============================================================================

class GeminiDetectionThread(QThread):
    """Thread for Gemini AI-based organ detection"""
    detection_complete = pyqtSignal(str, str)
    error = pyqtSignal(str)
    
    def __init__(self, volume, roi_start, roi_end):
        super().__init__()
        self.volume = volume
        self.roi_start = roi_start
        self.roi_end = roi_end
    
    def run(self):
        """Detect organ using Gemini API"""
        try:
            roi_volume = self.volume[self.roi_start:self.roi_end+1, :, :]
            
            middle_slice_idx = roi_volume.shape[0] // 2
            slice_data = roi_volume[middle_slice_idx, :, :]
            
            normalized_data = ((slice_data - np.min(slice_data)) / 
                             (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(normalized_data).convert("RGB")
            
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = """
            Analyze the provided medical image. Identify the primary human organ visible.
            If an organ is detected, state its name clearly.
            If no organ is visible or the image is not medical in nature, state 'No organ detected'.
            Format the output exactly like this:
            Organ: [Organ Name]
            
            Additional details (if applicable):
            - Brief description of what you see
            - Any notable features
            """
            
            response = model.generate_content([prompt, pil_img])
            response_text = response.text
            
            organ_name = "Unknown"
            for line in response_text.split('\n'):
                if line.strip().startswith("Organ:"):
                    organ_name = line.split(":", 1)[1].strip()
                    break
            
            self.detection_complete.emit(organ_name, response_text)
            
        except Exception as e:
            self.error.emit(f"Detection failed: {str(e)}")


class ObliqueLineHandle:
    """Handle for controlling oblique line endpoints"""
    def __init__(self, position, radius=8):
        self.position = position
        self.radius = radius
        self.is_hovered = False
        self.is_dragging = False
    
    def contains(self, point):
        dx = point.x() - self.position.x()
        dy = point.y() - self.position.y()
        return math.sqrt(dx*dx + dy*dy) <= self.radius
    
    def draw(self, painter):
        if self.is_dragging:
            painter.setBrush(QBrush(QColor(255, 100, 100, 200)))
        elif self.is_hovered:
            painter.setBrush(QBrush(QColor(255, 200, 100, 200)))
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
        
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2))
        painter.drawEllipse(self.position, self.radius, self.radius)


class SlicerViewer(QLabel):
    """3D Slicer-style viewer with reference lines"""
    
    position_changed = pyqtSignal(int, int)
    slice_changed = pyqtSignal(int)
    oblique_line_changed = pyqtSignal(QPointF, QPointF, float)
    
    def __init__(self, orientation, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000000; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        
        self.reference_position = [0, 0, 0]
        self.show_reference_lines = True
        
        self.show_oblique_line = False
        self.oblique_start = QPointF(100, 200)
        self.oblique_end = QPointF(300, 200)
        self.oblique_handle_start = ObliqueLineHandle(self.oblique_start)
        self.oblique_handle_end = ObliqueLineHandle(self.oblique_end)
        self.active_handle = None
        
        self.current_slice = None
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.window = 400
        self.level = 40
        
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_oblique_line_visible(self, visible):
        self.show_oblique_line = visible
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def update_oblique_line(self, start, end):
        self.oblique_start = start
        self.oblique_end = end
        self.oblique_handle_start.position = start
        self.oblique_handle_end.position = end
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
        
    def set_reference_position(self, position):
        self.reference_position = position
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
        
    def display_slice(self, slice_array, apply_wl=True):
        if slice_array is None:
            self.setText("No Image")
            return
        
        self.current_slice = slice_array
        
        if apply_wl and slice_array.dtype != np.uint8:
            min_val = self.level - self.window / 2
            max_val = self.level + self.window / 2
            windowed = np.clip(slice_array, min_val, max_val)
            normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            if slice_array.max() > 255 or slice_array.min() < 0:
                normalized = ((slice_array - slice_array.min()) / 
                            (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
            else:
                normalized = slice_array.astype(np.uint8)
        
        height, width = normalized.shape
        q_image = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
        self.original_pixmap = QPixmap.fromImage(q_image)
        
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        if self.show_reference_lines or (self.show_oblique_line and self.orientation == 'axial'):
            scaled_pixmap = self.draw_overlays(scaled_pixmap)
        
        self.setPixmap(scaled_pixmap)
    
    def draw_overlays(self, pixmap):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = pixmap.width()
        height = pixmap.height()
        
        if self.show_reference_lines:
            colors = {
                'axial': QColor(255, 0, 0, 180),
                'sagittal': QColor(0, 255, 0, 180),
                'coronal': QColor(255, 255, 0, 180)
            }
            
            line_thickness = 2
            
            if self.orientation == 'axial':
                x = int(self.reference_position[2] * self.zoom_factor)
                pen = QPen(colors['sagittal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                y = int(self.reference_position[1] * self.zoom_factor)
                pen = QPen(colors['coronal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
            elif self.orientation == 'sagittal':
                y = int(self.reference_position[0] * self.zoom_factor)
                pen = QPen(colors['axial'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                x = int(self.reference_position[1] * self.zoom_factor)
                pen = QPen(colors['coronal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
            elif self.orientation == 'coronal':
                y = int(self.reference_position[0] * self.zoom_factor)
                pen = QPen(colors['axial'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                x = int(self.reference_position[2] * self.zoom_factor)
                pen = QPen(colors['sagittal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
        
        if self.show_oblique_line and self.orientation == 'axial':
            start_scaled = QPointF(
                self.oblique_start.x() * self.zoom_factor,
                self.oblique_start.y() * self.zoom_factor
            )
            end_scaled = QPointF(
                self.oblique_end.x() * self.zoom_factor,
                self.oblique_end.y() * self.zoom_factor
            )
            
            pen = QPen(QColor(255, 0, 255, 220), 3)
            painter.setPen(pen)
            painter.drawLine(start_scaled, end_scaled)
            
            handle_radius = 10
            
            if self.oblique_handle_start.is_dragging:
                painter.setBrush(QBrush(QColor(255, 100, 100, 220)))
            elif self.oblique_handle_start.is_hovered:
                painter.setBrush(QBrush(QColor(255, 200, 100, 220)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(0, 0, 0, 255), 2))
            painter.drawEllipse(start_scaled, handle_radius, handle_radius)
            
            if self.oblique_handle_end.is_dragging:
                painter.setBrush(QBrush(QColor(255, 100, 100, 220)))
            elif self.oblique_handle_end.is_hovered:
                painter.setBrush(QBrush(QColor(255, 200, 100, 220)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(0, 0, 0, 255), 2))
            painter.drawEllipse(end_scaled, handle_radius, handle_radius)
            
            dx = self.oblique_end.x() - self.oblique_start.x()
            dy = self.oblique_end.y() - self.oblique_start.y()
            angle = math.degrees(math.atan2(dy, dx))
            
            mid_x = (start_scaled.x() + end_scaled.x()) / 2
            mid_y = (start_scaled.y() + end_scaled.y()) / 2
            
            font = QFont("Arial", 10, QFont.Bold)
            painter.setFont(font)
            text = f"Oblique: {angle:.1f}°"
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(text)
            text_rect.moveCenter(QPoint(int(mid_x + 10), int(mid_y - 20)))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
            painter.drawRect(text_rect.adjusted(-3, -2, 3, 2))
            
            painter.setPen(QPen(QColor(255, 0, 255), 1))
            painter.drawText(text_rect, Qt.AlignCenter, text)
        
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(10, 20, self.orientation.upper())
        
        painter.end()
        return pixmap
    
    def widget_to_image_coords(self, widget_pos):
        if self.original_pixmap is None:
            return None
        
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        
        pixmap_rect = pixmap.rect()
        offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
        offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
        
        pixmap_x = widget_pos.x() - offset_x
        pixmap_y = widget_pos.y() - offset_y
        
        if not pixmap_rect.contains(int(pixmap_x), int(pixmap_y)):
            return None
        
        img_x = pixmap_x / self.zoom_factor
        img_y = pixmap_y / self.zoom_factor
        
        img_x = max(0, min(img_x, self.original_pixmap.width() - 1))
        img_y = max(0, min(img_y, self.original_pixmap.height() - 1))
        
        return QPointF(img_x, img_y)
    
    def mousePressEvent(self, event: QMouseEvent):
        if self.original_pixmap is None:
            return
        
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            return
        
        if self.show_oblique_line and self.orientation == 'axial' and event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap:
                pixmap_rect = pixmap.rect()
                offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
                offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
                
                start_widget = QPointF(
                    self.oblique_start.x() * self.zoom_factor + offset_x,
                    self.oblique_start.y() * self.zoom_factor + offset_y
                )
                end_widget = QPointF(
                    self.oblique_end.x() * self.zoom_factor + offset_x,
                    self.oblique_end.y() * self.zoom_factor + offset_y
                )
                
                self.oblique_handle_start.position = start_widget
                self.oblique_handle_end.position = end_widget
                
                if self.oblique_handle_start.contains(event.pos()):
                    self.active_handle = self.oblique_handle_start
                    self.active_handle.is_dragging = True
                    return
                elif self.oblique_handle_end.contains(event.pos()):
                    self.active_handle = self.oblique_handle_end
                    self.active_handle.is_dragging = True
                    return
        
        if event.button() == Qt.LeftButton:
            img_coords = self.widget_to_image_coords(event.pos())
            if img_coords:
                self.position_changed.emit(int(img_coords.x()), int(img_coords.y()))
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            if self.current_slice is not None:
                self.display_slice(self.current_slice)
            return
        
        if self.active_handle and self.active_handle.is_dragging:
            img_coords = self.widget_to_image_coords(event.pos())
            if img_coords:
                if self.active_handle == self.oblique_handle_start:
                    self.oblique_start = img_coords
                else:
                    self.oblique_end = img_coords
                
                dx = self.oblique_end.x() - self.oblique_start.x()
                dy = self.oblique_end.y() - self.oblique_start.y()
                angle = math.degrees(math.atan2(dy, dx))
                self.oblique_line_changed.emit(self.oblique_start, self.oblique_end, angle)
                
                if self.current_slice is not None:
                    self.display_slice(self.current_slice)
            return
        
        if self.show_oblique_line and self.orientation == 'axial':
            pixmap = self.pixmap()
            if pixmap:
                pixmap_rect = pixmap.rect()
                offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
                offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
                
                start_widget = QPointF(
                    self.oblique_start.x() * self.zoom_factor + offset_x,
                    self.oblique_start.y() * self.zoom_factor + offset_y
                )
                end_widget = QPointF(
                    self.oblique_end.x() * self.zoom_factor + offset_x,
                    self.oblique_end.y() * self.zoom_factor + offset_y
                )
                
                self.oblique_handle_start.position = start_widget
                self.oblique_handle_end.position = end_widget
                
                start_hovered = self.oblique_handle_start.contains(event.pos())
                end_hovered = self.oblique_handle_end.contains(event.pos())
                
                if start_hovered != self.oblique_handle_start.is_hovered or \
                   end_hovered != self.oblique_handle_end.is_hovered:
                    self.oblique_handle_start.is_hovered = start_hovered
                    self.oblique_handle_end.is_hovered = end_hovered
                    if self.current_slice is not None:
                        self.display_slice(self.current_slice)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
        elif event.button() == Qt.LeftButton and self.active_handle:
            self.active_handle.is_dragging = False
            self.active_handle = None
            if self.current_slice is not None:
                self.display_slice(self.current_slice)
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.slice_changed.emit(1)
        else:
            self.slice_changed.emit(-1)
    
    def set_zoom(self, zoom):
        self.zoom_factor = zoom
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def set_window_level(self, window, level):
        self.window = window
        self.level = level
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def reset_pan(self):
        self.pan_offset = QPoint(0, 0)
        if self.current_slice is not None:
            self.display_slice(self.current_slice)


class ObliqueViewer(QLabel):
    """Oblique plane viewer with rotation capability"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        self.setText("Oblique View\n(Define line in Axial view)")
        self.setStyleSheet("background-color: #1a1a1a; color: #888; border: 2px solid #2a2a2a; font-size: 12pt;")
        
        self.volume_data = None
        self.oblique_start = QPointF(100, 200)
        self.oblique_end = QPointF(300, 200)
        self.current_axial_slice = 0
        self.zoom_factor = 1.0
        self.window = 400
        self.level = 40
        self.rotation_angle = 0
        
        self.roi_start = 0
        self.roi_end = 0
        self.roi_active = False
        
    def set_volume(self, volume):
        self.volume_data = volume
    
    def set_rotation(self, angle):
        self.rotation_angle = angle
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)
    
    def set_roi(self, roi_start, roi_end, roi_active):
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.roi_active = roi_active

    def update_oblique(self, start, end, axial_slice):
        if self.volume_data is None:
            return
        
        self.oblique_start = start
        self.oblique_end = end
        self.current_axial_slice = axial_slice
        
        oblique_slice = self.extract_oblique_plane()
        if oblique_slice is not None:
            self.display_slice(oblique_slice)
    
    def extract_oblique_plane(self):
        if self.volume_data is None:
            return None
        
        x1, y1 = int(self.oblique_start.x()), int(self.oblique_start.y())
        x2, y2 = int(self.oblique_end.x()), int(self.oblique_end.y())
        
        line_length = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if line_length == 0:
            return None
        
        if self.roi_active:
            volume = self.volume_data[self.roi_start:self.roi_end+1, :, :]
            depth = self.roi_end - self.roi_start + 1
        else:
            volume = self.volume_data
            depth = self.volume_data.shape[0]
        
        _, height, width = volume.shape
        
        oblique_plane = np.zeros((depth, line_length))
        
        for i in range(line_length):
            t = i / line_length
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            oblique_plane[:, i] = volume[:, y, x]
        
        if self.rotation_angle != 0 and SCIPY_AVAILABLE:
            oblique_plane = ndimage.rotate(oblique_plane, self.rotation_angle, reshape=False, order=1)
        
        return oblique_plane
    
    def display_slice(self, slice_array):
        if slice_array is None:
            return
        
        min_val = self.level - self.window / 2
        max_val = self.level + self.window / 2
        windowed = np.clip(slice_array, min_val, max_val)
        normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        height, width = normalized.shape
        q_image = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        
        pixmap = self.draw_crosshair(pixmap)
        
        scaled = pixmap.scaled(
            int(pixmap.width() * self.zoom_factor),
            int(pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled)
    
    def draw_crosshair(self, pixmap):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.roi_active:
            y = self.current_axial_slice - self.roi_start
        else:
            y = self.current_axial_slice
        
        if 0 <= y < pixmap.height():
            pen = QPen(QColor(255, 0, 0, 180), 2)
            painter.setPen(pen)
            painter.drawLine(0, y, pixmap.width(), y)
        
        painter.setPen(QPen(QColor(255, 0, 255), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        
        if self.roi_active:
            if self.rotation_angle != 0:
                painter.drawText(10, 20, f"OBLIQUE [ROI] (Rot: {self.rotation_angle:.0f}°)")
            else:
                painter.drawText(10, 20, "OBLIQUE [ROI]")
        else:
            if self.rotation_angle != 0:
                painter.drawText(10, 20, f"OBLIQUE (Rot: {self.rotation_angle:.0f}°)")
            else:
                painter.drawText(10, 20, "OBLIQUE")
        
        painter.end()
        return pixmap
    
    def set_zoom(self, zoom):
        self.zoom_factor = zoom
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)
    
    def set_window_level(self, window, level):
        self.window = window
        self.level = level
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)


class BrainSurfaceViewer(QLabel):
    """Brain Surface viewer - displays brain surface DICOM"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        self.setText("Brain Surface View\n(Click 'Load Brain Surface' to display)")
        self.setStyleSheet("background-color: #1a1a1a; color: #888; border: 2px solid #2a2a2a; font-size: 12pt;")
        
        self.brain_loader = MedicalImageLoader()
        self.current_slice = 0
        self.zoom_factor = 1.0
        self.window = 400
        self.level = 40
        
    def load_brain_surface(self, path):
        """Load brain surface DICOM"""
        if os.path.isdir(path):
            success, message = self.brain_loader.load_dicom_series(path)
        else:
            success, message = self.brain_loader.load_file(path)
        
        if success and self.brain_loader.image_array is not None:
            self.current_slice = self.brain_loader.get_num_slices('axial') // 2
            self.display_current_slice()
            return True, "Brain surface loaded successfully"
        else:
            return False, message
    
    def display_current_slice(self):
        """Display current slice"""
        if self.brain_loader.image_array is None:
            return
        
        slice_data = self.brain_loader.get_slice('axial', self.current_slice)
        if slice_data is None:
            return
        
        # Apply window/level
        min_val = self.level - self.window / 2
        max_val = self.level + self.window / 2
        windowed = np.clip(slice_data, min_val, max_val)
        normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        height, width = normalized.shape
        q_image = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        
        # Add label
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(100, 200, 255), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        
        total = self.brain_loader.get_num_slices('axial')
        painter.drawText(10, 20, f"BRAIN SURFACE [{self.current_slice + 1}/{total}]")
        painter.end()
        
        # Apply zoom
        scaled = pixmap.scaled(
            int(pixmap.width() * self.zoom_factor),
            int(pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled)
    
    def navigate_slice(self, direction):
        """Navigate through slices"""
        if self.brain_loader.image_array is None:
            return
        
        max_slices = self.brain_loader.get_num_slices('axial')
        self.current_slice = max(0, min(self.current_slice + direction, max_slices - 1))
        self.display_current_slice()
    
    def set_zoom(self, zoom):
        """Set zoom factor"""
        self.zoom_factor = zoom
        self.display_current_slice()
    
    def set_window_level(self, window, level):
        """Set window/level"""
        self.window = window
        self.level = level
        self.display_current_slice()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for slice navigation"""
        if self.brain_loader.image_array is not None:
            delta = event.angleDelta().y()
            if delta > 0:
                self.navigate_slice(1)
            else:
                self.navigate_slice(-1)


class ROIDialog(QDialog):
    """Dialog for ROI slice selection"""
    
    def __init__(self, max_slices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Slice Selection")
        self.setModal(True)
        self.resize(350, 200)
        
        layout = QVBoxLayout(self)
        
        info = QLabel(f"Select slice range (Total: {max_slices} slices)")
        info.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info)
        
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Slice:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max_slices - 1)
        self.start_spin.setValue(0)
        self.start_spin.setMinimumWidth(100)
        start_layout.addWidget(self.start_spin)
        start_layout.addStretch()
        layout.addLayout(start_layout)
        
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Slice:"))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, max_slices - 1)
        self.end_spin.setValue(max_slices - 1)
        self.end_spin.setMinimumWidth(100)
        end_layout.addWidget(self.end_spin)
        end_layout.addStretch()
        layout.addLayout(end_layout)
        
        self.preview_label = QLabel(f"Selected: {max_slices} slices")
        self.preview_label.setStyleSheet("color: #44ff44; font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.preview_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.start_spin.valueChanged.connect(self.update_preview)
        self.end_spin.valueChanged.connect(self.update_preview)
        
    def update_preview(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        count = max(0, end - start + 1)
        self.preview_label.setText(f"Selected: {count} slices (from {start} to {end})")
        
    def get_roi(self):
        return self.start_spin.value(), self.end_spin.value()


class MainWindow(QMainWindow):
    """3D Slicer-Style Medical Viewer - DICOM with Orientation Classification"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Slicer Medical Viewer - DICOM + AI Orientation")
        self.setGeometry(50, 50, 1800, 1000)
        
        self.image_loader = MedicalImageLoader()
        
        self.reference_position = [0, 0, 0]
        
        self.roi_start = 0
        self.roi_end = 0
        self.roi_active = False
        
        self.fourth_window_mode = "oblique"
        
        self.detected_organ = "Unknown"
        self.gemini_response = ""
        
        self.is_playing = False
        self.playback_speed = 100
        self.playback_timer = QTimer(self)
        
        self.dicom_manager = None
        
        self.init_ui()
        self.create_menu_bar()
        self.create_status_bar()
        
        # Connect playback timer after UI is initialized
        self.playback_timer.timeout.connect(self.advance_slice)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel)
        
        viewer_widget = QWidget()
        viewer_layout = QGridLayout(viewer_widget)
        viewer_layout.setSpacing(3)
        
        self.axial_panel = self.create_viewer_panel("Axial (Red)", "axial")
        self.sagittal_panel = self.create_viewer_panel("Sagittal (Green)", "sagittal")
        self.coronal_panel = self.create_viewer_panel("Coronal (Yellow)", "coronal")
        self.fourth_panel = self.create_fourth_panel()
        
        viewer_layout.addWidget(self.axial_panel, 0, 0)
        viewer_layout.addWidget(self.sagittal_panel, 0, 1)
        viewer_layout.addWidget(self.coronal_panel, 1, 0)
        viewer_layout.addWidget(self.fourth_panel, 1, 1)
        
        main_layout.addWidget(viewer_widget, stretch=1)
    
    def create_viewer_panel(self, title, orientation):
        panel = QGroupBox()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        
        top_bar = QHBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        
        if orientation == 'axial':
            title_label.setStyleSheet("color: #ff4444; font-size: 11pt;")
        elif orientation == 'sagittal':
            title_label.setStyleSheet("color: #44ff44; font-size: 11pt;")
        elif orientation == 'coronal':
            title_label.setStyleSheet("color: #ffff44; font-size: 11pt;")
            
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        
        slice_info = QLabel("Slice: 0/0")
        slice_info.setObjectName(f"{orientation}_info")
        top_bar.addWidget(slice_info)
        layout.addLayout(top_bar)
        
        viewer = SlicerViewer(orientation)
        viewer.setObjectName(f"{orientation}_viewer")
        viewer.position_changed.connect(lambda x, y: self.update_reference_from_view(orientation, x, y))
        viewer.slice_changed.connect(lambda d: self.navigate_slice(orientation, d))
        
        if orientation == 'axial':
            viewer.oblique_line_changed.connect(self.update_oblique_view)
        
        layout.addWidget(viewer)
        
        controls = QHBoxLayout()
        
        zoom_label = QLabel("Zoom:")
        controls.addWidget(zoom_label)
        
        zoom_spin = QDoubleSpinBox()
        zoom_spin.setRange(0.1, 5.0)
        zoom_spin.setSingleStep(0.1)
        zoom_spin.setValue(1.0)
        zoom_spin.setPrefix("×")
        zoom_spin.setObjectName(f"{orientation}_zoom")
        zoom_spin.valueChanged.connect(lambda v: self.set_view_zoom(orientation, v))
        controls.addWidget(zoom_spin)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(lambda: self.reset_single_view(orientation))
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        return panel
    
    def create_fourth_panel(self):
        """Create fourth panel with mode selection"""
        panel = QGroupBox()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Top bar with mode selection
        top_bar = QHBoxLayout()
        
        mode_label = QLabel("<b>View Mode:</b>")
        mode_label.setStyleSheet("color: #ff00ff; font-size: 10pt;")
        top_bar.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Oblique View", "Brain Surface"])
        self.mode_combo.currentTextChanged.connect(self.switch_fourth_mode)
        top_bar.addWidget(self.mode_combo)
        
        top_bar.addStretch()
        layout.addLayout(top_bar)
        
        # Stacked widget to hold both viewers
        self.fourth_stack = QWidget()
        stack_layout = QVBoxLayout(self.fourth_stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        
        # Oblique viewer
        self.oblique_viewer = ObliqueViewer()
        
        # Brain surface viewer
        self.brain_surface_viewer = BrainSurfaceViewer()
        
        # Add oblique viewer by default
        stack_layout.addWidget(self.oblique_viewer)
        self.oblique_viewer.setVisible(True)
        stack_layout.addWidget(self.brain_surface_viewer)
        self.brain_surface_viewer.setVisible(False)
        
        layout.addWidget(self.fourth_stack)
        
        # Controls
        controls = QHBoxLayout()
        
        # Oblique-specific controls
        self.rotation_x_label = QLabel("Rot X:")
        controls.addWidget(self.rotation_x_label)
        
        self.rotation_x_spin = QSpinBox()
        self.rotation_x_spin.setRange(-180, 180)
        self.rotation_x_spin.setValue(0)
        self.rotation_x_spin.setSuffix("°")
        self.rotation_x_spin.valueChanged.connect(self.update_rotation)
        controls.addWidget(self.rotation_x_spin)
        
        # Brain surface button
        self.load_brain_btn = QPushButton("🧠 Load Brain Surface")
        self.load_brain_btn.setStyleSheet("font-weight: bold; background-color: #5a5aff;")
        self.load_brain_btn.clicked.connect(self.load_brain_surface_to_viewer)
        self.load_brain_btn.setVisible(False)
        controls.addWidget(self.load_brain_btn)
        
        # Common zoom control
        self.fourth_zoom_label = QLabel("Zoom:")
        controls.addWidget(self.fourth_zoom_label)
        
        self.fourth_zoom_spin = QDoubleSpinBox()
        self.fourth_zoom_spin.setRange(0.1, 5.0)
        self.fourth_zoom_spin.setSingleStep(0.1)
        self.fourth_zoom_spin.setValue(1.0)
        self.fourth_zoom_spin.setPrefix("×")
        self.fourth_zoom_spin.valueChanged.connect(self.set_fourth_zoom)
        controls.addWidget(self.fourth_zoom_spin)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        return panel
    
    def switch_fourth_mode(self, mode_text):
        """Switch between oblique and brain surface modes"""
        if mode_text == "Oblique View":
            self.fourth_window_mode = "oblique"
            self.oblique_viewer.setVisible(True)
            self.brain_surface_viewer.setVisible(False)
            
            # Show oblique controls, hide brain surface controls
            self.rotation_x_label.setVisible(True)
            self.rotation_x_spin.setVisible(True)
            self.load_brain_btn.setVisible(False)
            
            self.statusBar().showMessage("Switched to Oblique View mode")
            
        elif mode_text == "Brain Surface":
            self.fourth_window_mode = "brain_surface"
            self.oblique_viewer.setVisible(False)
            self.brain_surface_viewer.setVisible(True)
            
            # Hide oblique controls, show brain surface controls
            self.rotation_x_label.setVisible(False)
            self.rotation_x_spin.setVisible(False)
            self.load_brain_btn.setVisible(True)
            
            self.statusBar().showMessage("Switched to Brain Surface mode - Click 'Load Brain Surface' button")
    
    def load_brain_surface_to_viewer(self):
        """Load brain surface into the brain surface viewer"""
        if not os.path.exists(BRAIN_SURFACE_DICOM_PATH):
            QMessageBox.warning(
                self, 
                "File Not Found", 
                f"Brain surface DICOM file not found at:\n{BRAIN_SURFACE_DICOM_PATH}\n\n"
                f"Please update the BRAIN_SURFACE_DICOM_PATH variable in the code."
            )
            return
        
        try:
            success, message = self.brain_surface_viewer.load_brain_surface(BRAIN_SURFACE_DICOM_PATH)
            
            if success:
                self.statusBar().showMessage(f"✓ Brain surface loaded successfully")
                QMessageBox.information(
                    self,
                    "Brain Surface Loaded",
                    f"Successfully loaded brain surface DICOM.\n\nPath: {BRAIN_SURFACE_DICOM_PATH}\n\nUse mouse wheel to navigate slices."
                )
            else:
                QMessageBox.critical(self, "Load Error", message)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load brain surface:\n{str(e)}"
            )
    
    def update_rotation(self):
        """Update oblique rotation"""
        x_rot = self.rotation_x_spin.value()
        self.oblique_viewer.set_rotation(x_rot)
    
    def set_fourth_zoom(self, zoom):
        """Set zoom for fourth viewer"""
        if self.fourth_window_mode == "oblique":
            self.oblique_viewer.set_zoom(zoom)
        elif self.fourth_window_mode == "brain_surface":
            self.brain_surface_viewer.set_zoom(zoom)
    
    def update_oblique_view(self, start, end, angle):
        """Update oblique view"""
        if self.image_loader.image_array is not None:
            self.oblique_viewer.update_oblique(start, end, self.reference_position[0])
            
            self.statusBar().showMessage(
                f"Oblique line: angle={angle:.1f}°, "
                f"start=({start.x():.0f}, {start.y():.0f}), "
                f"end=({end.x():.0f}, {end.y():.0f})"
            )
    
    def open_dicom_manager(self):
        """Open DICOM file manager window"""
        if self.dicom_manager is None:
            self.dicom_manager = DicomFileManagerWindow(self)
            self.dicom_manager.file_selected.connect(self.load_file_from_manager)
        
        self.dicom_manager.show()
        self.dicom_manager.raise_()
        self.dicom_manager.activateWindow()
    
    def load_file_from_manager(self, file_path):
        """Load file selected from DICOM manager"""
        self.load_image(file_path)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # File group
        file_group = QGroupBox("📁 File Operations")
        file_layout = QVBoxLayout()
        
        open_manager_btn = QPushButton("🗂️ Open DICOM Manager")
        open_manager_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        open_manager_btn.clicked.connect(self.open_dicom_manager)
        file_layout.addWidget(open_manager_btn)
        
        open_series_btn = QPushButton("📂 Quick Load DICOM Series")
        open_series_btn.clicked.connect(self.open_dicom_series)
        file_layout.addWidget(open_series_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # AI Organ Detection group
        ai_group = QGroupBox("🤖AI Detection")
        ai_layout = QVBoxLayout()
        
        detect_btn = QPushButton("🔍 Detect Main Organ ")
        detect_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        detect_btn.clicked.connect(self.detect_organ_gemini)
        ai_layout.addWidget(detect_btn)
        
        self.organ_label = QLabel("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        ai_layout.addWidget(self.organ_label)
        
        self.organ_details = QTextEdit()
        self.organ_details.setReadOnly(True)
        self.organ_details.setMaximumHeight(100)
        self.organ_details.setPlaceholderText("AI detection results will appear here...")
        self.organ_details.setStyleSheet("background-color: #1a1a1a; color: #aaa; font-size: 9pt;")
        ai_layout.addWidget(self.organ_details)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # ROI group
        roi_group = QGroupBox("📐 ROI Slice Selection")
        roi_layout = QVBoxLayout()
        
        roi_select_btn = QPushButton("Select ROI Slices")
        roi_select_btn.clicked.connect(self.select_roi)
        roi_layout.addWidget(roi_select_btn)
        
        self.roi_status = QLabel("ROI: Not Set")
        self.roi_status.setStyleSheet("color: #888;")
        roi_layout.addWidget(self.roi_status)
        
        roi_btns = QHBoxLayout()
        apply_roi_btn = QPushButton("Apply ROI")
        apply_roi_btn.clicked.connect(self.apply_roi)
        roi_btns.addWidget(apply_roi_btn)
        
        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self.clear_roi)
        roi_btns.addWidget(clear_roi_btn)
        roi_layout.addLayout(roi_btns)
        
        export_roi_btn = QPushButton("Export ROI Volume")
        export_roi_btn.clicked.connect(self.export_roi)
        roi_layout.addWidget(export_roi_btn)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # Navigation
        nav_group = QGroupBox("🎮 Navigation")
        nav_layout = QVBoxLayout()
        
        nav_layout.addWidget(QLabel("Axial (Red):"))
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.valueChanged.connect(lambda v: self.set_slice('axial', v))
        nav_layout.addWidget(self.axial_slider)
        
        nav_layout.addWidget(QLabel("Sagittal (Green):"))
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(lambda v: self.set_slice('sagittal', v))
        nav_layout.addWidget(self.sagittal_slider)
        
        nav_layout.addWidget(QLabel("Coronal (Yellow):"))
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(lambda v: self.set_slice('coronal', v))
        nav_layout.addWidget(self.coronal_slider)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Playback
        playback_group = QGroupBox("▶ Playback")
        playback_layout = QVBoxLayout()
        
        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.play_slices)
        btns.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.clicked.connect(self.pause_slices)
        self.pause_btn.setEnabled(False)
        btns.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.clicked.connect(self.stop_slices)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.stop_btn)
        playback_layout.addLayout(btns)
        
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 1000)
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.setSuffix(" ms")
        self.speed_spinbox.valueChanged.connect(lambda v: setattr(self, 'playback_speed', v))
        playback_layout.addWidget(QLabel("Speed:"))
        playback_layout.addWidget(self.speed_spinbox)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # Window/Level
        wl_group = QGroupBox("🎨 Window/Level")
        wl_layout = QVBoxLayout()
        
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(1, 2000)
        self.window_slider.setValue(400)
        self.window_slider.valueChanged.connect(self.update_window_level)
        wl_layout.addWidget(QLabel("Window:"))
        wl_layout.addWidget(self.window_slider)
        
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(-1000, 1000)
        self.level_slider.setValue(40)
        self.level_slider.valueChanged.connect(self.update_window_level)
        wl_layout.addWidget(QLabel("Level:"))
        wl_layout.addWidget(self.level_slider)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Brain", "Bone", "Lung", "Abdomen"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        wl_layout.addWidget(QLabel("Presets:"))
        wl_layout.addWidget(self.preset_combo)
        
        wl_group.setLayout(wl_layout)
        layout.addWidget(wl_group)
        
        # Reference lines
        ref_group = QGroupBox("➕ Reference Lines")
        ref_layout = QVBoxLayout()
        
        self.show_ref_checkbox = QCheckBox("Show Reference Lines")
        self.show_ref_checkbox.setChecked(True)
        self.show_ref_checkbox.stateChanged.connect(self.toggle_reference_lines)
        ref_layout.addWidget(self.show_ref_checkbox)
        
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        layout.addStretch()
        return panel
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        open_manager_action = QAction("Open DICOM Manager", self)
        open_manager_action.setShortcut("Ctrl+O")
        open_manager_action.triggered.connect(self.open_dicom_manager)
        file_menu.addAction(open_manager_action)
        
        open_series_action = QAction("Quick Load DICOM Series", self)
        open_series_action.triggered.connect(self.open_dicom_series)
        file_menu.addAction(open_series_action)
        
        export_action = QAction("Export ROI", self)
        export_action.triggered.connect(self.export_roi)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("View")
        reset_action = QAction("Reset All Views", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_action)
        
        ai_menu = menubar.addMenu("AI Tools")
        detect_action = QAction("Detect Organ ", self)
        detect_action.triggered.connect(self.detect_organ_gemini)
        ai_menu.addAction(detect_action)
    
    def create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage("Ready - Open DICOM Manager or Load DICOM Series to begin")
    
    # Playback methods
    def play_slices(self):
        """Start slice playback"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        self.is_playing = True
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.playback_timer.start(self.playback_speed)
        self.statusBar().showMessage("⏵ Playing slices...")

    def pause_slices(self):
        """Pause slice playback"""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("⏸ Playback paused")

    def stop_slices(self):
        """Stop slice playback"""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        # Reset to middle slice
        if self.roi_active:
            middle_slice = (self.roi_start + self.roi_end) // 2
        else:
            middle_slice = self.image_loader.get_num_slices('axial') // 2
        
        self.reference_position[0] = middle_slice
        self.axial_slider.setValue(middle_slice)
        self.update_axial_view()
        self.update_all_reference_lines()
        
        self.statusBar().showMessage("⏹ Playback stopped")

    def advance_slice(self):
        """Advance to next slice during playback"""
        if not self.is_playing or self.image_loader.image_array is None:
            return
        
        current_slice = self.reference_position[0]
        
        if self.roi_active:
            next_slice = current_slice + 1
            if next_slice > self.roi_end:
                next_slice = self.roi_start
        else:
            max_slices = self.image_loader.get_num_slices('axial')
            next_slice = (current_slice + 1) % max_slices
        
        self.reference_position[0] = next_slice
        self.axial_slider.setValue(next_slice)
        self.update_axial_view()
        self.update_all_reference_lines()
        
        # Update oblique view
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )
    
    def set_slice(self, orientation, value):
        """Set slice from slider"""
        if orientation == 'axial' and self.roi_active:
            value = max(self.roi_start, min(value, self.roi_end))
        
        if orientation == 'axial':
            self.reference_position[0] = value
            self.update_axial_view()
        elif orientation == 'sagittal':
            self.reference_position[2] = value
            self.update_sagittal_view()
        elif orientation == 'coronal':
            self.reference_position[1] = value
            self.update_coronal_view()
        
        self.update_all_reference_lines()
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )
    
    def set_view_zoom(self, orientation, zoom):
        """Set zoom for specific view"""
        viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
        if viewer:
            viewer.set_zoom(zoom)
    
    def reset_single_view(self, orientation):
        """Reset single view"""
        viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
        if viewer:
            viewer.zoom_factor = 1.0
            viewer.reset_pan()
            zoom_spin = self.findChild(QDoubleSpinBox, f"{orientation}_zoom")
            if zoom_spin:
                zoom_spin.setValue(1.0)
    
    def update_window_level(self):
        """Update window/level for all views"""
        window = self.window_slider.value()
        level = self.level_slider.value()
        
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.set_window_level(window, level)
        
        # Update fourth window based on mode
        if self.fourth_window_mode == "oblique":
            self.oblique_viewer.set_window_level(window, level)
        elif self.fourth_window_mode == "brain_surface":
            self.brain_surface_viewer.set_window_level(window, level)
    
    def apply_preset(self, preset):
        """Apply window/level preset"""
        presets = {
            "Brain": (80, 40),
            "Bone": (2000, 300),
            "Lung": (1500, -600),
            "Abdomen": (400, 40)
        }
        
        if preset in presets:
            w, l = presets[preset]
            self.window_slider.blockSignals(True)
            self.level_slider.blockSignals(True)
            self.window_slider.setValue(w)
            self.level_slider.setValue(l)
            self.window_slider.blockSignals(False)
            self.level_slider.blockSignals(False)
            self.update_window_level()
    
    def select_roi(self):
        """Open ROI selection dialog"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        max_slices = self.image_loader.get_num_slices('axial')
        dialog = ROIDialog(max_slices, self)
        
        if self.roi_active:
            dialog.start_spin.setValue(self.roi_start)
            dialog.end_spin.setValue(self.roi_end)
        
        if dialog.exec_() == QDialog.Accepted:
            self.roi_start, self.roi_end = dialog.get_roi()
            count = self.roi_end - self.roi_start + 1
            self.roi_status.setText(f"ROI: Slices {self.roi_start}-{self.roi_end} ({count} slices)")
            self.roi_status.setStyleSheet("color: #ffaa00;")
            self.statusBar().showMessage(f"ROI selected: {count} slices (not yet applied)")
    
    def apply_roi(self):
        """Apply ROI to limit view range"""
        if self.roi_start >= self.roi_end:
            QMessageBox.warning(self, "Invalid ROI", "Please select a valid ROI first.")
            return
        
        self.roi_active = True
        
        self.axial_slider.setMinimum(self.roi_start)
        self.axial_slider.setMaximum(self.roi_end)
        
        if self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            _, height, width = roi_volume.shape
            
            self.coronal_slider.setMaximum(height - 1)
            self.sagittal_slider.setMaximum(width - 1)
        
        if self.reference_position[0] < self.roi_start:
            self.reference_position[0] = self.roi_start
        elif self.reference_position[0] > self.roi_end:
            self.reference_position[0] = self.roi_end
        
        if self.image_loader.image_array is not None:
            if self.reference_position[1] >= height:
                self.reference_position[1] = height - 1
            if self.reference_position[2] >= width:
                self.reference_position[2] = width - 1
        
        self.axial_slider.setValue(self.reference_position[0])
        self.coronal_slider.setValue(self.reference_position[1])
        self.sagittal_slider.setValue(self.reference_position[2])
        
        count = self.roi_end - self.roi_start + 1
        self.roi_status.setText(f"ROI: Active ({self.roi_start}-{self.roi_end}, {count} slices)")
        self.roi_status.setStyleSheet("color: #44ff44;")
        
        self.display_all_views()
        self.oblique_viewer.set_roi(self.roi_start, self.roi_end, True)
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )
        
        QMessageBox.information(self, "ROI Applied", 
                               f"Navigation limited to ROI region:\n"
                               f"Axial slices: {self.roi_start}-{self.roi_end} ({count} slices)\n\n"
                               f"All views now show only the ROI volume.\n"
                               f"You can now detect the main organ using AI.")
    
    def clear_roi(self):
        """Clear ROI selection"""
        if self.image_loader.image_array is None:
            return
        
        self.roi_active = False
        self.roi_start = 0
        self.roi_end = self.image_loader.get_num_slices('axial') - 1
        
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(self.roi_end)
        
        sagittal_max = self.image_loader.get_num_slices('sagittal') - 1
        coronal_max = self.image_loader.get_num_slices('coronal') - 1
        self.sagittal_slider.setMaximum(sagittal_max)
        self.coronal_slider.setMaximum(coronal_max)
        
        self.roi_status.setText("ROI: Not Set")
        self.roi_status.setStyleSheet("color: #888;")
        
        self.detected_organ = "Unknown"
        self.organ_label.setText("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        self.organ_details.clear()
        
        axial_max = self.image_loader.get_num_slices('axial') - 1
        self.reference_position = [
            axial_max // 2,
            coronal_max // 2,
            sagittal_max // 2
        ]
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            width = self.image_loader.image_array.shape[2]
            height = self.image_loader.image_array.shape[1]
            axial_viewer.oblique_start = QPointF(width * 0.25, height * 0.5)
            axial_viewer.oblique_end = QPointF(width * 0.75, height * 0.5)
            axial_viewer.set_oblique_line_visible(True)
        
        self.oblique_viewer.set_volume(self.image_loader.image_array)
        
        self.display_all_views()
        self.setup_sliders()
    
    def setup_sliders(self):
        """Setup slider positions"""
        self.axial_slider.setValue(self.reference_position[0])
        self.sagittal_slider.setValue(self.reference_position[2])
        self.coronal_slider.setValue(self.reference_position[1])
    
    def display_all_views(self):
        """Display all views"""
        if self.image_loader.image_array is None:
            return
        
        self.update_axial_view()
        self.update_sagittal_view()
        self.update_coronal_view()
        self.update_all_reference_lines()
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )
    
    def update_axial_view(self):
        """Update axial view"""
        slice_idx = self.reference_position[0]
        
        slice_data = self.image_loader.get_slice('axial', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "axial_viewer")
        if viewer:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "axial_info")
        if info:
            if self.roi_active:
                total = self.roi_end - self.roi_start + 1
                relative_idx = slice_idx - self.roi_start
                info.setText(f"Slice: {relative_idx + 1}/{total} (Abs: {slice_idx + 1})")
            else:
                total = self.image_loader.get_num_slices('axial')
                info.setText(f"Slice: {slice_idx + 1}/{total}")
    
    def update_sagittal_view(self):
        """Update sagittal view"""
        slice_idx = self.reference_position[2]
        
        if self.roi_active and self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            if slice_idx < roi_volume.shape[2]:
                slice_data = roi_volume[:, :, slice_idx]
            else:
                slice_data = None
        else:
            slice_data = self.image_loader.get_slice('sagittal', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "sagittal_viewer")
        if viewer and slice_data is not None:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "sagittal_info")
        if info:
            if self.roi_active and self.image_loader.image_array is not None:
                total = self.image_loader.image_array.shape[2]
                info.setText(f"Slice: {slice_idx + 1}/{total} (ROI Mode)")
            else:
                total = self.image_loader.get_num_slices('sagittal')
                info.setText(f"Slice: {slice_idx + 1}/{total}")
    
    def update_coronal_view(self):
        """Update coronal view"""
        slice_idx = self.reference_position[1]
        
        if self.roi_active and self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            if slice_idx < roi_volume.shape[1]:
                slice_data = roi_volume[:, slice_idx, :]
            else:
                slice_data = None
        else:
            slice_data = self.image_loader.get_slice('coronal', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "coronal_viewer")
        if viewer and slice_data is not None:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "coronal_info")
        if info:
            if self.roi_active and self.image_loader.image_array is not None:
                total = self.image_loader.image_array.shape[1]
                info.setText(f"Slice: {slice_idx + 1}/{total} (ROI Mode)")
            else:
                total = self.image_loader.get_num_slices('coronal')
                info.setText(f"Slice: {slice_idx + 1}/{total}")
    
    def reset_view(self):
        """Reset all views to default"""
        if self.image_loader.image_array is None:
            return
        
        # Reset zoom for all views
        for orientation in ['axial', 'sagittal', 'coronal']:
            zoom_spin = self.findChild(QDoubleSpinBox, f"{orientation}_zoom")
            if zoom_spin:
                zoom_spin.setValue(1.0)
            
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.zoom_factor = 1.0
                viewer.reset_pan()
        
        # Reset fourth window zoom
        self.fourth_zoom_spin.setValue(1.0)
        
        # Reset to center slices
        if self.roi_active:
            axial_center = (self.roi_start + self.roi_end) // 2
        else:
            axial_center = self.image_loader.get_num_slices('axial') // 2
        
        coronal_center = self.image_loader.get_num_slices('coronal') // 2
        sagittal_center = self.image_loader.get_num_slices('sagittal') // 2
        
        self.reference_position = [axial_center, coronal_center, sagittal_center]
        
        self.axial_slider.setValue(axial_center)
        self.coronal_slider.setValue(coronal_center)
        self.sagittal_slider.setValue(sagittal_center)
        
        # Reset window/level
        self.window_slider.setValue(400)
        self.level_slider.setValue(40)
        
        self.display_all_views()
        self.statusBar().showMessage("✓ All views reset")

    def open_dicom_series(self):
        """Quick load DICOM series"""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Series Folder")
        if folder:
            self.load_dicom_series(folder)

    def load_image(self, path):
        """Load a single DICOM file"""
        success, message = self.image_loader.load_file(path)
        if success:
            self.statusBar().showMessage(f"✓ Loaded: {os.path.basename(path)}")
            self.initialize_views()
        else:
            QMessageBox.critical(self, "Load Error", message)

    def load_dicom_series(self, folder):
        """Load DICOM series from folder"""
        success, message = self.image_loader.load_dicom_series(folder)
        if success:
            self.statusBar().showMessage(f"✓ Loaded DICOM series: {os.path.basename(folder)}")
            self.initialize_views()
        else:
            QMessageBox.critical(self, "Load Error", message)

    def initialize_views(self):
        """Initialize all views after loading data"""
        if self.image_loader.image_array is None:
            return
        
        # Reset ROI
        self.roi_active = False
        self.roi_start = 0
        self.roi_end = self.image_loader.get_num_slices('axial') - 1
        
        # Setup sliders
        axial_max = self.image_loader.get_num_slices('axial') - 1
        sagittal_max = self.image_loader.get_num_slices('sagittal') - 1
        coronal_max = self.image_loader.get_num_slices('coronal') - 1
        
        self.axial_slider.setRange(0, axial_max)
        self.sagittal_slider.setRange(0, sagittal_max)
        self.coronal_slider.setRange(0, coronal_max)
        
        # Center position
        self.reference_position = [
            axial_max // 2,
            coronal_max // 2,
            sagittal_max // 2
        ]
        
        # Setup oblique line in axial view
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            width = self.image_loader.image_array.shape[2]
            height = self.image_loader.image_array.shape[1]
            axial_viewer.oblique_start = QPointF(width * 0.25, height * 0.5)
            axial_viewer.oblique_end = QPointF(width * 0.75, height * 0.5)
            axial_viewer.set_oblique_line_visible(True)
        
        # Set volume for oblique viewer
        self.oblique_viewer.set_volume(self.image_loader.image_array)
        
        # Display all views
        self.setup_sliders()
        self.display_all_views()
        
        # Reset organ detection
        self.detected_organ = "Unknown"
        self.organ_label.setText("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        self.organ_details.clear()

    def export_roi(self):
        """Export ROI volume"""
        if not self.roi_active or self.image_loader.image_array is None:
            QMessageBox.warning(self, "No ROI", "Please apply an ROI first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Volume", "",
            "NumPy Array (*.npy);;All Files (*.*)"
        )
        
        if file_path:
            try:
                roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
                np.save(file_path, roi_volume)
                QMessageBox.information(
                    self, "Export Success",
                    f"ROI volume exported successfully!\n\n"
                    f"File: {os.path.basename(file_path)}\n"
                    f"Shape: {roi_volume.shape}\n"
                    f"Slices: {self.roi_start}-{self.roi_end}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export ROI:\n{str(e)}")
    
    def detect_organ_gemini(self):
        """Detect organ using API"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load a DICOM image first.")
            return
        
        if not self.roi_active:
            QMessageBox.warning(self, "No ROI", "Please select and apply an ROI first.")
            return
        
        progress = QProgressDialog("Analyzing with AI...", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(30)
        
        self.gemini_thread = GeminiDetectionThread(
            self.image_loader.image_array,
            self.roi_start,
            self.roi_end
        )
        self.gemini_thread.detection_complete.connect(self.on_gemini_detection_complete)
        self.gemini_thread.error.connect(self.on_gemini_detection_error)
        self.gemini_thread.finished.connect(lambda: progress.setValue(100))
        self.gemini_thread.start()
        
        progress.setValue(50)
    
    def on_gemini_detection_complete(self, organ_name, full_response):
        """Handle detection completion"""
        self.detected_organ = organ_name
        self.gemini_response = full_response
        
        self.organ_label.setText(f"Organ: {organ_name}")
        self.organ_label.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 11pt;")
        
        self.organ_details.setText(full_response)
        
        self.statusBar().showMessage(f"✓ AI detected: {organ_name} (ROI: slices {self.roi_start}-{self.roi_end})")
        
        QMessageBox.information(self, " Detection Complete", 
                               f"Detected organ: {organ_name}\n\n"
                               f"ROI Range: Slices {self.roi_start}-{self.roi_end}")
    
    def on_gemini_detection_error(self, error_msg):
        """Handle  detection error"""
        self.organ_label.setText("Organ: Detection Failed")
        self.organ_label.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.organ_details.setText(f"Error: {error_msg}")
        
        QMessageBox.critical(self, "Detection Error", error_msg)
    
    def update_reference_from_view(self, orientation, x, y):
        """Update reference position from view click"""
        if self.image_loader.image_array is None:
            return
        
        if orientation == 'axial':
            self.reference_position[1] = y
            self.reference_position[2] = x
            
        elif orientation == 'sagittal':
            self.reference_position[0] = y
            self.reference_position[1] = x
            
        elif orientation == 'coronal':
            self.reference_position[0] = y
            self.reference_position[2] = x
        
        self.update_all_reference_lines()
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )
        
        self.statusBar().showMessage(
            f"Reference: Axial={self.reference_position[0]}, "
            f"Coronal={self.reference_position[1]}, "
            f"Sagittal={self.reference_position[2]}"
        )
    
    def update_all_reference_lines(self):
        """Update reference lines in all views"""
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.set_reference_position(self.reference_position)
    
    def toggle_reference_lines(self, state):
        """Toggle reference lines visibility"""
        show = state == Qt.Checked
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.show_reference_lines = show
                if viewer.current_slice is not None:
                    viewer.display_slice(viewer.current_slice)
    
    def navigate_slice(self, orientation, direction):
        """Navigate slices with mouse wheel"""
        if self.image_loader.image_array is None:
            return
        
        if orientation == 'axial':
            new_slice = self.reference_position[0] + direction
            
            if self.roi_active:
                new_slice = max(self.roi_start, min(new_slice, self.roi_end))
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
                new_slice = max(0, min(new_slice, max_slices - 1))
            
            self.reference_position[0] = new_slice
            self.axial_slider.blockSignals(True)
            self.axial_slider.setValue(new_slice)
            self.axial_slider.blockSignals(False)
            self.update_axial_view()
            
        elif orientation == 'sagittal':
            new_slice = self.reference_position[2] + direction
            
            if self.roi_active and self.image_loader.image_array is not None:
                roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
                max_slices = roi_volume.shape[2]
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
            
            new_slice = max(0, min(new_slice, max_slices - 1))
            self.reference_position[2] = new_slice
            self.sagittal_slider.blockSignals(True)
            self.sagittal_slider.setValue(new_slice)
            self.sagittal_slider.blockSignals(False)
            self.update_sagittal_view()
            
        elif orientation == 'coronal':
            new_slice = self.reference_position[1] + direction
            
            if self.roi_active and self.image_loader.image_array is not None:
                roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
                max_slices = roi_volume.shape[1]
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
            
            new_slice = max(0, min(new_slice, max_slices - 1))
            self.reference_position[1] = new_slice
            self.coronal_slider.blockSignals(True)
            self.coronal_slider.setValue(new_slice)
            self.coronal_slider.blockSignals(False)
            self.update_coronal_view()
        
        self.update_all_reference_lines()
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            self.oblique_viewer.update_oblique(
                axial_viewer.oblique_start,
                axial_viewer.oblique_end,
                self.reference_position[0]
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:disabled {
            background-color: #252525;
            color: #666;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999;
            height: 8px;
            background: #3a3a3a;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #5a5a5a;
            border: 1px solid #777;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 3px;
            border-radius: 3px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #aaa;
            margin-right: 5px;
        }
        QTextEdit {
            background-color: #1a1a1a;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
        }
        QListWidget {
            background-color: #1a1a1a;
            border: 1px solid #555;
            border-radius: 3px;
        }
        QListWidget::item:selected {
            background-color: #4a9eff;
        }
        QStatusBar {
            background-color: #1a1a1a;
            color: #aaa;
        }
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #3a3a3a;
        }
        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555;
        }
        QMenu::item:selected {
            background-color: #3a3a3a;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #3a3a3a;
        }
        QCheckBox::indicator:checked {
            background-color: #4a9eff;
            border-color: #4a9eff;
        }
        QDialog {
            background-color: #2b2b2b;
        }
        QProgressDialog {
            background-color: #2b2b2b;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())