# MultiPlanar_Reconstruction_with_AI_Support

# 🧠 3D Slicer-Style Medical Image Viewer

A Python-based medical image viewer inspired by 3D Slicer, featuring multi-planar reconstruction (MPR), AI-powered organ detection, oblique plane viewing, brain surface visualization, and DICOM orientation classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Features

- **Multi-Planar Reconstruction (MPR)** — Simultaneous Axial, Sagittal, and Coronal views with synchronized reference lines
- **Oblique Plane Viewer** — Define a custom cut line in the axial view and extract an oblique cross-section with rotation support
- **Brain Surface Viewer** — Dedicated fourth window for brain surface DICOM visualization (switchable via dropdown)
- **AI Organ Detection** — Uses Google Gemini API to automatically identify the primary organ in a selected ROI
- **DICOM Orientation Classifier** — Deep learning model (SAM/ViT-based) that classifies DICOM slices as Axial, Coronal, or Sagittal with confidence scores
- **ROI Slice Selection** — Select a sub-range of axial slices, apply as active ROI, and export the volume
- **Window/Level Controls** — Manual sliders plus presets (Brain, Bone, Lung, Abdomen)
- **Slice Playback** — Animated playback through slices with adjustable speed
- **Zoom & Pan** — Per-view zoom spinbox and middle-mouse-button panning
- **Dark Theme UI** — Full dark Fusion-style theme throughout

---

## 🖥️ Screenshots

> _Add screenshots of the viewer here_

---

## 🛠️ Requirements

### System
- Python 3.8+
- Windows / Linux / macOS

### Python Dependencies

```bash
pip install PyQt5 numpy Pillow google-generativeai scipy scikit-image pydicom torch torchvision
```

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework |
| `numpy` | Array operations |
| `Pillow` | Image processing |
| `google-generativeai` | Gemini AI organ detection |
| `scipy` | Oblique plane rotation |
| `scikit-image` | Morphological operations |
| `pydicom` | DICOM file reading |
| `torch` / `torchvision` | Orientation classifier model |

---

## ⚙️ Setup & Configuration

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths in the source file

At the top of the main script, update the following constants:

```python
# Your Gemini API key (for organ detection)
API_KEY = "your-gemini-api-key-here"

# Path to brain surface DICOM folder
BRAIN_SURFACE_DICOM_PATH = r"path/to/your/dicom_outlines"
```

### 4. Orientation Classifier Model (optional)

If you want to use the DICOM orientation classification feature, you need:

- A fine-tuned model checkpoint at:
  `D:\task2\orientation code\mri_orientation_finetuned.pth`
- The MRI Foundation codebase at:
  `D:\task2\orientation code\mri_foundation-master`

Update these paths inside `DicomFileManagerWindow.init_classifier()` to match your environment.

---

## 🚀 Running the Application

```bash
python main_viewer.py
```

---

## 📖 Usage Guide

### Loading Images
- Click **"Open DICOM Manager"** to open the classification + file browser window
- Or click **"Quick Load DICOM Series"** to directly load a folder of DICOM files

### Navigating Views
- **Mouse wheel** — scroll through slices in any view
- **Click in a view** — move the crosshair reference lines in all views
- **Middle mouse button drag** — pan the image within a view
- **Zoom spinbox** — zoom a specific view independently

### Oblique Plane
1. Load a DICOM series
2. In the **Axial view**, drag the magenta line handles to define your cut plane
3. The **Oblique View** (fourth window) updates in real time
4. Use the **Rot X** spinbox to rotate the extracted plane

### Brain Surface View
1. Select **"Brain Surface"** from the dropdown in the fourth window
2. Click **"🧠 Load Brain Surface"** to load from the configured DICOM path
3. Scroll with the mouse wheel to navigate slices

### ROI & Organ Detection
1. Click **"Select ROI Slices"** and choose a start/end axial slice
2. Click **"Apply ROI"** — navigation is then locked to that range
3. Click **"🔍 Detect Main Organ"** — Gemini AI will analyze the ROI and return the organ name
4. Click **"Export ROI Volume"** to save the ROI as a `.npy` file

### Orientation Classification
1. Open the DICOM Manager window
2. Browse for a `.dcm` file
3. Click **"🎯 Classify Orientation"** — the model will output Axial / Coronal / Sagittal with confidence scores and a probability bar chart

---

## 🗂️ Project Structure

```
├── main_viewer.py          # Main application entry point
├── image_loader.py         # MedicalImageLoader class (DICOM/NIfTI loading)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🤖 AI Components

### Gemini Organ Detection
Uses `gemini-2.0-flash-exp` via the Google Generative AI API. A middle slice from the selected ROI is sent to the model, which identifies the primary organ visible.

### Orientation Classifier
A SAM (Segment Anything Model) ViT-B backbone fine-tuned for 3-class classification:
- **0 → Axial**
- **1 → Coronal**
- **2 → Sagittal**

Falls back to a ResNet-50 head if the SAM foundation model is unavailable.

---

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Paste it into the `API_KEY` variable in the script

---

## ⚠️ Known Limitations

- The orientation classifier requires the MRI Foundation codebase and a fine-tuned checkpoint — it will show an error if these are not present
- Brain Surface loading requires a valid DICOM directory at the configured path
- Organ detection requires a valid Gemini API key and internet connection

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [3D Slicer](https://www.slicer.org/) — inspiration for the multi-planar viewer layout
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) — backbone for orientation classifier
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI organ detection
- [pydicom](https://pydicom.github.io/) — DICOM file handling
