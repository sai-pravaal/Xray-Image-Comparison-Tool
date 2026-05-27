# X-ray Image Comparison Tool

A sophisticated medical imaging tool designed to compare patient X-rays against reference images for diagnostic assistance. It utilizes advanced computer vision techniques to identify structural differences, density variations, and anatomical anomalies.

## 🌟 Key Features

- **Automated Reference Matching**: Intelligently searches a local database for the most appropriate reference image based on:
  - Body part (Chest, Femur, Hand, etc.)
  - Lateral side (Left, Right, Midline)
  - Patient gender and age bracket
- **Precision Imaging Core**:
  - **Rigid Registration**: Automatically aligns the test image to the reference using ORB feature matching and RANSAC affine estimation.
  - **Bone Masking & Centerlines**: Extracts anatomical skeletons to analyze width profiles and curvature.
  - **Signature Analysis**: Generates shape and radiodensity vectors for quantitative comparison.
- **Dynamic Visualization**:
  - **Contours**: Highlights structural differences in red overlays.
  - **Heatmaps**: Displays intensity-based differences using JET color maps.
  - **Signature Graphs**: Visualizes z-normalized width, curvature, and direction changes.
  - **Interactive Tools**: 100x Zoom cursor, Checkerboard view, and Flicker mode for rapid visual comparisons.
- **Reporting**: Generates comprehensive 6-panel PDF reports including test/reference previews, difference maps, and statistical scoring.
- **Format Support**: Handles standard image formats (PNG, JPG, TIFF) and medical DICOM files with VOI LUT support.

## 🛠️ Technology Stack

- **UI Framework**: Tkinter
- **Image Processing**: OpenCV, NumPy, Scikit-Image, Pillow
- **Data Management**: SQLite3
- **Medical Imaging**: PyDICOM
- **Visualization & Reporting**: Matplotlib, PyPDF

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sai-pravaal/Xray-Image-Comparison-Tool.git
   cd Xray-Image-Comparison-Tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Initialization

The tool automatically initializes its SQLite database (`xray.db`) and ingests images from the `./data` directory on startup. Ensure your data is organized in folders matching the expected hierarchy:
`data/[BodyPart]/[Side]/[Gender]/[AgeBand]/[Images]`

### Running the Application

Launch the main interface:
```bash
python frontend.py
```

## 📖 Usage Guide

1. **Upload X-ray**: Click the "Upload" button to select a patient image.
2. **Set Parameters**: Select the body part, side, gender, and age to find the correct reference.
3. **Compare**: Click "Compare" to run the registration and analysis pipeline.
4. **Analysis Views**: Use the "View" dropdown to switch between Contours, Heatmaps, and Signature graphs.
5. **Adjust Threshold**: Use the slider to fine-tune the sensitivity of the difference detection.
6. **Export**: Save your results as a high-quality PNG or a detailed PDF report.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed by Sai Pravaal (24BBS0115),
Prakyath Tejsundar (24BBS0079),
Rethish J Kanth (24BBS107) *
