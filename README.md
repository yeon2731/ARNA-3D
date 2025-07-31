# ARNA 3D Medical Image Processing Pipeline

A comprehensive Python pipeline for converting medical segmentation data (NIfTI format) to high-quality 3D models (GLB format) with advanced smoothing and reconstruction techniques.

## 🏥 Overview

This project provides a complete workflow for processing medical imaging data, specifically designed for kidney and related organ segmentation. It converts NIfTI segmentation files into smooth, production-ready 3D models suitable for visualization, 3D printing, or surgical planning.

## ✨ Features

- **Medical Image Processing**: Robust preprocessing of NIfTI segmentation data
- **Multi-organ Support**: Handles tumor, kidney, artery, vein, ureter, fat, and renal vessels
- **Advanced Smoothing**: Two-stage smoothing pipeline with configurable parameters
- **Poisson Reconstruction**: High-quality mesh reconstruction for improved geometry
- **Batch Processing**: Efficient processing of multiple subjects
- **Debug Mode**: Comprehensive intermediate file saving for pipeline inspection
- **Configurable Pipeline**: JSON-based configuration for different processing stages

## 🏗️ Architecture

The pipeline consists of five main stages:

1. **NIfTI Preprocessing** (`processNii.py`)

   - Noise reduction and morphological operations
   - Connected component analysis
   - Image smoothing and enhancement

2. **Initial GLB Generation** (`combineGLB.py`)

   - Marching cubes mesh extraction
   - Multi-label segmentation handling
   - Mesh optimization and normal fixing

3. **First-stage Smoothing** (`processMesh.py`)

   - Taubin and Laplacian smoothing
   - Configurable smoothing parameters per organ
   - Mesh dilation and offset operations

4. **Poisson Reconstruction**

   - Surface reconstruction for improved topology
   - Noise reduction and hole filling

5. **Second-stage Smoothing**
   - Final quality enhancement
   - Production-ready mesh generation

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8+
pip install SimpleITK
pip install trimesh
pip install pyvista
pip install open3d
pip install scikit-image
pip install scipy
pip install numpy
pip install opencv-python
```

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/arna-3d-smoothing.git
cd arna-3d-smoothing
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your data structure:

```
dataset/
├── nii/                    # Input NIfTI segmentation files
│   ├── S000_segmentation.nii.gz
│   ├── S001_segmentation.nii.gz
│   └── ...
└── output/                 # Generated output files
    ├── caseS000/
    ├── caseS001/
    └── ...
```

## 🚀 Usage

### Basic Usage

```python
python main.py
```

### Configuration

The pipeline uses JSON configuration files for different processing stages:

- `threeDRecon/config/parts_config1.json`: First-stage smoothing parameters
- `threeDRecon/config/parts_config2.json`: Second-stage smoothing parameters

Example configuration:

```json
{
  "name": "Tumor",
  "smoothing_func": "taubin",
  "smoothing_kwargs": {
    "n_iter": 100,
    "pass_band": 0.001,
    "feature_angle": 200.0,
    "boundary_smoothing": true
  },
  "dilation_func": "default",
  "dilation_kwargs": {
    "offset": -0.5
  }
}
```

### Customization

Modify the subject ID and paths in `main.py`:

```python
subject_id = "S000"  # Change to your subject ID
base_folder = "path/to/your/project"
```

## 📊 Supported Anatomical Structures

| Label   | Structure                | ID   |
| ------- | ------------------------ | ---- |
| Tumor   | Primary tumor            | 1    |
| Kidney  | Kidney (including tumor) | 1, 2 |
| Artery  | Arterial system          | 3    |
| Vein    | Venous system            | 4    |
| Ureter  | Ureter                   | 5    |
| Fat     | Surrounding fat          | 6    |
| Renal_a | Renal artery             | 7    |
| Renal_v | Renal vein               | 8    |

## 🔧 Pipeline Configuration

### Smoothing Functions

- **Taubin**: Volume-preserving smoothing
- **Laplacian**: Traditional Laplacian smoothing

### Dilation Functions

- **Default**: Normal-based mesh offset
- **Custom**: User-defined dilation methods

## 📁 Output Files

For each processed subject, the pipeline generates:

- `{subject_id}_preprocessed.nii.gz`: Preprocessed segmentation
- `{subject_id}_combined.glb`: Initial mesh combination
- `{subject_id}_1st_smoothed.glb`: First-stage smoothed mesh
- `{subject_id}_poisson_recon.glb`: Poisson reconstructed mesh
- `{subject_id}_2nd_smoothed.glb`: Final production mesh

## 🐛 Debug Mode

Enable debug mode for comprehensive pipeline inspection:

```python
debug = True  # in main.py
```

This saves intermediate results at each processing stage for quality control and troubleshooting.

## 📈 Performance

Typical processing times (depending on hardware):

- Preprocessing: ~10-30 seconds
- Initial GLB generation: ~30-60 seconds
- First smoothing: ~60-120 seconds
- Poisson reconstruction: ~30-90 seconds
- Second smoothing: ~60-120 seconds

Total: ~3-6 minutes per subject

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [SimpleITK](https://simpleitk.org/) for medical image processing
- [Trimesh](https://trimsh.org/) for 3D mesh processing
- [PyVista](https://pyvista.org/) for advanced mesh operations
- [Open3D](http://www.open3d.org/) for geometric processing

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the development team.

---

**Note**: This software is intended for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
