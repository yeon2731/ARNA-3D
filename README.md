# ARNA 3D Medical Image Processing Pipeline

A comprehensive Python pipeline for converting medical segmentation data (NIfTI format) to high-quality 3D models (GLB format) with advanced smoothing and reconstruction techniques.

## Overview

This project provides a complete workflow for processing medical imaging data, specifically designed for kidney and related organ segmentation. It converts NIfTI segmentation files into smooth, production-ready 3D models suitable for visualization, 3D printing, or surgical planning.

## Features

- **Medical Image Processing**: Robust preprocessing of NIfTI segmentation data
- **Multi-organ Support**: Handles tumor, kidney, artery, vein, ureter, fat, and renal vessels
- **Advanced Smoothing**: Two-stage smoothing pipeline with configurable parameters
- **Poisson Reconstruction**: High-quality mesh reconstruction for improved geometry
- **Batch Processing**: Efficient processing of multiple subjects
- **Debug Mode**: Comprehensive intermediate file saving for pipeline inspection
- **Configurable Pipeline**: JSON-based configuration for different processing stages

## Architecture

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

### Setup

1. Prepare your data structure:

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

## Usage

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

## Label for Anatomical Structures

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

## Pipeline Configuration

### Smoothing Functions

- **Taubin**: Volume-preserving smoothing
- **Laplacian**: Traditional Laplacian smoothing

### Dilation Functions

- **Default**: Normal-based mesh offset
- **Custom**: User-defined dilation methods

## Output Files

For each processed subject, the pipeline generates:

- `{subject_id}_preprocessed.nii.gz`: Preprocessed segmentation
- `{subject_id}_combined.glb`: Initial mesh combination
- `{subject_id}_1st_smoothed.glb`: First-stage smoothed mesh
- `{subject_id}_poisson_recon.glb`: Poisson reconstructed mesh
- `{subject_id}_2nd_smoothed.glb`: Final production mesh

## Debug Mode

Enable debug mode for comprehensive pipeline inspection:

```python
debug = True  # in main.py
```

This saves intermediate results and process time at each stage.

## Performance

Typical processing times (depending on hardware and data resolution):

- Preprocessing: seconds
- Initial GLB generation: seconds
- First smoothing: seconds
- Poisson reconstruction: seconds
- Second smoothing: seconds

Total: ~40 seconds per subject (Ryzen 5 7600 / 4070 TI Super / 32GB RAM / 1mm Image)

## Libraries used

- [SimpleITK](https://simpleitk.org/) for medical image processing
- [Trimesh](https://trimsh.org/) for 3D mesh processing
- [PyVista](https://pyvista.org/) for advanced mesh operations
- [Open3D](http://www.open3d.org/) for geometric processing
