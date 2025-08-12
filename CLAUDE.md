# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Settings

- 모든 설명과 응답은 한글로 제공
- 코드 주석은 한글로 작성

## Project Overview

ARNA-3D is a medical image processing pipeline that converts NIfTI segmentation files into high-quality 3D GLB models. The pipeline processes medical imaging data (specifically kidney and related organ segmentation) through a five-stage process: NIfTI preprocessing, initial GLB generation, first-stage smoothing, Poisson reconstruction, and second-stage smoothing.

## Core Commands

### Running the Pipeline

```bash
python main.py
```

This is the primary entry point that processes medical segmentation data from NIfTI files to 3D GLB models.

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Installs all required packages including SimpleITK, trimesh, PyVista, Open3D, and scientific computing libraries.

### Testing

Basic testing functionality is available in the `test/` directory:

```bash
python test/main2.py          # Alternative test main
python test/labelconvert.py   # Label conversion utilities
```

## Architecture Overview

### Main Pipeline (main.py)

- **Entry Point**: `main(case_path)` function processes a single case
- **Input**: Path to segmentation mask (e.g., `./data/case_0000/mask/segment_A.nii.gz`)
- **Output**: 3D GLB model saved to `./data/case_0000/3d/obj_A.glb`
- **Parallel Processing**: Utilizes multiprocessing for mesh smoothing operations with configurable worker count

### Core Processing Modules (threeDRecon/)

1. **processNii.py**: Medical image preprocessing

   - Noise reduction and morphological operations
   - Connected component analysis with `get_largest_component()`
   - Distance transform-based processing with `get_max_inscribed_circle()`

2. **combineGLB.py**: 3D mesh generation

   - Marching cubes mesh extraction via `mask_to_mesh()`
   - Multi-label anatomical structure handling using `LABELS` mapping
   - Mesh optimization and normal correction with `mask_to_mesh_fixnormal()`

3. **processMesh.py**: Advanced mesh processing
   - Two-stage smoothing pipeline (Taubin and Laplacian)
   - Mesh dilation and offset operations via `default_dilation()`
   - Poisson surface reconstruction with `process_poisson()`

### Configuration System

- **parts_config1.json**: First-stage smoothing parameters per anatomical structure
- **parts_config2.json**: Second-stage smoothing parameters
- Each structure has configurable `smoothing_func`, `smoothing_kwargs`, `dilation_func`, and `dilation_kwargs`

### Data Structure

Expected input data organization:

```
data/
├── case_XXXX/
│   ├── mask/
│   │   └── segment_[A|P].nii.gz    # Input segmentation
│   └── 3d/
│       └── obj_[A|P].glb           # Output 3D model
```

### Anatomical Label Mapping

- Tumor: 1
- Kidney: [1, 2] (includes tumor)
- Artery: 3
- Vein: 4
- Ureter: 5
- Fat: 6
- Renal artery: 7
- Renal vein: 8

## Development Notes

### Pipeline Stages

1. **Preprocessing**: NIfTI data cleanup and enhancement
2. **Initial GLB**: Marching cubes mesh generation from segmentation
3. **First Smoothing**: Taubin/Laplacian smoothing with dilation
4. **Poisson Reconstruction**: Surface reconstruction for improved topology
5. **Second Smoothing**: Final quality enhancement

### Multiprocessing Architecture

- Utilizes `scene_to_dict()` and `dict_to_scene()` for serializable mesh data
- `parallel_mesh_smoothing()` processes multiple anatomical structures concurrently
- Configurable worker count (default: 4)

### Key Functions

- `parse_info(case_path)`: Extracts case ID and phase from file paths
- `main(case_path)`: Complete pipeline execution
- `process_single_mesh()`: Worker function for parallel processing

### Configuration Modification

To adjust processing parameters, modify the JSON config files:

- Smoothing parameters: `n_iter`, `pass_band`, `feature_angle`
- Dilation parameters: `offset` values
- Per-structure customization available

The pipeline typically processes one subject in ~40 seconds on modern hardware (Ryzen 5 7600 / RTX 4070 TI Super / 32GB RAM) with 1mm resolution images.
