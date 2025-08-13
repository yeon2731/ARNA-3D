# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Settings

- 모든 설명과 응답은 한글로 제공
- 코드 주석은 한글로 작성

## Project Overview

ARNA-3D is a medical image processing pipeline that converts NIfTI segmentation files into high-quality 3D GLB models. It processes kidney segmentation data through a 5-stage pipeline: NIfTI preprocessing, initial GLB generation, first-stage smoothing, Poisson reconstruction, and second-stage smoothing.

## Common Commands

### Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main pipeline with default settings
python main.py

# The pipeline processes a single case specified in main.py line 84
# Modify case_path variable to process different cases
```

### Testing

```bash
# Run fast test
python test/fast/main2.py

# Run label conversion test
python test/labelconvert.py
```

## Architecture and Pipeline

The codebase follows a modular architecture with three main processing modules:

### Core Processing Flow

1. **Input**: NIfTI segmentation files (`segment_A.nii.gz`) from `data/case_*/mask/`
2. **processNii.py**: Medical image preprocessing with morphological operations
3. **combineGLB.py**: Marching cubes mesh extraction and multi-label handling
4. **processMesh.py**: Two-stage smoothing pipeline with configurable parameters
5. **Output**: Final GLB files saved to `data/case_*/3d/obj_A.glb`

### Key Components

**threeDRecon Module Structure:**

- `processNii.py`: Handles NIfTI preprocessing, connected component analysis, noise reduction
- `combineGLB.py`: Converts label arrays to meshes using marching cubes, includes LABELS mapping for anatomical structures
- `processMesh.py`: Mesh smoothing operations (Taubin/Laplacian) and Poisson reconstruction
- `config/`: JSON configuration files for two-stage smoothing parameters

**Configuration System:**

- `parts_config1.json`: First-stage smoothing parameters per anatomical structure
- `parts_config2.json`: Second-stage smoothing parameters
- Each structure has configurable smoothing functions (taubin/laplacian) and dilation operations

### Anatomical Structure Labels

The pipeline recognizes these anatomical structures with specific IDs:

- Tumor (1), Kidney (1,2), Artery (3), Vein (4), Ureter (5), Fat (6), Renal_a (7), Renal_v (8)

## Key Implementation Details

### Main Processing Function

- `main(case_path)` in main.py orchestrates the entire pipeline
- Uses regex parsing to extract case IDs and phases from file paths
- Applies two-stage smoothing with JSON-configured parameters

### Mesh Processing Pipeline

- First smoothing stage applies organ-specific parameters from parts_config1.json
- Poisson reconstruction improves mesh topology between smoothing stages
- Second smoothing stage uses parts_config2.json for final quality enhancement
- Pipeline supports debug mode for intermediate file inspection

### Data Structure

```
data/
├── case_XXXX/
│   ├── mask/segment_A.nii.gz    # Input segmentation
│   └── 3d/obj_A.glb            # Output 3D model
└── inference/                   # Additional datasets
```

## Development Notes

- The pipeline processes one case at a time; modify `case_path` in main.py for different cases
- Smoothing parameters are highly specialized for medical imaging and should be adjusted carefully
- The system expects specific file naming conventions (case_XXXX, segment_A pattern)
- Processing time is approximately 40 seconds per subject on modern hardware
- All intermediate processing uses spacing information preserved from original NIfTI files
