# 신장, 종양 기반 혈관 fitting

## 가상환경

python=3.11 (사용함)
SimpleITK>=2.2.0
numpy>=1.21.0

## function 정리

-   split_kidney_label

    label 2로 구성된 kidney segmentation에서 connected component로 좌우 kidney 분리 (label 10: left, label 20: right)

-   extract_labels

    지정된 label만 추출하여 새로운 mask 이미지 반환

-   apply_transform_to_labels

    moving_image에 transform을 적용하여 reference_image 공간으로 정렬

-   combine_images

    두 label mask를 병합하되 image2의 label이 우선 (덮어쓰기)

-   resample_to_fixed_spacing

    moving image를 fixed image의 spacing으로 resample.
    반환값: (moving_resampled, fixed_img), 기준은 fixed image의 spacing

-   get_label_centroid

    지정된 label의 무게중심 좌표 추출

-   create_centroid_translation_transform

    중점 기준 평행이동 transform 생성

-   create_xy_rotation_transform

    XY 평면 회전 transform 생성

-   create_scaling_transform

    스케일링 transform 생성

-   create_z_shear_transform

    Z축 전단 transform 생성

-   z_shear_xy_rigid_affine

    단계별 transform을 순차적으로 적용

-   _align_and_combine_kidney_images_

    신장 영상을 정렬하고 결합하는 function
    가장 중요!
    변경 가능한 변수

    1. fixed_path : segmentation 결과 파일(더 많은 label이 있어도 됨)
    2. moving_path : template인 정맥과 동맥 파일
    3. output_dir : 합쳐진 결과를 저장할 디렉토리 경로
    4. fixed_kidney_label : fixed_path 파일에서 kidney label, default는 2
    5. moving_av_labels : moving_path 파일에서 동맥과 정맥의 label, default는 [3, 4]
    6. fixed_base_labels : fixed_path 파일에서 최종 결과 정합에 사용할 kidney, tumor label

## 함수 상세 설명

### 핵심 함수들

#### `align_and_combine_kidney_images(fixed_path, moving_path, output_dir, ...)`

메인 함수로, 전체 정렬 및 결합 과정을 수행합니다.

**매개변수:**

-   `fixed_path`: 환자 영상 경로
-   `moving_path`: 템플릿 영상 경로
-   `output_dir`: 결과 저장 디렉토리
-   `fixed_kidney_label`: 신장 라벨 (기본값: 2)
-   `moving_av_labels`: 템플릿에서 추출할 라벨들 (기본값: [3, 4])
-   `fixed_base_labels`: 환자에서 유지할 기본 라벨들 (기본값: [1, 2])

#### `z_shear_xy_rigid_affine(moving_kidneys, fixed_kidneys)`

단계별 변환을 순차적으로 적용하여 최종 변환 행렬을 생성합니다.

### 유틸리티 함수들

-   `extract_labels(image, keep_labels)`: 지정된 라벨만 추출
-   `apply_transform_to_labels(moving_image, reference_image, transform)`: 변환 적용
-   `get_label_centroid(image, label)`: 라벨의 중심점 계산
-   `combine_images(image1, image2)`: 두 영상 결합

## input & output

-   input : fixed .nii 파일, moving .nii 파일, 둘 다 mask 파일만 있으면 됨

## 사용 방법

### 기본 사용법

```python
from main2 import align_and_combine_kidney_images

# 영상 경로 설정
fixed_path = "환자_영상.nii.gz"
moving_path = "템플릿_영상.nii.gz"
output_dir = "결과_저장_폴더"

# 정렬 및 결합 실행
final = align_and_combine_kidney_images(
    fixed_path=fixed_path,
    moving_path=moving_path,
    output_dir=output_dir
)
```

### 스크립트 직접 실행

```bash
python main2.py
```

## 출력 파일

결과 디렉토리에 다음 파일들이 생성됩니다:

1. **`transformed_av.nii.gz`**: 변환된 템플릿 영상
2. **`transformed_kidneys.nii.gz`**: 변환된 신장 영상
3. **`fixed_base.nii.gz`**: 환자 기본 라벨 영상
4. **`combined_result.nii.gz`**: 최종 결합된 영상
