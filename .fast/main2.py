import SimpleITK as sitk
import numpy as np
import os

def split_kidney_labels(image, label=2):
    """
    label 2로 구성된 kidney segmentation에서 connected component로
    좌우 kidney 분리 (label 10: left, label 20: right)
    """
    array = sitk.GetArrayFromImage(image)
    # 디버깅: 라벨 값 확인
    unique_labels = np.unique(array)
    print(f"이미지의 고유 라벨 값들: {unique_labels}")
    print(f"라벨 {label}의 픽셀 수: {np.sum(array == label)}")

    binary_array = (array == label).astype(np.uint8)
    binary_image = sitk.GetImageFromArray(binary_array)
    binary_image.CopyInformation(image)

    components = sitk.ConnectedComponent(binary_image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)

    labels = list(stats.GetLabels())
    print(f"연결된 컴포넌트 라벨들: {labels}")
    print(f"각 컴포넌트의 크기: {[stats.GetPhysicalSize(label) for label in labels]}")
    
    if len(labels) < 2:
        raise ValueError(f"신장 라벨 {label}에서 2개의 연결된 컴포넌트를 찾을 수 없습니다. 발견된 컴포넌트 수: {len(labels)}")
    
    # 1. 크기가 가장 큰 두 개의 컴포넌트 선택
    sizes = [stats.GetPhysicalSize(label) for label in labels]
    sorted_indices = np.argsort(sizes)[-2:]  # 크기가 가장 큰 2개의 인덱스
    selected_labels = [labels[i] for i in sorted_indices]
    
    # 2. 선택된 두 컴포넌트의 중심점 구하기
    centroids = [stats.GetCentroid(label) for label in selected_labels]
    
    # 3. x좌표 기준으로 left/right 결정
    if centroids[0][0] < centroids[1][0]:
        left_label, right_label = selected_labels[0], selected_labels[1]
    else:
        left_label, right_label = selected_labels[1], selected_labels[0]

    comp_array = sitk.GetArrayFromImage(components)
    split_array = np.zeros_like(comp_array)
    split_array[comp_array == left_label] = 10   # left kidney
    split_array[comp_array == right_label] = 20  # right kidney

    split_image = sitk.GetImageFromArray(split_array)
    split_image.CopyInformation(image)
    return split_image

def extract_labels(image, keep_labels):
    """
    지정된 label만 추출하여 새로운 mask 이미지 반환
    """

    array = sitk.GetArrayFromImage(image)
    filtered = np.zeros_like(array)
    for label in keep_labels:
        filtered[array == label] = label

    out_image = sitk.GetImageFromArray(filtered)
    out_image.CopyInformation(image)
    return out_image

def apply_transform_to_labels(moving_image, reference_image, transform):
    """
    moving_image에 transform을 적용하여 reference_image 공간으로 정렬
    """
    return sitk.Resample(
        moving_image,
        reference_image,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        moving_image.GetPixelID()
    )

def combine_images(image1, image2):
    """
    두 label mask를 병합하되 image2의 label이 우선 (덮어쓰기)
    """

    arr1 = sitk.GetArrayFromImage(image1)
    arr2 = sitk.GetArrayFromImage(image2)
    combined = np.where(arr2 > 0, arr2, arr1)

    out = sitk.GetImageFromArray(combined)
    out.CopyInformation(image1)
    return out

def resample_to_fixed_spacing(moving_img, fixed_img):
    """
    moving image를 fixed image의 spacing으로 resample.
    반환값: (moving_resampled, fixed_img), 기준은 fixed image의 spacing
    """
    # spacing 정보 출력
    moving_spacing = moving_img.GetSpacing()
    fixed_spacing = fixed_img.GetSpacing()

    print(f"\n=== Fixed spacing으로 맞추기 상세 정보 ===")
    print(f"Moving spacing: {moving_spacing}")
    print(f"Fixed spacing: {fixed_spacing}")

    # 리샘플링 전 라벨 값 확인
    moving_array = sitk.GetArrayFromImage(moving_img)
    fixed_array = sitk.GetArrayFromImage(fixed_img)
    print(f"리샘플링 전 Moving 이미지 라벨: {np.unique(moving_array)}")
    print(f"리샘플링 전 Fixed 이미지 라벨: {np.unique(fixed_array)}")
    print(f"Moving 이미지 픽셀 타입: {moving_img.GetPixelIDTypeAsString()}")
    print(f"Fixed 이미지 픽셀 타입: {fixed_img.GetPixelIDTypeAsString()}")

    # 이미지를 UInt8로 변환
    moving_img = sitk.Cast(moving_img, sitk.sitkUInt8)
    fixed_img = sitk.Cast(fixed_img, sitk.sitkUInt8)

    # 리샘플링 전에 이미지 크기 출력
    print(f"Moving 이미지 크기: {moving_img.GetSize()}")
    print(f"Fixed 이미지 크기: {fixed_img.GetSize()}")
    
    # moving image를 fixed image의 spacing으로 리샘플링
    moving_resampled = sitk.Resample(
        moving_img,
        fixed_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUInt8
    )
    
    # 리샘플링 후 이미지 크기 출력
    print(f"리샘플링 후 Moving 이미지 크기: {moving_resampled.GetSize()}")
    
    # 라벨 값 확인
    moving_array = sitk.GetArrayFromImage(moving_resampled)
    print(f"리샘플링 후 Moving 이미지 라벨: {np.unique(moving_array)}")
    
    # 라벨 값이 모두 0이면 원본 이미지 반환
    if len(np.unique(moving_array)) <= 1:
        print("경고: 리샘플링 후 라벨이 모두 0이 되었습니다. 원본 이미지를 사용합니다.")
        return moving_img, fixed_img
        
    return moving_resampled, fixed_img

def get_label_centroid(image, label):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    return np.array(stats.GetCentroid(label))

def create_centroid_translation_transform(fixed_kidneys, moving_kidneys):
    """중점 기준 평행이동 transform 생성"""
    # 중심점 추출
    p_left_3d  = get_label_centroid(fixed_kidneys, 10)
    p_right_3d = get_label_centroid(fixed_kidneys, 20)
    t_left_3d  = get_label_centroid(moving_kidneys, 10)
    t_right_3d = get_label_centroid(moving_kidneys, 20)

    # 중점 계산
    p_center = (p_left_3d + p_right_3d) / 2
    t_center = (t_left_3d + t_right_3d) / 2

    # 평행이동 transform 생성
    transform = sitk.TranslationTransform(3)
    translation = t_center - p_center
    transform.SetOffset(translation.tolist())
    
    return transform

def create_xy_rotation_transform(fixed_kidneys, moving_kidneys):
    """XY 평면 회전 transform 생성"""
    # 중심점 추출
    p_left_3d  = get_label_centroid(fixed_kidneys, 10)
    p_right_3d = get_label_centroid(fixed_kidneys, 20)
    t_left_3d  = get_label_centroid(moving_kidneys, 10)
    t_right_3d = get_label_centroid(moving_kidneys, 20)

    # 중점 계산
    p_center = (p_left_3d + p_right_3d) / 2

    # XY 좌표 추출
    p_left, p_right = p_left_3d[:2], p_right_3d[:2]
    t_left, t_right = t_left_3d[:2], t_right_3d[:2]

    # 회전각 계산
    v_patient  = p_right - p_left
    v_template = t_right - t_left
    unit_vp = v_patient / np.linalg.norm(v_patient)
    unit_vt = v_template / np.linalg.norm(v_template)
    cos_theta = np.clip(np.dot(unit_vp, unit_vt), -1.0, 1.0)
    sin_theta = np.cross(unit_vp, unit_vt)
    theta = np.arctan2(sin_theta, cos_theta)

    # 회전 transform 생성
    transform = sitk.AffineTransform(3)
    R = np.eye(3)
    R[0, 0], R[0, 1] = np.cos(theta), -np.sin(theta)
    R[1, 0], R[1, 1] = np.sin(theta),  np.cos(theta)
    transform.SetMatrix(R.flatten().tolist())
    transform.SetCenter(p_center.tolist())
    
    return transform

def create_scaling_transform(fixed_kidneys, moving_kidneys):
    """스케일링 transform 생성"""
    # 중심점 추출
    p_left_3d  = get_label_centroid(fixed_kidneys, 10)
    p_right_3d = get_label_centroid(fixed_kidneys, 20)
    t_left_3d  = get_label_centroid(moving_kidneys, 10)
    t_right_3d = get_label_centroid(moving_kidneys, 20)

    # 중점 계산
    p_center = (p_left_3d + p_right_3d) / 2

    # 스케일링 팩터 계산
    dist_patient = np.linalg.norm(p_right_3d - p_left_3d)
    dist_template = np.linalg.norm(t_right_3d - t_left_3d)
    scale = dist_template / dist_patient

    # 스케일링 transform 생성
    transform = sitk.AffineTransform(3)
    R = np.eye(3) * scale
    transform.SetMatrix(R.flatten().tolist())
    transform.SetCenter(p_center.tolist())
    
    return transform

def create_z_shear_transform(fixed_kidneys, moving_kidneys):
    """Z축 전단 transform 생성"""
    # 중심점 추출
    p_left_3d  = get_label_centroid(fixed_kidneys, 10)
    p_right_3d = get_label_centroid(fixed_kidneys, 20)
    t_left_3d  = get_label_centroid(moving_kidneys, 10)
    t_right_3d = get_label_centroid(moving_kidneys, 20)

    # 중점 계산
    p_center = (p_left_3d + p_right_3d) / 2

    # Z축 차이 계산
    dz_left = t_left_3d[2] - p_left_3d[2]
    dz_right = t_right_3d[2] - p_right_3d[2]

    # 전단 계수 계산
    dx = p_right_3d[0] - p_left_3d[0]
    dy = p_right_3d[1] - p_left_3d[1]
    a02 = (dz_right - dz_left) / dx if abs(dx) > 1e-3 else 0.0
    a12 = (dz_right - dz_left) / dy if abs(dy) > 1e-3 else 0.0

    # 전단 transform 생성
    transform = sitk.AffineTransform(3)
    R = np.eye(3)
    R[2, 0] = np.clip(a02, -0.2, 0.2)
    R[2, 1] = np.clip(a12, -0.2, 0.2)
    transform.SetMatrix(R.flatten().tolist())
    transform.SetCenter(p_center.tolist())
    
    return transform

def z_shear_xy_rigid_affine(moving_kidneys, fixed_kidneys):
    """단계별 transform을 순차적으로 적용"""
    # 1. 중점 기준 평행이동
    translation_transform = create_centroid_translation_transform(fixed_kidneys, moving_kidneys)
    translated_kidneys = apply_transform_to_labels(moving_kidneys, fixed_kidneys, translation_transform)
    
    # 2. XY 평면 회전
    rotation_transform = create_xy_rotation_transform(fixed_kidneys, translated_kidneys)
    rotated_kidneys = apply_transform_to_labels(translated_kidneys, fixed_kidneys, rotation_transform)
    
    # 3. 스케일링
    scaling_transform = create_scaling_transform(fixed_kidneys, rotated_kidneys)
    scaled_kidneys = apply_transform_to_labels(rotated_kidneys, fixed_kidneys, scaling_transform)
    
    # 4. Z축 전단 보정
    shear_transform = create_z_shear_transform(fixed_kidneys, scaled_kidneys)
    
    # 모든 transform을 하나로 결합
    final_transform = sitk.CompositeTransform(3)
    final_transform.AddTransform(translation_transform)
    final_transform.AddTransform(rotation_transform)
    final_transform.AddTransform(scaling_transform)
    final_transform.AddTransform(shear_transform)
    
    return final_transform

def align_and_combine_kidney_images(
    fixed_path,
    moving_path,
    output_dir,
    fixed_kidney_label=2,
    moving_av_labels=[3, 4],
    fixed_base_labels=[1, 2]
):
    """
    신장 영상을 정렬하고 결합하는 함수
    """
    print("\n=== 이미지 로드 시작 ===")
    print(f"Fixed path: {fixed_path}")
    print(f"Moving path: {moving_path}")
    
    # 1. 영상 로드
    print("\n1. 영상 로드")
    fixed = sitk.ReadImage(fixed_path, sitk.sitkUInt8)
    print("Fixed 이미지 로드 완료")
    moving = sitk.ReadImage(moving_path, sitk.sitkUInt8)
    print("Moving 이미지 로드 완료")
    
    # 디버깅: 라벨 값 확인
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    print("\n=== 라벨 값 확인 ===")
    print(f"환자 이미지 라벨: {np.unique(fixed_array)}")
    print(f"템플릿 이미지 라벨: {np.unique(moving_array)}")
    print(f"환자 이미지 shape: {fixed_array.shape}")
    print(f"템플릿 이미지 shape: {moving_array.shape}")
    
    # 2. 해상도 맞추기
    print("\n2. 해상도 맞추기")
    moving, fixed = resample_to_fixed_spacing(moving, fixed)
    print("해상도 맞추기 완료")
    
    # 3. 좌우 신장 분리
    print("\n3. 좌우 신장 분리")
    print("Fixed kidneys 분리 시작")
    fixed_kidneys = split_kidney_labels(fixed, fixed_kidney_label)
    print("Fixed kidneys 분리 완료")
    print("Moving kidneys 분리 시작")
    moving_kidneys = split_kidney_labels(moving, fixed_kidney_label)
    print("Moving kidneys 분리 완료")
    
    # 4. 변환 계산 및 적용
    print("\n4. 변환 계산 및 적용")
    transform = z_shear_xy_rigid_affine(moving_kidneys, fixed_kidneys)
    moving_av = extract_labels(moving, moving_av_labels)
    transformed_av = apply_transform_to_labels(moving_av, fixed, transform)
    
    # 5. 변환된 신장 영상 생성
    transformed_moving_kidneys = apply_transform_to_labels(moving_kidneys, fixed, transform)
        
    # 7. 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(transformed_av, os.path.join(output_dir, "transformed_av.nii.gz"))
    sitk.WriteImage(transformed_moving_kidneys, os.path.join(output_dir, "transformed_kidneys.nii.gz"))
    
    # 8. 기본 라벨 추출 및 저장
    fixed_base = extract_labels(fixed, fixed_base_labels)
    sitk.WriteImage(fixed_base, os.path.join(output_dir, "fixed_base.nii.gz"))
    
    # 9. 최종 결합 및 저장
    final = combine_images(fixed_base, transformed_av)
    sitk.WriteImage(final, os.path.join(output_dir, "combined_result.nii.gz"))
    
    return final

if __name__ == "__main__":
    # 예시 사용
    fixed_path = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\fast\data\patient_image.nii.gz"
    moving_path = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\fast\data\template.nii.gz"
    output_dir = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\fast\data\S000_segmentation.nii.gz"

    final = align_and_combine_kidney_images(
        fixed_path=fixed_path,
        moving_path=moving_path,
        output_dir=output_dir
    )