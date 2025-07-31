import os
import SimpleITK as sitk
import numpy as np

# 파일 경로
seg_path = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\dataset\nii\S001_segmentation.nii.gz"
kidney_path = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\nii\totalseg_kidney\S001.nii.gz"
output_path = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\dataset\nii\S001_segmentation_relabel.nii.gz"

# 기존 segmentation 읽기
seg_img = sitk.ReadImage(seg_path)
seg_arr = sitk.GetArrayFromImage(seg_img)  # (z, y, x)

# 1 → 3, 2 → 4
seg_arr[seg_arr == 1] = 3
seg_arr[seg_arr == 2] = 4

# Kidney 라벨 읽기
kidney_img = sitk.ReadImage(kidney_path)
kidney_arr = sitk.GetArrayFromImage(kidney_img)  # (z, y, x)

# label == 1 인 영역만 선택
kidney_mask = (kidney_arr == 1).astype(np.uint8)

# 기존 segmentation 배열에 label=2로 병합 (덮어쓰지 않도록 기존 값 0인 곳에만)
seg_arr = np.where((kidney_mask == 1) & (seg_arr == 0), 2, seg_arr)

# 저장
new_seg_img = sitk.GetImageFromArray(seg_arr)
new_seg_img.CopyInformation(seg_img)
sitk.WriteImage(new_seg_img, output_path)

print("완료:", output_path)
