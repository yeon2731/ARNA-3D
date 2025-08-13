import SimpleITK as sitk
import numpy as np
from skimage import measure
from scipy.spatial import cKDTree
import trimesh
from trimesh.sample import sample_surface_even
import os
import open3d as o3d
import pyvista as pv

LABELS = {
    "Tumor": 1,
    "Kidney": [1, 2],   # Include tumor
    "Artery": 3,
    "Vein": 4,
    "Ureter": 5,
    "Fat": 6,
    "Renal_a": 7,
    "Renal_v": 8
}

def create_mask(label_array, label):
    if isinstance(label, list):
        return np.isin(label_array, label)
    else:
        return label_array == label

def mask_to_mesh(mask, spacing=(1.0, 1.0, 1.0)):
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

def mask_to_mesh_fixnormal(mask, spacing=(1.0, 1.0, 1.0)):
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # 메시 정리 및 노멀 수정
    mesh.remove_degenerate_faces()  # 퇴화된 면 제거
    mesh.remove_duplicate_faces()   # 중복된 면 제거
    mesh.merge_vertices()           # 중복된 버텍스 병합
    mesh.fix_normals()              # 노멀 방향 일관성 수정
    return mesh

def rotate_and_center(scene):
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    
    # 회전 행렬: R1: Z축 기준 시계방향 90도 (-90도)
    angle_rad = np.pi / 2
    R1 = trimesh.transformations.rotation_matrix(angle_rad, direction=[0, 0, 1], point=center)
    R2 = trimesh.transformations.rotation_matrix(angle_rad, direction=[0, -1, 0], point=center)

    # 중심 이동 행렬
    T = trimesh.transformations.translation_matrix(-center)
    # X축 반전 행렬
    X_flip = np.diag([-1, 1, 1, 1])
    
    # 변환
    combined_transform = trimesh.transformations.concatenate_matrices(X_flip, T, R2, R1)
    scene.apply_transform(combined_transform)
    
    # 변환을 메시 데이터에 직접 적용
    transformed_geometries = scene.dump()
    new_scene = trimesh.Scene(transformed_geometries)
    return new_scene

def combine_glb(label_array, spacing):
    # ===== Scene 구성 =====
    scene = trimesh.Scene() # 결과 Scene
    for name, label in LABELS.items():
        mask = create_mask(label_array, label)

        if not np.any(mask):
            print(f"[WARN] Skipping {name}: mask is empty.")
            continue

        try:
            if name in ["Kidney", "Fat"]:   # Kidney/Fat Split (L/R 매핑)
                mesh = mask_to_mesh_fixnormal(mask, spacing=spacing)
                parts = mesh.split(only_watertight=False)
                if len(parts) != 2:  # 최대 크기 2개 선택
                    parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)
                parts = sorted(parts[:2], key=lambda m: m.centroid[2])  # x축 기준 L/R 정렬
                sides = ["L", "R"]
                for part, side in zip(parts, sides):
                    part_name = f"{name}-{side}"
                    part.metadata["name"] = part_name
                    scene.add_geometry(part, node_name=part_name)
            elif name in ["Tumor"]:    # Tumor 노멀 재계산
                mesh = mask_to_mesh_fixnormal(mask, spacing=spacing)
                mesh.metadata["name"] = name
                scene.add_geometry(mesh, node_name=name)
            else:
                mesh = mask_to_mesh(mask, spacing=spacing)
                mesh.metadata["name"] = name
                scene.add_geometry(mesh, node_name=name)

        except Exception as e:
            print(f"[Error] Error processing {name}: {e}")

    scene = rotate_and_center(scene)
    return scene