import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d


def default_dilation(mesh: pv.PolyData, offset: float = -0.5) -> pv.PolyData:
    mesh.compute_normals(inplace=True)
    mesh.points += offset * mesh.point_normals
    return mesh

def mesh_smoothing(
    scene: trimesh.Scene,
    new_scene: trimesh.Scene,
    part_name: str,
    smoothing_func: str = None,
    smoothing_kwargs: dict = None,
    dilation_func: str = None,
    dilation_kwargs: dict = None,
    ):
    """
    Extract a mesh by name from the scene, apply optional smoothing and dilation,
    then convert back to trimesh and add to new_scene.

    :param scene: original trimesh.Scene
    :param new_scene: output trimesh.Scene
    :param part_name: key name of the mesh geometry
    :param smoothing_func: function to smooth a PyVista mesh (e.g., PolyData.smooth_taubin or laplacian)
    :param smoothing_kwargs: keyword arguments for smoothing_func
    :param dilation_func: function to dilate (offset) a PyVista mesh along normals
    :param dilation_kwargs: keyword arguments for dilation_func
    """
    
    SMOOTHING_FUNC_MAP = {
        "laplacian": pv.PolyData.smooth,
        "taubin": pv.PolyData.smooth_taubin,
        None: lambda mesh, **kwargs: mesh  # None일 경우 아무 처리도 하지 않음
    }
    DILATION_FUNC_MAP = {
        "default": default_dilation,
        None: lambda mesh, **kwargs: mesh  # None일 경우 아무 처리도 하지 않음
    }

    mesh = scene.geometry.get(part_name)
    if mesh is None:
        print(f"[WARN] '{part_name}' not found in scene.")
        return

    print(f"[INFO]: Processing {part_name}")
    # Convert to PyVista PolyData: faces=[n, v0, v1, v2, ...]
    vertices = mesh.vertices
    faces = np.hstack([[3, *f] for f in mesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    # Dilation
    if dilation_func:
        dilation_func = DILATION_FUNC_MAP.get(dilation_func)
        if dilation_func is None:
            raise ValueError(f"Unknown dilation_type: {dilation_type}")
        print(f"    Dilation: {part_name}")
        pv_mesh = dilation_func(pv_mesh, **(dilation_kwargs or {}))

    # Smoothing
    if smoothing_func:
        smoothing_func = SMOOTHING_FUNC_MAP.get(smoothing_func)
        if smoothing_func is None:
            raise ValueError(f"Unknown smoothing_type: {smoothing_type}")
        print(f"    Smoothing: {part_name}")
        pv_mesh = smoothing_func(pv_mesh, **(smoothing_kwargs or {}))

    # Convert back to trimesh
    final_mesh = trimesh.Trimesh(
        vertices=pv_mesh.points,
        faces=pv_mesh.faces.reshape(-1, 4)[:, 1:]
    )
    final_mesh.metadata["name"] = part_name
    new_scene.add_geometry(final_mesh, node_name=part_name)

def poisson_reconstruction(mesh_list, depth=8):
    if not mesh_list:
        raise ValueError("입력 mesh_list가 비어 있습니다.")
    
    print(f"[INFO] 병합된 mesh 수: {len(mesh_list)}")

    # 병합
    merged = trimesh.util.concatenate(mesh_list)

    # Trimesh → Open3D 변환
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(merged.vertices),
        triangles=o3d.utility.Vector3iVector(merged.faces)
    )
    o3d_mesh.compute_vertex_normals()

    # 포인트 샘플링
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=40000)
    pcd.estimate_normals()

    # Poisson 재구성
    mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh_out = mesh_out.crop(pcd.get_axis_aligned_bounding_box())

    # 다시 Trimesh로 변환
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_out.vertices),
        faces=np.asarray(mesh_out.triangles)
    )

    # Poisson 결과에 대해 LCC 적용
    components = tri_mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"[INFO] Poisson 결과에서 연결된 컴포넌트 개수: {len(components)} → 가장 큰 컴포넌트만 사용")
        tri_mesh = max(components, key=lambda c: c.area)

    return tri_mesh

def process_poisson(scene):
    # 이름 → mesh 매핑
    name_to_mesh = {mesh.metadata.get("name", k): mesh for k, mesh in scene.geometry.items()}

    # 병합 대상 그룹 정의
    artery_group = ["Artery", "Renal_a"]
    vein_group = ["Vein", "Renal_v"]

    # 결과 scene
    rec_scene = trimesh.Scene()

    # Artery (Aretry + Renal_a)
    artery_meshes = [name_to_mesh[name] for name in artery_group if name in name_to_mesh]
    if artery_meshes:
        print("[INFO] Artery 그룹 처리 중...")
        smoothed_artery = poisson_reconstruction(artery_meshes, depth=8)
        smoothed_artery.metadata["name"] = "Artery"
        rec_scene.add_geometry(smoothed_artery, node_name="Artery")

    # Vein System (Vein + Renal_v)
    vein_meshes = [name_to_mesh[name] for name in vein_group if name in name_to_mesh]
    if vein_meshes:
        print("[INFO] Vein 그룹 처리 중...")
        smoothed_vein = poisson_reconstruction(vein_meshes, depth=8)
        smoothed_vein.metadata["name"] = "Vein"
        rec_scene.add_geometry(smoothed_vein, node_name="Vein")

    # 나머지 구조물은 그대로 추가
    excluded = set(artery_group + vein_group)
    for name, mesh in name_to_mesh.items():
        if name not in excluded:
            mesh.name = name
            mesh.metadata["name"] = name
            rec_scene.add_geometry(mesh, node_name=name)

    return rec_scene

# def poisson_reconstruction(mesh_list, depth=8):
#     if not mesh_list:
#         raise ValueError("입력 mesh_list가 비어 있습니다.")
    
#     # 메시별 포인트 수 설정
#     point_config = {
#         "Renal_a": 10000,
#         "Artery": 10000,
#         "default": 10000
#     }
    
#     print(f"[INFO] 개별 mesh 수: {len(mesh_list)}")
    
#     reconstructed_meshes = []
    
#     # 각 메시를 개별적으로 Poisson 재구성
#     for i, mesh in enumerate(mesh_list):
#         mesh_name = mesh.metadata.get("name", "")
#         num_points = point_config.get(mesh_name, point_config["default"])
        
#         print(f"[INFO] {i+1}/{len(mesh_list)} 메시 ({mesh_name}) Poisson 재구성 중... (포인트: {num_points}개)")
        
#         # Trimesh → Open3D 변환
#         o3d_mesh = o3d.geometry.TriangleMesh(
#             vertices=o3d.utility.Vector3dVector(mesh.vertices),
#             triangles=o3d.utility.Vector3iVector(mesh.faces)
#         )
#         o3d_mesh.compute_vertex_normals()

#         # 포인트 샘플링
#         pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points)
#         pcd.estimate_normals()

#         # Poisson 재구성
#         mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
#         mesh_out = mesh_out.crop(pcd.get_axis_aligned_bounding_box())

#         # 다시 Trimesh로 변환
#         tri_mesh = trimesh.Trimesh(
#             vertices=np.asarray(mesh_out.vertices),
#             faces=np.asarray(mesh_out.triangles)
#         )

#         # LCC 적용
#         components = tri_mesh.split(only_watertight=False)
#         if len(components) > 1:
#             print(f"[INFO] 메시 {i+1}에서 연결된 컴포넌트 개수: {len(components)} → 가장 큰 컴포넌트만 사용")
#             tri_mesh = max(components, key=lambda c: c.area)
        
#         # Renal_a인 경우 dilation 적용
#         if mesh_name == "Renal_a":
#             print(f"[INFO] {mesh_name}에 dilation 적용 중...")
            
#             # Trimesh → PyVista 변환
#             pv_mesh = pv.PolyData(tri_mesh.vertices, 
#                                  np.c_[np.full(len(tri_mesh.faces), 3), tri_mesh.faces])
            
#             # Dilation 적용
#             pv_mesh = default_dilation(pv_mesh, offset=2)
            
#             # PyVista → Trimesh 변환
#             faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # 첫 번째 열(3) 제거
#             tri_mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)
        
#         reconstructed_meshes.append(tri_mesh)
    
#     # 재구성된 메시들을 병합
#     print(f"[INFO] {len(reconstructed_meshes)}개의 재구성된 메시 병합 중...")
#     final_merged = trimesh.util.concatenate(reconstructed_meshes)
    
#     # 최종 병합 결과에 대해서도 LCC 적용
#     # components = final_merged.split(only_watertight=False)
#     # if len(components) > 1:
#     #     print(f"[INFO] 최종 병합에서 연결된 컴포넌트 개수: {len(components)} → 가장 큰 컴포넌트만 사용")
#     #     final_merged = max(components, key=lambda c: c.area)

#     return final_merged

def process_poisson(scene):
    # 이름 → mesh 매핑
    name_to_mesh = {mesh.metadata.get("name", k): mesh for k, mesh in scene.geometry.items()}

    # 병합 대상 그룹 정의
    artery_group = ["Artery", "Renal_a"]
    vein_group = ["Vein", "Renal_v"]

    # 결과 scene
    rec_scene = trimesh.Scene()

    # Artery (Aretry + Renal_a)
    artery_meshes = [name_to_mesh[name] for name in artery_group if name in name_to_mesh]
    if artery_meshes:
        print("[INFO] Artery 그룹 처리 중...")
        smoothed_artery = poisson_reconstruction(artery_meshes, depth=8)
        smoothed_artery.metadata["name"] = "Artery"
        rec_scene.add_geometry(smoothed_artery, node_name="Artery")

    # Vein System (Vein + Renal_v)
    vein_meshes = [name_to_mesh[name] for name in vein_group if name in name_to_mesh]
    if vein_meshes:
        print("[INFO] Vein 그룹 처리 중...")
        smoothed_vein = poisson_reconstruction(vein_meshes, depth=8)
        smoothed_vein.metadata["name"] = "Vein"
        rec_scene.add_geometry(smoothed_vein, node_name="Vein")

    # 나머지 구조물은 그대로 추가
    excluded = set(artery_group + vein_group)
    for name, mesh in name_to_mesh.items():
        if name not in excluded:
            mesh.name = name
            mesh.metadata["name"] = name
            rec_scene.add_geometry(mesh, node_name=name)

    return rec_scene