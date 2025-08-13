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
        print(f"[WARN] Skipping {part_name}: mesh not found in scene.")
        return

    print(f"[INFO] Processing {part_name}")
    # Convert to PyVista PolyData: faces=[n, v0, v1, v2, ...]
    vertices = mesh.vertices
    faces = np.hstack([[3, *f] for f in mesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    # Dilation
    if dilation_func:
        dilation_func = DILATION_FUNC_MAP.get(dilation_func)
        if dilation_func is None:
            raise ValueError(f"[ERROR] Unknown dilation_type: {dilation_type}")
        print(f"{'':7}- Apply Dilation")
        pv_mesh = dilation_func(pv_mesh, **(dilation_kwargs or {}))

    # Smoothing
    if smoothing_func:
        smoothing_func = SMOOTHING_FUNC_MAP.get(smoothing_func)
        if smoothing_func is None:
            raise ValueError(f"[ERROR] Unknown smoothing_type: {smoothing_type}")
        print(f"{'':7}- Apply Smoothing")
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
        raise ValueError("[ERROR] Input mesh_list empty")
    
    print(f"{'':7}- Merged mesh: {len(mesh_list)}")

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
        print(f"{'':7}- Connected components: {len(components)} - using largest")
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
        print("[INFO] Processing Artery group")
        smoothed_artery = poisson_reconstruction(artery_meshes, depth=8)
        smoothed_artery.metadata["name"] = "Artery"
        rec_scene.add_geometry(smoothed_artery, node_name="Artery")

    # Vein System (Vein + Renal_v)
    vein_meshes = [name_to_mesh[name] for name in vein_group if name in name_to_mesh]
    if vein_meshes:
        print("[INFO] Processing Vein group")
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