import os, sys, json, re, time
import SimpleITK as sitk
import trimesh
import pyvista as pv
from pathlib import Path
from threeDRecon import combineGLB, processNii, processMesh

def parse_info(case_path):
    # id extraction
    case_pattern = r'case_([a-f0-9\-]+)'
    case_match = re.search(case_pattern, case_path)
    case_id = case_match.group(1) if case_match else None
    # phase extraction 
    filename = os.path.basename(case_path)
    phase_pattern = r'segment_([A-Z])'
    phase_match = re.search(phase_pattern, filename)
    case_phase =  phase_match.group(1) if phase_match else None
    return case_id, case_phase

def main(case_path):
    start_time = time.time()
    _, phase = parse_info(case_path)
    base_path = Path(case_path).parent.parent
    
    img = sitk.ReadImage(case_path)
    label_array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    # spacing = img.GetSpacing()[::-1]  # (X, Y, Z) → (Z, Y, X)
    
    processed_img = processNii.preprocess(img, label_array)
    processed_label_array = sitk.GetArrayFromImage(processed_img)
    processed_spacing = processed_img.GetSpacing()[::-1]

    # get ndarray, return Trimesh scene
    construct_glb = combineGLB.combine_glb(processed_label_array, processed_spacing)

    # 1st smoothing
    print("[INFO] Step1")
    new_scene = trimesh.Scene()
    with open(os.path.join("threeDRecon", "config", "parts_config1.json"), "r") as f:
        parts_config1 = json.load(f)
    for cfg in parts_config1:
        processMesh.mesh_smoothing(
            scene=construct_glb,
            new_scene=new_scene,
            part_name=cfg["name"],
            smoothing_func=cfg.get("smoothing_func"),
            smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
            dilation_func=cfg.get("dilation_func"),
            dilation_kwargs=cfg.get("dilation_kwargs", {}),
        )

    # poisson reconstruction
    poisson_recon = processMesh.process_poisson(new_scene)
    
    # 2nd smoothing
    print("[INFO] Step2")
    final_scene = trimesh.Scene()
    with open(os.path.join("threeDRecon", "config", "parts_config2.json"), "r") as f:
        parts_config2 = json.load(f)
    for cfg in parts_config2:
        processMesh.mesh_smoothing(
            scene=poisson_recon,
            new_scene=final_scene,
            part_name=cfg["name"],
            smoothing_func=cfg.get("smoothing_func"),
            smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
            dilation_func=cfg.get("dilation_func"),
            dilation_kwargs=cfg.get("dilation_kwargs", {}),
        )

    save_dir = os.path.join(base_path, '3d')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'obj_{phase}.glb')
    final_scene.export(save_path)
    end_time = time.time()
    print(f"Process Done.\nExecution Time: {end_time - start_time:.2f} seconds")
    return save_path

if __name__ == "__main__":
    '''
    입력은 mask 경로로 받습니다.
    input = "path/case_0000/mask/segment_A.nii.gz"
    
    출력은 결과가 저장된 경로를 반환합니다.
    output = "path/case_0000/3d/obj_A.nii.gz"
    '''
    case_path = r".\data\case_0002_fallbacktest\mask\segment_A.nii.gz"
    result = main(case_path)