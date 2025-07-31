import time, argparse, os, sys, json
import SimpleITK as sitk
import trimesh
import pyvista as pv

sys.path.append(r'C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523')
from threeDRecon import combineGLB, processNii, processMesh

if __name__ == '__main__':
    start_time = time.time()
    # parser = argparse.ArgumentParser(description='Converted script from main_0522.ipynb')
    # parser.add_argument('input', help='Input directory or file')
    # parser.add_argument('output', help='Output directory or file')
    # parser.add_argument('--debug', action='store_true', help='Save intermediate results if set')
    # args = parser.parse_args()

    subject_id = "S000"
    base_folder = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523"
    nii_folder = os.path.join(base_folder, r'dataset\nii')
    glb_folder = os.path.join(base_folder, r'dataset\glb')
    output_folder = os.path.join(base_folder, r'dataset\output')
    threeDRecon_folder = os.path.join(base_folder, r'threeDRecon')
    nii_path = os.path.join(nii_folder, f"{subject_id}_segmentation.nii.gz")

    caseid_folder = os.path.join(output_folder, f"case{subject_id}")
    if not os.path.exists(caseid_folder):
        os.makedirs(caseid_folder)
    debug = True

    img = sitk.ReadImage(nii_path)
    label_array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = img.GetSpacing()[::-1]  # (X, Y, Z) → (Z, Y, X)

    #============ 1. nii전처리 ============
    pp_img = processNii.preprocess(img, label_array)
    pp_label_array = sitk.GetArrayFromImage(pp_img)
    pp_spacing = pp_img.GetSpacing()[::-1]
    print(f"[INFO] Preprocessing label array shape: {pp_label_array.shape}, spacing: {pp_spacing}")
    if debug:
        # Save preprocessed image for debugging
        debug_nii_path = os.path.join(caseid_folder, f"{subject_id}_preprocessed.nii.gz")
        sitk.WriteImage(pp_img, debug_nii_path)
        print(f"[DEBUG] Saved preprocessed NII: {debug_nii_path}")

    #============ 2. 초기 glb 생성 ============
    # get ndarray, return Trimesh scene
    construct_glb = combineGLB.combine_glb(pp_label_array, pp_spacing)
    if debug:
        # Save combined GLB for debugging
        debug_glb_path = os.path.join(caseid_folder, f"{subject_id}_combined.glb")
        construct_glb.export(debug_glb_path)
        print(f"[DEBUG] Saved combined GLB: {debug_glb_path}")

    #============ 3. 1차 스무딩 ============
    new_scene = trimesh.Scene()
    with open(os.path.join(threeDRecon_folder, r"config\parts_config1.json"), "r") as f:
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
    if debug:
        # for key, mesh in new_scene.geometry.items():
        #     if mesh.metadata.get("name") == "Renal_a":
        #         print("Renal A Scailing")
        #         mesh.apply_scale(2)

        time1 = time.time()
        print(f"[DEBUG] 1st smoothing completed in {time1 - start_time:.2f} seconds")
        debug_first_smoothed_path = os.path.join(caseid_folder, f"{subject_id}_1st_smoothed.glb")
        new_scene.export(debug_first_smoothed_path)
        print(f"[DEBUG] Saved 1st smoothed GLB: {debug_first_smoothed_path}")

    #============ 4. 푸아송 재구성 ============
    poisson_recon = processMesh.process_poisson(new_scene)
    if debug:
        time2 = time.time()
        print(f"[DEBUG] Poisson reconstruction completed in {time2 - time1:.2f} seconds")
        debug_poisson_path = os.path.join(caseid_folder, f"{subject_id}_poisson_recon.glb")
        poisson_recon.export(debug_poisson_path)
        print(f"[DEBUG] Saved Poisson reconstructed GLB: {debug_poisson_path}")
    
    #============ 5. 2차 스무딩 ============
    final_scene = trimesh.Scene()
    with open(os.path.join(threeDRecon_folder, r"config\parts_config2.json"), "r") as f:
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
    if debug:
        time3 = time.time()
        print(f"[DEBUG] 2nd smoothing completed in {time3 - time2:.2f} seconds")
        # debug_second_smoothed_path = os.path.join(caseid_folder, f"{subject_id}_2nd_smoothed.glb")
        # final_scene.export(debug_second_smoothed_path)
        # print(f"[DEBUG] Saved 2nd smoothed GLB: {debug_second_smoothed_path}")
    
    # 시간 측정
    end_time = time.time()
    print(f'[INFO] Total execution time: {end_time - start_time:.2f} seconds')

    # 저장
    file_path = os.path.join(caseid_folder, f"{subject_id}_2nd_smoothed.glb")
    final_scene.export(file_path)
    print(f"[INFO] Saved smoothed GLB: {file_path}")