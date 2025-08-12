import os, sys, json, re
import SimpleITK as sitk
import trimesh
import pyvista as pv
from pathlib import Path
from threeDRecon import combineGLB, processNii, processMesh
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import numpy as np

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

def process_single_mesh(scene_dict, cfg):
    """단일 메시 처리를 위한 워커 함수 - 메모리 효율적인 버전"""
    part_name = cfg["name"]
    
    # 해당 part만 추출하여 처리
    if part_name not in scene_dict or scene_dict[part_name] is None:
        print(f"[WARN] '{part_name}' not found in scene.")
        return None
    
    mesh_data = scene_dict[part_name]
    mesh = trimesh.Trimesh(
        vertices=mesh_data['vertices'],
        faces=mesh_data['faces']
    )
    
    # 단일 메시로 Scene 구성
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=part_name)
    
    # 새로운 Scene 생성
    new_scene = trimesh.Scene()
    
    # 메시 스무딩 처리
    processMesh.mesh_smoothing(
        scene=scene,
        new_scene=new_scene,
        part_name=part_name,
        smoothing_func=cfg.get("smoothing_func"),
        smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
        dilation_func=cfg.get("dilation_func"),
        dilation_kwargs=cfg.get("dilation_kwargs", {}),
    )
    
    # 결과 반환
    if part_name in new_scene.geometry:
        processed_mesh = new_scene.geometry[part_name]
        return {
            'name': part_name,
            'vertices': processed_mesh.vertices.copy(),
            'faces': processed_mesh.faces.copy()
        }
    
    return None

def scene_to_dict(scene):
    """Scene을 직렬화 가능한 딕셔너리로 변환"""
    scene_dict = {}
    for name, mesh in scene.geometry.items():
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            scene_dict[name] = {
                'vertices': mesh.vertices,
                'faces': mesh.faces
            }
        else:
            scene_dict[name] = None
    return scene_dict

def dict_to_scene(scene_dict):
    """딕셔너리를 Scene으로 변환"""
    scene = trimesh.Scene()
    for name, mesh_data in scene_dict.items():
        if mesh_data is not None:
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces']
            )
            scene.add_geometry(mesh, node_name=name)
    return scene

def parallel_mesh_smoothing(scene, parts_config, max_workers=None, use_threads=True):
    """
    개선된 병렬 메시 스무딩 함수
    
    Args:
        scene: 입력 trimesh.Scene
        parts_config: 설정 리스트
        max_workers: 최대 워커 수 (None이면 자동 설정)
        use_threads: True면 스레드 사용, False면 프로세스 사용
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(parts_config), 4)  # 최대 4개로 제한
    
    # Scene을 직렬화 가능한 형태로 변환
    scene_dict = scene_to_dict(scene)
    
    # 실제 존재하는 part만 필터링
    valid_configs = []
    for cfg in parts_config:
        part_name = cfg["name"]
        if part_name in scene_dict and scene_dict[part_name] is not None:
            valid_configs.append(cfg)
        else:
            print(f"[WARN] '{part_name}' not found in scene, skipping...")
    
    if not valid_configs:
        print("[WARN] 처리할 유효한 메시가 없습니다.")
        return scene
    
    print(f"[INFO] {len(valid_configs)}개의 메시를 {max_workers}개 워커로 병렬 처리 중...")
    
    # 스레드 vs 프로세스 선택
    if use_threads:
        # I/O 집약적이고 GIL 해제가 많은 작업에 적합
        ExecutorClass = ThreadPoolExecutor
    else:
        # CPU 집약적 작업에 적합하지만 pickle 오버헤드 있음
        ExecutorClass = ProcessPoolExecutor
    
    # 병렬 실행
    final_scene = trimesh.Scene()
    
    with ExecutorClass(max_workers=max_workers) as executor:
        # partial을 사용하여 scene_dict를 고정
        worker_func = functools.partial(process_single_mesh, scene_dict)
        results = list(executor.map(worker_func, valid_configs))
    
    # 결과 합치기
    for result in results:
        if result is not None:
            mesh = trimesh.Trimesh(
                vertices=result['vertices'],
                faces=result['faces']
            )
            mesh.metadata["name"] = result['name']
            final_scene.add_geometry(mesh, node_name=result['name'])
    
    # 처리되지 않은 메시들도 추가 (설정에 없는 메시들)
    processed_names = {result['name'] for result in results if result is not None}
    for name, mesh_data in scene_dict.items():
        if name not in processed_names and mesh_data is not None:
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces']
            )
            mesh.metadata["name"] = name
            final_scene.add_geometry(mesh, node_name=name)
            print(f"[INFO] '{name}' 메시를 처리 없이 추가")
    
    print(f"[INFO] 최종 Scene에 포함된 메시: {list(final_scene.geometry.keys())}")
    return final_scene

def sequential_mesh_smoothing(scene, parts_config):
    """
    순차적 메시 스무딩 - 비교용
    """
    new_scene = trimesh.Scene()
    
    print(f"[DEBUG] 순차 처리 - 입력 메시: {list(scene.geometry.keys())}")
    print(f"[DEBUG] 순차 처리 - 설정 파일 이름: {[cfg['name'] for cfg in parts_config]}")
    
    processed_count = 0
    for cfg in parts_config:
        part_name = cfg["name"]
        if part_name in scene.geometry:
            processMesh.mesh_smoothing(
                scene=scene,
                new_scene=new_scene,
                part_name=part_name,
                smoothing_func=cfg.get("smoothing_func"),
                smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
                dilation_func=cfg.get("dilation_func"),
                dilation_kwargs=cfg.get("dilation_kwargs", {}),
            )
            processed_count += 1
        else:
            print(f"[WARN] '{part_name}' not found in scene, skipping...")
    
    # 처리되지 않은 메시들도 추가
    processed_names = {cfg["name"] for cfg in parts_config if cfg["name"] in scene.geometry}
    for name, mesh in scene.geometry.items():
        if name not in processed_names:
            mesh_copy = mesh.copy()
            mesh_copy.metadata["name"] = name
            new_scene.add_geometry(mesh_copy, node_name=name)
            print(f"[INFO] '{name}' 메시를 처리 없이 추가")
    
    print(f"[DEBUG] 순차 처리 완료 - 처리된 메시: {processed_count}개, 최종 메시: {list(new_scene.geometry.keys())}")
    return new_scene

def main(case_path, enable_parallel=True, use_threads=True):
    """
    메인 파이프라인 함수
    
    Args:
        case_path: 입력 케이스 경로
        enable_parallel: 병렬 처리 활성화 여부
        use_threads: True면 스레드, False면 프로세스 사용
    """
    import time
    
    _, phase = parse_info(case_path)
    base_path = Path(case_path).parent.parent
    
    print(f"[INFO] 케이스 처리 시작: {case_path}")
    print(f"[INFO] 병렬 처리: {'활성화' if enable_parallel else '비활성화'}")
    if enable_parallel:
        print(f"[INFO] 병렬 처리 방식: {'스레드' if use_threads else '프로세스'}")
    
    # 전처리
    print("[INFO] 1/5 - NIfTI 전처리 중...")
    start_time = time.time()
    
    img = sitk.ReadImage(case_path)
    label_array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    
    processed_img = processNii.preprocess(img, label_array)
    processed_label_array = sitk.GetArrayFromImage(processed_img)
    processed_spacing = processed_img.GetSpacing()[::-1]

    print(f"    전처리 완료: {time.time() - start_time:.2f}초")

    # 초기 GLB 생성
    print("[INFO] 2/5 - 초기 GLB 생성 중...")
    start_time = time.time()
    
    construct_glb = combineGLB.combine_glb(processed_label_array, processed_spacing)
    print(f"    GLB 생성 완료: {time.time() - start_time:.2f}초")
    print(f"    생성된 메시 개수: {len(construct_glb.geometry)}")
    print(f"    생성된 메시 이름들: {list(construct_glb.geometry.keys())}")

    # 1차 스무딩
    print("[INFO] 3/5 - 1차 스무딩 중...")
    start_time = time.time()
    
    with open(os.path.join("threeDRecon", "config", "parts_config1.json"), "r") as f:
        parts_config1 = json.load(f)
    
    if enable_parallel:
        new_scene = parallel_mesh_smoothing(
            construct_glb, parts_config1, 
            max_workers=4, use_threads=use_threads
        )
    else:
        new_scene = sequential_mesh_smoothing(construct_glb, parts_config1)
    
    print(f"    1차 스무딩 완료: {time.time() - start_time:.2f}초")

    # Poisson 재구성
    print("[INFO] 4/5 - Poisson 재구성 중...")
    start_time = time.time()
    
    poisson_recon = processMesh.process_poisson(new_scene)
    print(f"    Poisson 재구성 완료: {time.time() - start_time:.2f}초")
    
    # 2차 스무딩
    print("[INFO] 5/5 - 2차 스무딩 중...")
    start_time = time.time()
    
    with open(os.path.join("threeDRecon", "config", "parts_config2.json"), "r") as f:
        parts_config2 = json.load(f)
    
    if enable_parallel:
        final_scene = parallel_mesh_smoothing(
            poisson_recon, parts_config2, 
            max_workers=4, use_threads=use_threads
        )
    else:
        final_scene = sequential_mesh_smoothing(poisson_recon, parts_config2)
    
    print(f"    2차 스무딩 완료: {time.time() - start_time:.2f}초")

    # 저장
    save_dir = os.path.join(base_path, '3d')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'obj_{phase}.glb')
    
    # 빈 scene 체크
    if len(final_scene.geometry) == 0:
        print(f"[ERROR] 최종 Scene이 비어있어 저장할 수 없습니다!")
        print(f"[DEBUG] 2차 스무딩 후 메시 개수: {len(final_scene.geometry)}")
        return None
    
    final_scene.export(save_path)
    
    print(f"[INFO] 최종 결과 저장: {save_path}")
    print(f"[INFO] 최종 저장된 메시 개수: {len(final_scene.geometry)}")
    return save_path

if __name__ == "__main__":
    '''
    입력은 mask 경로로 받습니다.
    input = "path/case_0000/mask/segment_A.nii.gz"
    
    출력은 결과가 저장된 경로를 반환합니다.
    output = "path/case_0000/3d/obj_A.glb"
    
    사용법:
    python main.py                          # 기본: 병렬(스레드) 처리
    python main.py --sequential             # 순차 처리
    python main.py --parallel --processes   # 병렬(프로세스) 처리
    '''
    import time
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='ARNA-3D 메디컬 이미지 처리 파이프라인')
    parser.add_argument('--case-path', default=r".\data\case_0004\mask\segment_A.nii.gz",
                        help='입력 케이스 파일 경로')
    parser.add_argument('--sequential', action='store_true',
                        help='순차 처리 (병렬 처리 비활성화)')
    parser.add_argument('--processes', action='store_true',
                        help='프로세스 기반 병렬 처리 (기본: 스레드)')
    parser.add_argument('--benchmark', action='store_true',
                        help='순차/병렬 처리 성능 비교')
    
    args = parser.parse_args()
    
    case_path = args.case_path
    
    if args.benchmark:
        print("="*60)
        print("성능 비교 벤치마크 시작")
        print("="*60)
        
        # 1. 순차 처리
        print("\n[BENCHMARK] 순차 처리 테스트")
        print("-"*40)
        start_time = time.time()
        result1 = main(case_path, enable_parallel=False)
        seq_time = time.time() - start_time
        print(f"순차 처리 완료: {seq_time:.2f}초")
        
        # 2. 병렬(스레드) 처리  
        print("\n[BENCHMARK] 병렬(스레드) 처리 테스트")
        print("-"*40)
        start_time = time.time()
        result2 = main(case_path, enable_parallel=True, use_threads=True)
        thread_time = time.time() - start_time
        print(f"병렬(스레드) 처리 완료: {thread_time:.2f}초")
        
        # 3. 병렬(프로세스) 처리
        print("\n[BENCHMARK] 병렬(프로세스) 처리 테스트")
        print("-"*40)
        start_time = time.time()
        result3 = main(case_path, enable_parallel=True, use_threads=False)
        process_time = time.time() - start_time
        print(f"병렬(프로세스) 처리 완료: {process_time:.2f}초")
        
        # 결과 요약
        print("\n" + "="*60)
        print("성능 비교 결과")
        print("="*60)
        print(f"순차 처리:        {seq_time:.2f}초 (기준)")
        print(f"병렬(스레드):      {thread_time:.2f}초 (가속비: {seq_time/thread_time:.2f}x)")
        print(f"병렬(프로세스):    {process_time:.2f}초 (가속비: {seq_time/process_time:.2f}x)")
        print("\n추천 설정:")
        if thread_time < seq_time and thread_time <= process_time:
            print("→ 병렬(스레드) 처리 권장")
        elif process_time < seq_time and process_time < thread_time:
            print("→ 병렬(프로세스) 처리 권장")
        else:
            print("→ 순차 처리 권장")
            
    else:
        # 일반 실행
        start_time = time.time()
        
        if args.sequential:
            result = main(case_path, enable_parallel=False)
        else:
            result = main(case_path, enable_parallel=True, use_threads=not args.processes)
        
        end_time = time.time()
        
        print("\n" + "="*60)
        print(f"처리 완료! 결과 저장 위치: {result}")
        print(f"총 실행 시간: {end_time - start_time:.2f}초")
        print("="*60)