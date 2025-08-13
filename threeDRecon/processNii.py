import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
import cv2
from scipy.stats import zscore

def get_largest_component(mask, n=1):
    if mask.ndim == 2:
        structure = np.ones((3,3), dtype=bool)      # 8-neighbor for 2D
    elif mask.ndim == 3:
        structure = np.ones((3,3,3), dtype=bool)    # 26-neighbor for 3D
    else:
        raise ValueError("Input mask must be 2D or 3D.")
    
    labeled, _ = scipy.ndimage.label(mask, structure=structure)
    props = regionprops(labeled)
    if not props or n <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)
    labels = [p.label for p in props_sorted[:n]]
    result = np.zeros_like(mask, dtype=np.uint8)
    for lbl in labels:
        result[labeled == lbl] = 1
    return result

def get_max_inscribed_circle(mask_2d):
    labeled, _ = scipy.ndimage.label(mask_2d)
    if labeled.max() == 0:
        return None, None
    # 가장 큰 CC
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lbl = sizes.argmax()
    cc = (labeled == lbl)
    dist = distance_transform_edt(cc)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return (y, x), dist.max()

def get_radii_array(mask_3d, z_start, z_end):
    z_dim = mask_3d.shape[0]
    radii = np.zeros((z_dim, 2))
    for z in range(z_start, z_end+1):
        slice_m = get_largest_component(mask_3d[z])
        if slice_m.sum() == 0:
            continue
        _, r_max = get_max_inscribed_circle(slice_m)
        # 최소 외접반경은 convex hull에서 centroid-vertex 최대 거리로 계산 가능
        ch = convex_hull_image(slice_m)
        props = regionprops(label(ch))
        if props:
            cy, cx = props[0].centroid
            coords = props[0].coords
            dists = np.sqrt((coords[:,0]-cy)**2 + (coords[:,1]-cx)**2)
            r_min = dists.max()
        else:
            r_min = 0
        radii[z] = [r_max, r_min]
    return radii

def get_gradient_range(mask_3d, z_start, z_end, percentile=95):
    radii = get_radii_array(mask_3d, z_start, z_end)
    # gradient
    valid = np.all(radii>0, axis=1)
    grad = np.zeros_like(valid, dtype=float)
    if valid.sum()>=2:
        grad_sub = np.gradient(radii[valid,1] - radii[valid,0])
        grad[valid] = np.abs(grad_sub)
    
    # Calculate threshold using percentile instead of fixed value
    valid_grads = grad[valid]
    if len(valid_grads) > 0:
        threshold = np.percentile(valid_grads, percentile)
    else:
        threshold = 0
    
    # z_front/back
    zf = np.argmax(grad >= threshold)
    zb = len(grad) - 1 - np.argmax((grad >= threshold)[::-1])
    return zf, zb

def get_zscore_range(mask_3d, z_start, z_end, window_size=5):
    radii = get_radii_array(mask_3d, z_start, z_end)
    z = np.arange(radii.shape[0])
    max_r = radii[:, 0]
    min_r = radii[:, 1]
        # 2. 유효 영역 필터링
    valid_mask = (min_r > 0) & (max_r > 0)
    valid_indices = np.where(valid_mask)[0]
    radius_diff_valid = np.abs(min_r[valid_mask] - max_r[valid_mask])
    z_scores_diff_valid = zscore(radius_diff_valid)

    # 3. 전체 z 크기 유지한 z-score 배열 생성
    z_scores_diff = np.zeros_like(z, dtype=float)
    z_scores_diff[valid_indices] = z_scores_diff_valid

    # 4. 연속된 0 이상 구간의 시작점 찾기
    found_first = False
    for i in range(len(z_scores_diff_valid) - window_size + 1):
        window = z_scores_diff_valid[i:i + window_size]
        if np.all(window >= 0):
            first_zero_idx_valid = i
            found_first = True
            break
    if found_first:
        zf = z[valid_indices[first_zero_idx_valid]]
    else:
        fallback_idx = np.where(z_scores_diff_valid >= 0)[0]
        zf = z[valid_indices[fallback_idx[0]]] if len(fallback_idx) > 0 else None

    # 5. 연속된 0 이상 구간의 끝점 찾기
    found_last = False
    for i in range(len(z_scores_diff_valid) - window_size, -1, -1):
        window = z_scores_diff_valid[i:i + window_size]
        if np.all(window >= 0):
            last_zero_idx_valid = i + window_size - 1
            found_last = True
            break
    if found_last:
        zb = z[valid_indices[last_zero_idx_valid]]
    else:
        fallback_idx = np.where(z_scores_diff_valid >= 0)[0]
        zb = z[valid_indices[fallback_idx[-1]]] if len(fallback_idx) > 0 else None
    return zf, zb

def interpolate_circle_bridge(mask_3d, zf, zb):
    z0, z1 = zf-1, zb+1
    n = z1 - z0 + 1
    interp = np.zeros((n, *mask_3d.shape[1:]), dtype=np.uint8)

    c0, r0 = get_max_inscribed_circle(mask_3d[z0])
    c1, r1 = get_max_inscribed_circle(mask_3d[z1])
    for i in range(n):
        alpha = i/(n-1)
        if c0 is None or c1 is None:
            continue
        y = int((1-alpha)*c0[0] + alpha*c1[0])
        x = int((1-alpha)*c0[1] + alpha*c1[1])
        r = (1-alpha)*r0 + alpha*r1

        Y, X = np.ogrid[:mask_3d.shape[1], :mask_3d.shape[2]]
        circle = ((Y-y)**2 + (X-x)**2) <= r**2
        dil = binary_dilation(circle, iterations=3)
        interp[i] = dil.astype(np.uint8)

    # 덮어쓰기
    out = mask_3d.copy()
    out[z0:z1+1] = interp
    out = get_largest_component(out, n=1)
    return out, (z0, z1)

def extract_branches(original, bridged, top_n=2):
    llc = get_largest_component(bridged, n=1)
    branches = (original.astype(bool) & ~llc.astype(bool)).astype(np.uint8)
    # return get_largest_component(branches, n=top_n)
    return branches

def get_fitted_ellipse(mask_2d):
    contours, _ = cv2.findContours(mask_2d.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    return cv2.fitEllipse(cnt)  # ((cx, cy), (major, minor), angle)

def draw_ellipse_mask(shape, ellipse):
    canvas = np.zeros(shape, dtype=np.uint8)
    if ellipse is None:
        return canvas
    pts = cv2.ellipse2Poly(
        center=(int(ellipse[0][0]), int(ellipse[0][1])),
        axes=(int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
        angle=int(ellipse[2]),
        arcStart=0, arcEnd=360, delta=1
    )
    cv2.fillConvexPoly(canvas, pts, 1)
    return canvas

def interpolate_vein(mask3d, z_front, z_back, dilation_iters=3):
    z0, z1 = z_front - 1, z_back + 1
    n_slices = z1 - z0 + 1

    orig = mask3d.copy().astype(np.uint8)
    e0 = get_fitted_ellipse(orig[z0])
    e1 = get_fitted_ellipse(orig[z1])

    interp_stack = np.zeros((n_slices, *mask3d.shape[1:]), dtype=np.uint8)

    for i in range(n_slices):
        if e0 is None or e1 is None:
            continue
        alpha = i / (n_slices - 1)
        # 선형 보간: 중심, 축, 각도
        cx = (1-alpha)*e0[0][0] + alpha*e1[0][0]
        cy = (1-alpha)*e0[0][1] + alpha*e1[0][1]
        major = (1-alpha)*e0[1][0] + alpha*e1[1][0]
        minor = (1-alpha)*e0[1][1] + alpha*e1[1][1]
        angle = (1-alpha)*e0[2]      + alpha*e1[2]

        ellipse = ((cx, cy), (major, minor), angle)
        mask2d  = draw_ellipse_mask(orig.shape[1:], ellipse)
        dil     = binary_dilation(mask2d, iterations=dilation_iters).astype(np.uint8)

        interp_stack[i] = dil.astype(np.uint8)

    # 결과 합치기
    bridged = orig.copy()
    bridged[z0:z1+1] = interp_stack
    # 가장 큰 연결 요소만 남기기
    bridged = get_largest_component(bridged, n=1)
    return bridged

def process_vessels(label_array, z0, z1):
    artery_orig = (label_array==3).astype(np.uint8)
    vein_orig = (label_array==4).astype(np.uint8)
    
    # 혈관 라벨 체크
    artery_exists = artery_orig.any()
    vein_exists = vein_orig.any()
    renal_a = np.zeros_like(artery_orig)
    renal_v = np.zeros_like(vein_orig)
    
    try:
        total_slices = label_array.shape[0]
        if artery_exists:
            zf_a, zb_a = get_gradient_range(artery_orig, z0, z1, percentile=95)
            artery_range = zb_a - zf_a + 1 if (zf_a is not None and zb_a is not None) else total_slices
            print(f"[INFO] Artery: index=[{zf_a}-{zb_a}], gradient range={artery_range}/{total_slices} ({artery_range/total_slices*100:.1f}%)")
            
            if artery_range < total_slices * 0.5 and zf_a is not None and zb_a is not None:
                artery_bridged, _ = interpolate_circle_bridge(artery_orig, zf_a, zb_a)
                renal_a = extract_branches(artery_orig, artery_bridged, top_n=2)
            else:
                print("[WARN] Artery: gradient range exceeded - return zero array")
        else:
            print("[WARN] Artery: label not found, skipping.")
        
        if vein_exists:
            zf_v, zb_v = get_gradient_range(vein_orig, z0, z1, percentile=90)
            vein_range = zb_v - zf_v + 1 if (zf_v is not None and zb_v is not None) else total_slices
            print(f"[INFO] Vein  : index=[{zf_v}-{zb_v}], gradient range={vein_range}/{total_slices} ({vein_range/total_slices*100:.1f}%)")
            
            if vein_range < total_slices * 0.7 and zf_v is not None and zb_v is not None:
                vein_bridged = interpolate_vein(vein_orig, zf_v, zb_v)
                renal_v = extract_branches(vein_orig, vein_bridged, top_n=2)
            else:
                print("[WARN] Vein: gradient range exceeded - return zero array")
        else:
            print("[WARN] Vein: label not found, skipping.")
                
        return renal_a, renal_v
        
    except Exception as e:
        print(f"[ERROR] Vessel processing failed: {e} - return zero array")
        return renal_a, renal_v

def preprocess(img, label_array):
    # ===== Auto Branch Split =====
    kidney = (label_array==2).any(axis=(1,2))
    z0, z1 = np.where(kidney)[0][[0,-1]]
    renal_a, renal_v = process_vessels(label_array, z0, z1)

    # 혈관 마스크 생성
    vessel_mask = (label_array == 3) | (label_array == 4) | renal_a.astype(bool) | renal_v.astype(bool)

    # ===== Fat dilation (혈관 영역 제외) =====
    kidney_mask = (label_array == 2)
    fat_mask = (label_array == 6)
    tumor_mask = (label_array == 1)

    structure = np.ones((3,3,3), dtype=bool)
    dilated_kidney = binary_dilation(kidney_mask, structure=structure, iterations=2)
    kidney_boundary = dilated_kidney & ~kidney_mask
    kidney_boundary[tumor_mask] = False
    fat_mask_dilated = fat_mask | kidney_boundary

    # 원본 segmentation 위에 renal labels 덮어쓰기
    out_arr = label_array.copy()
    out_arr[fat_mask_dilated] = 6                    # fat label 추가
    out_arr[vessel_mask] = label_array[vessel_mask]  # 원본 혈관 라벨 복원
    out_arr[renal_a.astype(bool)] = 7
    out_arr[renal_v.astype(bool)] = 8
    out_img = sitk.GetImageFromArray(out_arr.astype(label_array.dtype))
    out_img.CopyInformation(img)
    return out_img

if __name__ == '__main__':
    import sys
    sys.path.append(r'C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523')
    from threeDRecon import combineGLB, processNii, processMesh

    subject_id = "S000"
    base_folder = r"C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523"
    nii_folder = os.path.join(base_folder, r'dataset\nii')
    glb_folder = os.path.join(base_folder, r'dataset\glb')
    output_folder = os.path.join(base_folder, r'dataset\output')
    threeDRecon_folder = os.path.join(base_folder, r'threeDRecon')
    nii_path = os.path.join(nii_folder, f"{subject_id}_segmentation.nii.gz")

    img = sitk.ReadImage(nii_path)
    label_array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = img.GetSpacing()[::-1]  # (X, Y, Z) → (Z, Y, X)

    pp_img = processNii.preprocess(img, label_array)
    pp_label_array = sitk.GetArrayFromImage(pp_img)
    pp_spacing = pp_img.GetSpacing()[::-1]
    print("processed")
    # Save preprocessed image for debugging
    debug_nii_path = os.path.join(r'C:\Users\USER\Documents\vscode_projects\arna_3d_smoothing\ver0523\dataset\output', f"{subject_id}_d_preprocessed.nii.gz")
    sitk.WriteImage(pp_img, debug_nii_path)
    print(f"[DEBUG] Saved preprocessed NII: {debug_nii_path}")