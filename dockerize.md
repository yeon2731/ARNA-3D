# ARNA 3D 로직 재구성

이 레포는 ARNA Viewer에서 .nii.gz를 .glb로 변환하는 로직을 담고 있습니다.
도커간 gRPC통신을 통해서 case_id_path를 받아와 mask에 접근할 수 있습니다.
input - 신장, 종양, 혈관, 요관이 모두 포함된 최종 mask 하나, raw kidney mask하나
output - main.py에서 저장 후 저장경로 리턴

## 1. 폴더구조

```
case_0d2a9959-b9fc-48de-b006-769609ba9ab2/
├── 3d/
│   └── obj_D.glb
├── image/
│   ├── image_A.nii.gz
│   ├── image_D.nii.gz
│   ├── image_N.nii.gz
│   └── image_P.nii.gz
├── mask/
│   ├── segment_A.nii.gz
│   ├── segment_D.nii.gz
│   ├── segment_N.nii.gz
│   ├── segment_P.nii.gz
│   ├── total_A.nii.gz
│   ├── total_D.nii.gz
│   ├── total_N.nii.gz
│   └── total_P.nii.gz
└── small_size_image/
    ├── image_A.nii.gz
    ├── image_D.nii.gz
    ├── image_N.nii.gz
    └── image_P.nii.gz
```

## 3. 함수

신장은 totalseg의 2랑 24, 3이랑 23 (신장-cist쌍)

## 2. 함수

main(case_id_path):
케이스 경로를 입력으로 받아 완성된 glb파일을 저장합니다.

추가로 지방층 작업시에

- 1픽셀 패딩
- 내부 지방 제거
