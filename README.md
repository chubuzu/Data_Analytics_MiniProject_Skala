# MIT-Stanford ESS 배터리 데이터 분석 미니 프로젝트

배터리 데이터셋의 MATLAB 파일(`.mat`) 구조를 확인하고, 셀 단위 데이터를 정리한 뒤 탐색적 분석(EDA), 특성 생성, 모델링으로 이어가는 프로젝트입니다.

## 현재 진행 상태

| 단계 | 파일 | 상태 |
| --- | --- | --- |
| 데이터 구조 확인 및 셀 단위 추출 | `notebooks/01_data_check.ipynb` | Batch1 완료 |
| 탐색적 분석(EDA) | `notebooks/02_eda.ipynb` | 시작됨: 셀 단위 CSV 불러오기 |
| 특성 생성 | `notebooks/03_feature_engineering.ipynb` | 미작성 |
| 모델링 | `notebooks/04_modeling.ipynb` | 미작성 |

현재 노트북은 `Batch1.mat`을 읽어 셀 46개의 `cell_id`, `channel`, `policy`, `cycle_life`를 `data/processed/batch1_cells.csv`에 저장합니다. Batch2, Batch3, `extra.mat`은 파일 존재 여부만 확인하며 분석에 사용하지 않습니다.

## 시작하기

1. 프로젝트 루트의 `data/`에 `Batch1.mat`, `Batch2.mat`, `Batch3.mat`, `extra.mat`을 둡니다. 원본 `.mat` 파일은 Git 추적 대상에서 제외됩니다.
2. 프로젝트 루트에서 가상환경을 만들고 패키지를 설치합니다.

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. `notebooks/` 디렉터리에서 Jupyter를 실행하고 `01_data_check.ipynb`를 위 가상환경의 커널로 열어 셀을 순서대로 실행합니다. 노트북의 상대 경로가 `../data`를 가리키므로 실행 위치가 중요합니다.

   ```bash
   cd notebooks
   ../.venv/bin/python -m jupyter notebook
   ```

## 01 노트북에서 하는 일

1. 실행 중인 Python과 원본 데이터 파일 경로를 확인합니다.
2. `Batch1.mat`의 최상위 키와 `batch` 필드 구조를 살펴봅니다. 이 파일은 MATLAB v7.3 형식이므로 `scipy.io.loadmat` 대신 `h5py`로 읽습니다.
3. HDF5 객체 참조를 역참조해 셀별 `barcode`, `channel_id`, `policy`, `policy_readable`, `cycle_life` 값을 추출합니다.
4. `channel_id`의 여섯 요소 중 Batch1에서 셀마다 달라지는 다섯 번째 요소(`channel_4`)를 `channel`로 사용합니다. 이는 현재 Batch1 관찰 결과이며 다른 배치에 그대로 적용된다고 가정하지 않습니다.
5. EDA에 필요한 네 컬럼만 남겨 `data/processed/batch1_cells.csv`로 저장합니다.

`cell_id`는 Batch1 내부의 1부터 시작하는 행 번호입니다. `policy`는 사람이 읽기 쉬운 충전 정책 문자열(`policy_readable`)이고, `cycle_life`는 사이클 수입니다. 생성된 CSV는 후속 분석의 입력으로 사용할 수 있습니다.

## 폴더 구성

```text
data/          원본 .mat 및 processed/batch1_cells.csv
notebooks/     단계별 분석 노트북
requirements.txt
README.md
```

Batch1의 `summary`, `cycles`, `Vdlin` 구조도 확인해 cycle 요약 CSV, profile NPZ, 전압축 CSV를 저장했습니다. 다음 단계에서는 이 데이터를 활용해 cycle-level 열화 패턴을 시각화할 예정입니다.
