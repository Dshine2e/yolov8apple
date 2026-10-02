# 기존 Roboflow 데이터셋 학습 및 적용

2026-10-02 현재 코드·의존성 설치·빌드와 단위 검증을 완료했다.
사용자가 제공한 `src/asac_perception/resource/Apple.v2i.yolov8.zip`의 CRC 검사 후
`datasets/apple-v2`에 압축을 해제했다. train 1,102 / validation 366 / test 366장과
7개 detect 클래스를 확인했으며 전체 이미지 파일 무결성 검사도 통과했다.
최초 API 인증 누락 차단은 로컬 ZIP으로 해결되어 재다운로드·API 키는 필요 없다.
사용자는 **다른 GPU 환경에서 학습 후 best.pt를 전달**하는 방식을 선택했다.
[외부 환경용 학습 묶음과 명령](EXTERNAL_TRAINING.md)을 따른다.
이후 전달받은 `models/imported/apple-v2/best.pt`와 학습 기록을 확인하고 로컬 test 평가를
완료했으며 기존 detector의 기본 모델로 선택했다. 실제 결과는
[받은 모델 검증 기록](IMPORTED_MODEL_VERIFICATION.md)을 따른다.

## 데이터셋과 환경

기존 README와 다운로드 코드에 있는 `rita-kmex0/apple-fbyiy`, version **2**, export
format **yolov8**을 유지한다. [공개 프로젝트](https://universe.roboflow.com/rita-kmex0/apple-fbyiy)는
Object Detection, 원본 프로젝트 영상 1,834개, CC BY 4.0, 작성자 Rita로 표시된다.
표시된 클래스는 `Calyx`, `Disease`, `Intact`, `Miscolor`, `Misshape`, `Scar`, `Stem`이다.
이는 공개 프로젝트 정보이며 v2 export의 실제 증강 영상 수·split·클래스 ID는 다운로드 후
`data.yaml`과 라벨에서 다시 확인한다. 이 사이트의 기존 모델 mAP는 새로 학습할 모델의 성능이 아니다.

현재 환경은 Ubuntu 22.04.5 / WSL2, ROS 2 Humble, Python 3.10.12,
Ultralytics 8.3.40, PyTorch 2.5.1+cpu, NumPy 1.26.4, OpenCV 4.10.0이다.
GPU는 Windows 조회 결과 AMD Radeon RX 9060 XT이며 현재 WSL에 ROCm 학습 런타임이 없다.
PyTorch는 CPU 전용이다. 다운로드 SDK Roboflow 1.5.1은
`.venv-roboflow`에 설치되어 ROS/YOLO의 OpenCV와 분리된다.
설치 버전은 `requirements.txt`, `requirements-roboflow.txt`를 따른다.

이 export의 bbox를 분할 Mask로 변환하지 않는다. 새 detect `best.pt`는 기존 detector로만
적용하며, `scripts/run_topview.sh`의 YOLOv8-seg 모델은 별도로 유지한다.
결함/부위 클래스 검출을 사과 전체 instance나 최종 품질 등급으로 해석하지 않는다.

## 인증 또는 기존 ZIP 준비

아래 방식 중 하나면 된다. 기존 `.env`가 있다면 덮어쓰지 말고 필요한 값만 추가한다.
파일은 `.gitignore`에 포함된다. 키와 signed URL은 채팅·커밋·셸 명령 이력에 넣지 않는다.

- 환경변수: `read -r -s -p 'Roboflow API key: ' ROBOFLOW_API_KEY; export ROBOFLOW_API_KEY`
- 프로젝트 루트의 `.env`에 `ROBOFLOW_API_KEY=실제키`를 에디터로 저장하고 `chmod 600 .env` 실행.
- API 키 대신 `.env`에 `ROBOFLOW_DATASET_URL='https://...실제-ZIP-export-URL...'` 저장.
- 이미 받은 ZIP은 아래 명령으로 검사·압축 해제한다. 원본 ZIP은 보존된다.

```bash
cd /home/dshine/yolov8apple
.venv-roboflow/bin/python scripts/download_dataset.py \
  --zip /절대/경로/roboflow.zip --output datasets/apple-v2
```

인증키 위치는 [Roboflow 공식 안내](https://docs.roboflow.com/reference/authentication/authentication/find-your-roboflow-api-key.md),
ZIP export는 [다운로드 안내](https://docs.roboflow.com/datasets/create-and-upload/download-a-dataset.md)를 참고한다.
웹페이지 URL과 signed ZIP 다운로드 URL은 다르다.

다운로더는 SDK 출력을 직접 노출하지 않고, URL/키를 오류에서 가린다.
별도 staging 디렉터리에서 다운로드 후 `data.yaml`을 확인해야 최종 폴더로 옮긴다.
직접 제공된 ZIP은 CRC와 경로 탈출·심볼릭 링크도 검사한다. 실패한 staging은
오류에 표시되는 `datasets/.apple-v2-download-*`에 남고 학습 폴더로 승격되지 않는다.
비어 있지 않은 기존 최종 디렉터리는 덮어쓰지 않는다.

## 다운로드 → 학습 → 평가 → 기본 모델 선택

현재 작업공간에서는 필요한 두 venv와 기존 pretrained 모델이 이미 준비되어 있다.
현재 사용자는 외부 GPU 학습을 선택했다. 아래는 이후 로컬 CPU 학습을 선택할 때의 명령이다.
지금은 이미 완전한 export가 있으므로 다운로드를 생략한다.

```bash
cd /home/dshine/yolov8apple
bash scripts/train_roboflow.sh
```

기본은 YOLOv8n, **전체 train split**, 100 epochs, imgsz 640, batch 8, CPU,
workers 2, seed 42이다. CPU에서 전체 학습은 오래 걸릴 수 있다. 이 기본 설정은
시작점이며 성능 보장이 아니다. 마지막 인자로 학습 설정을 변경할 수 있다.

```bash
bash scripts/train_roboflow.sh --epochs 100 --imgsz 640 --batch 4 --workers 2
# CUDA가 설치·확인된 다른 환경에서만 --device 0 지정
```

완전한 export의 `data.yaml`이 있으면 재다운로드를 생략한다. 라벨의 클래스 ID, 숫자 유효성,
정규화 좌표, bbox 크기, polygon 면적, 실제 detect/segment 형식과 split별 객체 수를 검사한다.
라벨 파일 누락은 YOLO 계약에 따라 background로 집계하고 수를 명시한다.
box와 polygon의 혼용, 모델/라벨 task 불일치, train/val의 객체 라벨 부재는 학습 전에 거부한다.
실제 이미지 디코딩/손상 여부는 Ultralytics 로더에서도 검사한다.

출력은 `runs/apple/yolov8n_roboflow_v2*/weights/best.pt`이다. 같은 실험 이름이 있으면
새 디렉터리를 사용한다. 종료 시 **실제 경로**를 출력한다. 원본 data.yaml은 보존하고 별도
절대 경로 YAML을 만든다. best.pt를 다시 로드해 별도 test split이 있으면 test로,
없거나 test 객체가 없으면 val로 평가한다. 후자의 결과는 독립 test 평가가 아니다.
`training_report.json`에는 actual model.names, 라벨 수, 실행 환경, 실제 완료 epoch 수,
평가 split, precision/recall/mAP 등을 기록하며 비유한 metric은 JSON null로 쓴다.

학습·평가가 성공적으로 반환된 다음에만 `models/active_detector.json`에 best.pt 절대 경로와
SHA256을 기록한다. 이는 런타임 선택 파일이며 가중치 원본은 학습 디렉터리에 남는다.
기존 선택 파일은 timestamp 백업을 남긴다. 1 epoch 확인용 실행과 일부 데이터만 사용한
학습은 기본 모델로 적용하지 않는다. 2 epoch 이상이라는 제한은 smoke 방지 규칙일 뿐
충분한 학습/정확도 기준이 아니다. 완료 후 실제 metric과 현장 영상을 확인해야 한다.

## 실행과 모델 보존

학습·평가·선택이 끝난 뒤 기존 검출 명령이 선택된 best.pt를 읽는다.

```bash
bash scripts/run_detector.sh
# 카메라 드라이버가 이미 실행 중인 경우
bash scripts/run_detector.sh start_camera:=false image_size:=640
```

자동 선택 시 모든 실제 학습 클래스 이름을 그대로 표시한다. `apple`이라는 가상 클래스를
강제로 선택하지 않는다. 명시적으로 일부 클래스만 보려면 `target_classes:=Intact` 등 실제
model.names의 이름을 지정한다. 모델 선택 파일이 없으면 이전 기본 COCO 모델과 apple 필터를
사용한다. 선택된 가중치가 없어지거나 hash가 바뀌면 묵시적으로 다른 모델로 전환하지 않고 실패한다.

기존 명시적 모델 실행과 탑뷰 분할 실행은 계속 사용할 수 있다.

```bash
bash scripts/run_detector.sh models/yolov8n.pt
bash scripts/run_topview.sh sim models/yolov8n-seg.pt
```

## 확인된 검증과 미완료 시험

라벨 형식·혼용·비유한 값·클래스 번호 오류, subset 자동 적용 거부, 잘못된 ZIP 경로,
인증 로그 가림, `.env`를 실행하지 않고 읽기, 원본 ZIP 보존, 선택된 가중치 hash 및
기존 선택 백업에 대한 테스트를 실행했다. 합성 fixture만 사용한 이 시험은 실제 모델 학습이나
Roboflow 사과 영상 추론 성공을 의미하지 않는다.

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/local_setup.bash
python -m colcon build --symlink-install --packages-select asac_perception
python -m colcon test --event-handlers console_direct+
python -m colcon test-result --verbose
```

현재 재실행 결과: **67 tests, 0 errors, 0 failures, 0 skipped**.
시스템 `/usr/bin/colcon`을 바로 호출하면 venv의 PyBullet을 찾지 못할 수 있어,
위 명령은 활성화된 venv의 Python으로 colcon을 실행한다.

외부에서 학습한 best.pt를 받은 후 해야 할 실제 검증: task/클래스/학습 기록 확인 →
best.pt test 평가 → 실제 test 이미지의 검출 사례 확인 → 선택된 모델 ROS 실행.
물리 카메라가 연결되지 않은 상태에서는 카메라 검출 성공을 주장하지 않는다.
이 명령들은 PiPER CAN/enable/하드웨어 제어를 시작하지 않는다.
