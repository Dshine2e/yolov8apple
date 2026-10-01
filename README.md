# ASAC — D455 RGB-D / YOLOv8 객체 탐지

Ubuntu 22.04, ROS 2 Humble, Python 3.10 기준의 ROS 2 워크스페이스입니다.
RealSense D455의 RGB 영상으로 객체를 탐지하고 RGB에 정렬된 깊이 영상에서
거리 및 카메라 좌표계의 3차원 위치를 추정합니다.

학습 데이터는 [Roboflow apple-fbyiy version 2](https://universe.roboflow.com/rita-kmex0/apple-fbyiy/dataset/2/download/yolov8)를 사용합니다.
데이터셋은 학습용 영상/라벨이며 실행용 가중치가 아닙니다. 이 데이터셋으로
YOLOv8을 학습한 `best.pt`가 필요합니다. 클래스 이름과 ID는 실제 `data.yaml` 및
학습 가중치에서 읽으며 임의로 사과 클래스 번호를 지정하지 않습니다.

## 현재 상태에서 바로 실행

Roboflow 다운로드는 사용자 요청으로 중단했습니다. 불완전한 파일은
`datasets/apple-v2-incomplete/roboflow.zip`에 보존했으며 학습에는 사용하지 않습니다.
현재 PC에는 완전히 다운로드된 `models/yolov8n.pt`와 실행용 `.venv`, 빌드 결과가 있습니다.
이 기본 COCO 모델의 `apple` 클래스만 탐지하는 실행 명령입니다.

```bash
cd ~/ASAC
bash scripts/run_detector.sh
```

이 PC에서는 기존 ROS 환경의 Jazzy를 사용합니다. Ubuntu 22.04 / Humble 장비에서는
아래 설치·빌드를 마친 뒤 `ASAC_ROS_DISTRO=humble bash scripts/run_detector.sh`로 실행합니다.
스크립트는 데이터를 다운로드하거나 학습하지 않습니다. 가중치 파일은 Git에서 제외되므로
새 clone에는 직접 준비해야 합니다. 이미 카메라 드라이버가 실행 중이면 다음처럼 실행합니다.

```bash
bash scripts/run_detector.sh models/yolov8n.pt start_camera:=false
```

이 모델은 지정 Roboflow 데이터셋으로 학습된 모델이 아닙니다. 커스텀 `best.pt`가 준비되면
`bash scripts/run_detector.sh /절대/경로/best.pt`로 교체합니다. 이때 클래스 이름이 `apple`과
다르면 `target_classes:=실제클래스이름`을 추가하거나 `target_classes:=''`로 전체 클래스를
선택합니다. 모델 이름과 맞지 않는 클래스는 시작 시 명확한 오류로 표시합니다.

## 설치와 빌드

Ubuntu 22.04에 ROS 2 Humble이 설치되어 있다는 전제입니다.
RealSense SDK/드라이버를 이미 별도 설치했다면 중복 설치 전에
[공식 RealSense 설치 안내](https://github.com/realsenseai/realsense-ros#installation-on-ubuntu)를 확인하세요.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-colcon-common-extensions \
  python3-pytest python3-yaml ros-humble-cv-bridge \
  ros-humble-message-filters ros-humble-realsense2-camera

git clone https://github.com/Dshine2e/ASAC.git
cd ASAC
source /opt/ros/humble/setup.bash

# Humble 바이너리의 Python 버전에 맞춰 시스템 Python 3.10 사용
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU 환경: 큰 CUDA 패키지를 설치하지 않는 PyTorch wheel 사용
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# venv Python으로 console script를 생성해야 ROS 실행에서도 YOLO를 import할 수 있다.
python -m colcon build --symlink-install
source install/setup.bash
```

NVIDIA GPU를 사용하려면 CPU wheel 설치 명령 대신
[PyTorch 공식 버전별 설치 명령](https://pytorch.org/get-started/previous-versions/#v251)을
사용해 `torch 2.5.1` / `torchvision 0.20.1`과 드라이버에 맞는 CUDA wheel을 설치합니다.
그 후 아래 학습/실행 명령의 `device`를 `0`으로 설정합니다. GPU 사용 가능 여부는
`python -c 'import torch; print(torch.cuda.is_available())'`로 확인합니다.

ROS Humble의 `cv_bridge`와 NumPy ABI 호환성을 유지하기 위해 NumPy를 1.26.4로
고정합니다. Python 3.12 가상환경으로 Humble Python 3.10 바이너리를 사용할 수는 없습니다.

## 지정한 Roboflow 데이터셋 학습

1. 아래 SDK 명령 또는 지정한 링크에서 **version 2 / YOLOv8** 형식으로 다운로드합니다.
   로그인/접근 권한이 필요하면 Roboflow에서 처리합니다. API key는 저장소에 넣지 않습니다.
2. ZIP을 `datasets/apple-v2`에 풀어 `datasets/apple-v2/data.yaml`,
   `train/images`, `train/labels`, `valid/images`, `valid/labels`가 있는지 확인합니다.
   클래스 수·이름·영상 개수·라이선스는 실제 export에서 확인합니다.
3. 아래 명령으로 경로를 검증한 뒤 학습합니다.

사용자가 제공한 Roboflow SDK 방식은 다음처럼 환경변수로 인증합니다.
다운로드 SDK가 설치하는 OpenCV를 ROS/YOLO 환경과 분리하기 위해 별도 venv를 사용합니다.
노트북의 `!pip install`은 셸에서는 `python -m pip install`로 실행합니다.

```bash
cd ~/ASAC
/usr/bin/python3 -m venv .venv-roboflow
.venv-roboflow/bin/python -m pip install -r requirements-roboflow.txt
read -r -s -p 'Roboflow API key: ' ROBOFLOW_API_KEY
export ROBOFLOW_API_KEY
.venv-roboflow/bin/python scripts/download_dataset.py --output datasets/apple-v2
unset ROBOFLOW_API_KEY
```

스크립트는 `rita-kmex0` → `apple-fbyiy` → version `2` → `yolov8`을 요청하며
비어 있지 않은 출력 폴더에는 덮어쓰지 않습니다. 인증 키를 코드/명령 이력/`.env` 파일에
작성하지 않습니다. SDK 오류는 인증 키를 가린 후 실패 상태로 반환합니다.

```bash
cd ~/ASAC
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run asac_perception train --data datasets/apple-v2/data.yaml --validate-only

ros2 run asac_perception train \
  --data datasets/apple-v2/data.yaml \
  --model yolov8n.pt --epochs 100 --imgsz 640 --batch 8 \
  --device cpu --workers 2 --seed 42 \
  --project runs/apple --name yolov8n_v2
```

파이프라인만 먼저 확인할 때는 별도 실험 이름에 `--epochs 1 --imgsz 320 --fraction 0.05`를
추가해 train 영상의 5%만 사용합니다. 이 가중치는 실행 확인용이며 실제 탐지 성능을 평가한
최종 모델로 취급하지 않습니다. 정식 학습은 기본 `--fraction 1.0`으로 전체 train split을 사용합니다.

`yolov8n.pt`는 첫 학습 실행 시 Ultralytics에서 다운로드됩니다. 위 설정은 시작점이며
최적 파라미터나 검증된 성능을 의미하지 않습니다. CPU 학습은 오래 걸릴 수 있으므로
CUDA GPU가 있다면 `--device 0`을 사용합니다. 기본 검증 split으로 학습 중 평가하며,
원본 export YAML은 보존하고 실행별 절대 경로 YAML을 `runs/apple`에 생성합니다.
Roboflow의 `../train/images`처럼 export 위치와 맞지 않는 경로는 실제 폴더가 존재하는지
확인한 뒤 보정합니다. 이 도구는 split별 이미지 디렉터리 형태의 YOLOv8 export를 지원합니다.

최종 가중치는 일반적으로 `runs/apple/yolov8n_v2/weights/best.pt`에 생성됩니다.
같은 실험 이름이 존재하면 Ultralytics가 새 디렉터리 이름을 선택하므로 **출력되는 실제 경로**를
사용하세요. 별도 test split이 있다면 학습 후 독립 평가합니다.

```bash
yolo detect val model=runs/apple/yolov8n_v2/weights/best.pt \
  data=/절대/경로/runs/apple/dataset_실제파일명.yaml split=test device=cpu
```

평가는 precision, recall, mAP50, mAP50-95와 가림/조명/거리별 실패 사례를 함께 기록합니다.
D455 영상은 학습 데이터와 촬영 환경이 다를 수 있으므로 실제 카메라 영상에서도 검증해야 합니다.

## 카메라와 탐지 실행

D455를 USB 3 포트에 연결하고 학습한 모델 경로를 전달합니다.

```bash
cd ~/ASAC
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 launch asac_perception d455_yolo.launch.py \
  model_path:="$PWD/runs/apple/yolov8n_v2/weights/best.pt" device:=cpu
```

launch는 RealSense 드라이버와 탐지 노드를 함께 실행합니다.
기본 스트림은 RGB/depth 각각 `640x480x30`, RGB 정렬 `align_depth.enable:=true`,
동기화 `enable_sync:=true`입니다. 실제 지원 프로파일은 장치/드라이버에서 확인하세요.
시리얼로 장치를 선택할 때는 `serial_no:=_실제시리얼번호`를 전달합니다.
드라이버가 이미 실행 중이면 `start_camera:=false`를 추가해 탐지 노드만 실행합니다.

```bash
ros2 topic echo /asac/detections
ros2 run rqt_image_view rqt_image_view /asac/annotated_image
```

영상 뷰어는 필요 시 `sudo apt install ros-humble-rqt-image-view`로 설치합니다.
디스플레이 없는 환경에서도 탐지는 동작하며 `cv2.imshow`를 사용하지 않습니다.

rosbag 테스트에서는 `start_camera:=false use_sim_time:=true`와
`ros2 bag play /경로/rosbag --clock`을 함께 사용합니다. 입력 세 토픽이 모두 기록되어야 합니다.

## 입력과 출력

| 종류 | 기본 토픽 | 메시지 |
|---|---|---|
| RGB | `/camera/camera/color/image_raw` | `sensor_msgs/Image` |
| RGB에 정렬된 depth | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| RGB 내부 파라미터 | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| 객체 탐지 | `/asac/detections` | `asac_interfaces/DetectedObjectArray` |
| 박스/깊이를 표시한 영상 | `/asac/annotated_image` | `sensor_msgs/Image` |

`camera_namespace`와 `camera_name` launch 인자를 바꾸면 입력 토픽도 함께 바뀝니다.
다른 드라이버/토픽 구조에서는 탐지 노드를 직접 실행할 수 있습니다.

```bash
ros2 run asac_perception detector --ros-args -r __ns:=/asac \
  --params-file src/asac_perception/config/detector.yaml \
  -p model_path:="$PWD/runs/apple/yolov8n_v2/weights/best.pt" \
  -p color_topic:=/camera/color/image_raw \
  -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info
```

`header`는 RGB 촬영 시각과 `camera_color_optical_frame` 등 실제 입력 optical frame을
보존합니다. optical 좌표는 오른손 좌표계이며 **X 오른쪽, Y 아래, Z 전방**입니다.
로봇 `base_link`나 `map` 좌표가 필요하면 해당 시각의 TF로 별도 변환해야 합니다.

각 객체는 클래스 ID/이름, 신뢰도, 원본 RGB pixel 기준 `[xmin,ymin,xmax,ymax]`,
깊이 유효 여부, 유효 샘플 수, 위치 `(X,Y,Z)` [m], `depth_m` [m], `range_m` [m]를 포함합니다.
`depth_valid=false`일 때 위치/거리값은 **NaN**이며 0 m로 처리하지 않습니다.
정상 추론에서 객체가 없으면 빈 배열을 발행합니다. 입력/추론 오류 또는 오래된 프레임은
결과를 발행하지 않고 로그에 표시하므로 하위 노드는 자체 timeout도 설정해야 합니다.

## 깊이 계산과 설정

ROS RealSense `16UC1` 깊이는 mm에서 m로 변환하며 `32FC1`은 m로 사용합니다.
RGB에 정렬된 동일 해상도의 depth만 허용합니다. 바운딩 박스 중심부의 가로/세로
각각 30% 영역에서 0, NaN, Inf 및 설정 범위 밖의 값을 제외하고 중앙값 `Z`를 계산합니다.
유효 샘플이 5개 미만이면 3차원 위치를 무효로 표시합니다.

왜곡 없는 핀홀 모델에서 박스 중심 `(u,v)`의 역투영은 다음과 같습니다.

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
range = sqrt(X² + Y² + Z²)
```

`fx,fy,cx,cy`는 RGB `CameraInfo.K`의 pixel 단위 값입니다. Raw RGB에 왜곡 계수가
있으면 `plumb_bob`/`rational_polynomial` 또는 `equidistant` 모델을 OpenCV로 역변환한
정규화 광선을 사용합니다. 보정되지 않은 카메라, 지원하지 않는 왜곡 모델, 다른 frame,
서로 다른 해상도 및 보정되지 않은 binning/crop은 오류로 처리합니다.

설정은 `src/asac_perception/config/detector.yaml`에 있습니다.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `confidence` / `iou` | 0.5 / 0.45 | 탐지 confidence / NMS IoU |
| `target_classes` | 빈 문자열 | 전체 클래스; `apple` 또는 쉼표로 구분한 모델 클래스 이름 |
| `image_size` | 416 | YOLO 입력 크기; 결과 박스는 원본 RGB 크기 |
| `cpu_threads` | 2 | CPU 추론 thread 수; 카메라/ROS 처리에 CPU 여유 확보 |
| `inference_rate_hz` | 5 Hz | 최대 추론 시작 주기; 달성 FPS는 연산 시간에 의존 |
| `sync_slop_sec` | 0.05 s | RGB/depth/CameraInfo 촬영 시각 허용 차이 |
| `sync_queue_size` | 5 | 입력 동기화 queue 크기 |
| `max_frame_age_sec` | 2.0 s | 추론 전·후 촬영 시각 기준 age 제한 |
| `depth_roi_fraction` | 0.3 | 박스 중앙 깊이 샘플 영역 비율 |
| `min_depth_m` / `max_depth_m` | 0.2 / 6.0 m | 소프트웨어 깊이 유효 범위; 장치 정밀도 보장 범위 아님 |
| `min_depth_samples` | 5 | 중앙값 계산에 필요한 최소 유효 샘플 수 |
| `publish_annotated_image` | true | 표시 영상 발행 여부 |

노드 시작 시 설정을 읽으며 런타임 파라미터 변경은 허용하지 않습니다. 변경 후 재시작합니다.
입력은 best-effort/keep-last-1 QoS로 구독하며, 동기화 callback과 추론 callback을
서로 다른 그룹에서 2개 executor thread로 실행합니다. 동기화된 최신 프레임 1개를
보존하여 추론이 느릴 때 과거 프레임이 계속 쌓이는 것을 제한합니다.
구독 시작 전에 모델을 워밍업하여 첫 추론의 초기화 지연을 줄입니다. Ultralytics가 첫
`predict()`에서 CPU thread 수를 다시 설정하므로 초기화 후 `cpu_threads`를 적용합니다.
현재 CPU용 기본값은 416 입력 / 2 threads / 최대 5 Hz / 결과 age 제한 2 s입니다.
입력 크기를 줄이면 작은 물체 탐지 성능이 낮아질 수 있으므로 실제 사과 영상으로 비교해야 합니다.

`Inference exceeded max_frame_age_sec; result discarded`는 촬영 후 경과 시간이 제한을
넘어 결과를 폐기했다는 뜻입니다. 로그의 `input_age`, `inference`, `processing`, `output_age`는
각각 처리 시작 시 영상 나이, 모델 추론 시간, 노드 처리 시간, 발행 직전 영상 나이이며 단위는 s입니다.
정상 처리 시에도 약 5초마다 이 수치를 출력합니다. 2 s는 허용 상한이며 목표 지연이나 제어 성능
보장값이 아닙니다. 로봇 제어에 연결할 때는 요구 지연에 맞게 낮추고 결과 timestamp를 검사하세요.

CPU 부하와 허용 지연은 launch 인자로 조정할 수 있습니다. 지정하지 않으면 YAML 값을 사용합니다.
이미 실행 중인 노드는 `Ctrl+C`로 종료한 뒤 재시작하세요.

```bash
bash scripts/run_detector.sh models/yolov8n.pt \
  image_size:=416 cpu_threads:=2 inference_rate_hz:=5.0 max_frame_age_sec:=2.0
```

드라이버 로그가 `Device USB type: 2.1`을 표시하면 USB 2 속도로 연결된 상태입니다.
USB 3 데이터 케이블과 USB 3 포트를 확인하세요. 같은 연결을 유지하면서 카메라 부하를 줄여
시험하려면 `color_profile:=640x480x15 depth_profile:=640x480x15`를 추가할 수 있습니다.

## 테스트

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
python -m colcon test --event-handlers console_direct+
python -m colcon test-result --verbose
```

단위/역투영/왜곡/깊이 이상치/데이터셋 경로 테스트와 실제 ROS 메시지·DDS를 사용하는
동기화 입력→결과 발행 테스트가 포함되어 있습니다. ROS 통합 테스트는 YOLO 모델만
결정론적 대역으로 교체하므로 가중치 다운로드와 카메라 없이 실행할 수 있습니다.
이는 학습된 모델의 정확도 또는 실제 D455의 성능 검증을 대체하지 않습니다.
GitHub Actions는 Ubuntu 22.04 / Humble 컨테이너에서 빌드와 테스트하도록 구성했습니다.
수행한 검사와 환경별 한계는 [검증 기록](docs/VERIFICATION.md)에 기록합니다.

## 구조와 한계

```text
src/asac_interfaces/       객체 탐지 메시지
scripts/download_dataset.py  환경변수 인증 Roboflow SDK 다운로드
src/asac_perception/
  asac_perception/         ROS 노드, 깊이 계산, 학습 도구
  config/                 추론/동기화/깊이 파라미터
  launch/                 D455 + detector 실행
  test/                   수치 계산과 ROS 통합 테스트
datasets/                 로컬 export (Git 제외)
runs/                     학습 결과 (Git 제외)
models/                   선택적 가중치 보관 (Git 제외)
```

현재 방법은 박스 중앙 영역의 대표 깊이를 박스 중심 광선에 배치하는 근사입니다.
이는 사과의 기하학적 중심/자세나 파지점이 아닙니다. 배경/가림/반사/깊이 결측으로
오차가 발생할 수 있으며 물체 tracking, 충돌 회피, TF 변환은 포함하지 않습니다.
실제 거리별 기준 측정과 비교하여 오차를 평가한 뒤 로봇 제어 입력으로 사용하세요.

Ultralytics는 AGPL-3.0/Enterprise 라이선스 조건을 사용하므로 배포 방식에 맞는
[라이선스](https://www.ultralytics.com/license)를 확인하고 데이터셋 라이선스도 별도 확인합니다.
구현의 API/설정 근거는 [YOLOv8 문서](https://docs.ultralytics.com/models/yolov8/),
[학습 문서](https://docs.ultralytics.com/modes/train/),
[YOLO 데이터셋 형식](https://docs.ultralytics.com/datasets/detect/),
[RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros)입니다.
