# 구현 검증 기록

검증일: 2026-10-01. 대상 환경은 Ubuntu 22.04 / ROS 2 Humble / Python 3.10이며,
현재 실행 환경은 Ubuntu 24.04.5 / ROS 2 Jazzy / Python 3.12.3입니다.
Humble 컨테이너용 GitHub Actions 설정은 추가했으나 원격 실행 결과는 아직 없습니다.
로컬 Docker는 socket 접근 권한이 없어 Humble 컨테이너 검증에 사용하지 않았습니다.

## 빌드와 자동 테스트

- `python -m colcon build --symlink-install`: 두 패키지 빌드 성공.
- `python -m colcon test` / `python -m colcon test-result --verbose`: 20개 통과.
- 깊이 단위, 광축/직선거리 구분, 역투영 부호, 왜곡 역변환, 이상치·결측값,
  export 경로 처리, 인증 키 로그 차단, 기존 export 보존을 확인했습니다.
- ROS 테스트는 실제 DDS로 세 입력 메시지를 보내 동기화, 클래스/깊이/영상 출력,
  촬영 시각/frame 보존 및 깊이 결측 표시를 확인했습니다. 이 테스트에서 YOLO만 대역입니다.
- 지연된 추론이 제한을 넘으면 결과를 폐기하고, 제한 안이면 원본 촬영 시각으로 발행하는
  회귀 테스트 2개를 포함합니다.
- `flake8 --config src/asac_perception/setup.cfg src/asac_perception scripts`,
  Python 구문 컴파일, `git diff --check`: 통과.

## 실제 YOLOv8 추론

검증 라이브러리: Ultralytics 8.3.40, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu,
NumPy 1.26.4, OpenCV 4.10.0.

Ultralytics 공식 `yolov8n.pt` 및 공식 `bus.jpg`를 사용해 실제 모델로 추론하고
노드의 결과/표시 영상이 DDS로 수신되는지 검사했습니다. 버스 1개, 사람 3개가 출력됐으며,
검증용으로 만든 2.0 m 균일 깊이 영상에 대해 모든 객체의 `depth_valid=true`,
`depth_m=2.0`과 입력 header 보존을 확인했습니다.
측정한 처리/수신 시간은 약 1.545 s로, 워밍업/실행 조건을 포함한 1회 관측값입니다.
이 값으로 지속 FPS나 D455 실측 거리 정확도를 주장하지 않습니다.

실제 모델을 전달한 `ros2 launch ... start_camera:=false`에서 노드 초기화와
입력 미수신 경고도 확인했습니다. 이 테스트는 카메라 연결을 검증하지 않습니다.
현재 실행 스크립트는 기본 COCO 모델의 이름 `apple`을 모델 metadata에서 찾아 선택합니다.
실제 가중치에서 선택되는 ID는 47이었으며, 이 필터를 전달한 실제 추론도 실행했습니다.
ID를 코드에 하드코딩하지 않으며 커스텀 모델에서는 해당 모델의 클래스 이름을 사용합니다.
`models/yolov8n.pt`와 `datasets/smoke/bus.jpg`는 로컬 검사 자산이며 Git에서 제외됩니다.

검사 자산 출처:

- [YOLOv8n 가중치](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)
- [Ultralytics v8.3.40 bus.jpg](https://github.com/ultralytics/ultralytics/blob/v8.3.40/ultralytics/assets/bus.jpg)

## 환경에서 확인한 문제

사용자 전역 Python의 NumPy 2 / OpenCV 5가 ROS `cv_bridge`와 충돌했습니다.
프로젝트 venv의 NumPy 1.26.4 / OpenCV 4.10으로 빌드·테스트·실제 추론이 통과했습니다.
전역 패키지는 변경하지 않았습니다. 다운로드용 SDK는 별도 venv에 설치하여 OpenCV
headless 의존성과 ROS 영상 처리 환경을 분리했습니다. 다운로드 venv의 `pip check`는 통과했습니다.

YOLO venv는 ROS 패키지를 읽기 위해 시스템 패키지를 상속합니다. 전체 `pip check`는
상속된 `pynacl`의 `cffi` 누락을 보고했으며, 이는 위 객체 탐지 검사 경로에서 사용하지 않은
기존 패키지입니다. 시스템 패키지 수정을 수행하지 않았습니다.

## Roboflow 다운로드 상태

제공된 인증으로 version 2 YOLOv8 export 다운로드가 시작됐으나, 사용자 요청에 따라
전송 중단했습니다. 약 1.4 GiB의 불완전한 ZIP을 `datasets/apple-v2-incomplete/roboflow.zip`에
보존했습니다. 다운로드 프로세스는 종료했으며 해당 파일을 추출하거나 학습에 사용하지 않았습니다.
따라서 실제 데이터셋의 클래스 목록·분할 개수·라이선스 및 커스텀 학습 결과는 아직 확인하지
않았습니다. 현재 실행에 사용하는 모델은 완전히 받아 둔 COCO `yolov8n.pt`입니다.
API 키를 소스/문서/저장소 파일에 기록하지 않았습니다.

## 남은 검증

깊이의 실측 오차, 조명/가림별 탐지 정확도, 장시간 FPS와 지연, USB 3 연결,
GPU 실행, 실제 Humble 환경은 아직 검증하지 않았습니다.

## D455 실시간 결과 폐기 문제 수정

D455를 연결한 후 기존 설정에서 `Inference exceeded max_frame_age_sec` 경고를 재현했습니다.
RGB와 정렬 depth는 640×480 / 30 Hz, 드라이버는 RealSense ROS 4.58.4 /
librealsense 2.58.4였습니다. 장치는 USB 2.1 연결로 보고됐습니다.

기존 설정은 입력 크기 640, 최대 추론 주기 10 Hz, 촬영 시각 기준 결과 age 제한 0.5 s였습니다.
실시간 로그에서 입력 age는 약 0.06 s였지만 첫 추론은 2.484 s, 이후에도 간헐적으로
1.332 s가 걸렸습니다. 결과 age가 제한을 넘어 발행되지 않는 것이 확인됐습니다.
단독 정지영상 추론은 빠르더라도 실시간 카메라/ROS와 함께 실행하면 지연이 달라질 수 있습니다.

변경 사항:

- 영상 구독 전에 모델을 워밍업합니다.
- 첫 `predict()`의 backend 초기화가 끝난 뒤 CPU thread 수를 2로 적용합니다.
  설치된 Ultralytics 8.3.40은 첫 CPU backend 설정에서 thread 수를 다시 설정하므로
  모델 로드 전에만 제한하면 유지되지 않습니다.
- CPU 기본 입력 크기는 416, 최대 추론 주기는 5 Hz, 결과 age 상한은 2 s로 조정했습니다.
- 입력 age, 추론 시간, 발행 전 처리 시간, 출력 age를 로그로 확인할 수 있습니다.
- 오래된 입력/결과 폐기와 원본 RGB timestamp 보존은 유지합니다.
- `image_size`, `cpu_threads`, `inference_rate_hz`, `max_frame_age_sec`를 launch 인자로
  지정할 수 있으며 생략하면 YAML 값을 유지합니다.

수정한 기본 설정으로 실제 D455를 약 40초 실행했습니다. 별도 ROS 구독자로 약 30초 관찰한 결과:

| 항목 | 관측값 |
|---|---|
| 탐지 메시지 수신 | 96개 |
| 표시 영상 수신 | 98개 |
| 탐지 메시지 수신 주기 | 약 3.36 Hz |
| 촬영→탐지 메시지 수신 지연 중앙값 | 0.168 s |
| 촬영→탐지 메시지 수신 지연 최댓값 | 0.941 s |
| 촬영 timestamp | 수신 순서대로 증가 |
| 시간 초과 결과 폐기 경고 | 이 실행에서는 관측되지 않음 |

현재 촬영 장면의 `apple` 탐지 결과는 빈 배열이었습니다. 따라서 이 검사는 카메라 입력,
실제 추론, 결과/영상 발행과 지연을 확인한 것이며 사과 탐지 정확도나 거리 정확도 검증은 아닙니다.
최대 5 Hz는 설정 상한이고 위 관측 FPS는 보장값이 아닙니다. 입력 크기 축소에 따른 작은
사과 탐지 성능 변화와 2 s 허용 상한의 제어 적합성은 별도로 평가해야 합니다.
