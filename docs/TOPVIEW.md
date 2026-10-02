# PiPER 고정 탑뷰 인식 파이프라인

RGB(-D) → 실제 YOLOv8-seg instance Mask → 관측 표면/근사 구 중심·반지름 → 촬영 시각 TF
→ 세션 Apple ID → RGB/Mask crop·2D 원형도·관측 방향·validity를 출력한다.
기존 detection 실행과 새 segmentation 실행은 별도 entry point이며 기존 코드는 보존했다.
결함 모델, Retinex, 착색률, 등급, MoveIt 경로, Pick and Place는 구현하지 않는다.

## 확인한 환경과 설치

현재 실행 환경: Ubuntu 22.04.5 WSL2 x86_64, ROS 2 Humble, Python 3.10.12,
Ultralytics 8.3.40, torch 2.5.1+cpu, torchvision 0.20.1+cpu,
NumPy 1.26.4, OpenCV 4.10.0, SciPy 1.11.4, PyBullet 3.2.7.
CUDA 추론 장치와 `/dev/video*` 장치가 확인되지 않았고 RealSense ROS 드라이버도 설치되어 있지 않다.
기존 README의 실제 D455 성공 이력은 이전 장비/환경에서의 기록이다.

Humble 설치가 된 새 Ubuntu 22.04 환경에서는:

```bash
sudo apt install python3-venv python3-pip python3-colcon-common-extensions \
  python3-pytest python3-scipy python3-yaml ros-humble-cv-bridge \
  ros-humble-message-filters ros-humble-tf2-ros ros-humble-robot-state-publisher ros-humble-rviz2
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-topview.txt
python scripts/download_topview_models.py
python -m colcon build --symlink-install
source install/setup.bash
```

시스템 Python 패키지나 ROS 배포판은 이전하지 않았다. 현재 작업공간에는 위 venv와
빌드 결과가 준비되어 있다. 처음에는 pip/ensurepip가 없어서 공식 get-pip를 프로젝트
venv에만 설치했고, Humble 테스트 플러그인 호환을 위해 pytest 7.4.4 /
setuptools 68.2.2를 고정했다. 새 환경에서는 위 apt/venv 방법을 권장한다.

## 한 launch로 시뮬레이션 실행

```bash
bash scripts/run_topview.sh sim models/yolov8n-seg.pt
# GUI 없이 센서·인식·TF·토픽을 실행
bash scripts/run_topview.sh sim models/yolov8n-seg.pt rviz:=false
# 평가 노드는 정답을 별도 구독한다. 결과는 모델 성공 여부와 무관하게 실제 수치를 기록한다.
bash scripts/run_topview.sh sim models/yolov8n-seg.pt rviz:=false evaluate:=true \
  evaluation_duration:=30.0 evaluation_output:=/tmp/asac-evaluation.json
```

공식 `agilexrobotics/piper_ros`의 실제 `humble` 브랜치,
commit `017ffefa64511bc6325bd77ddc4e16065c152051`의 URDF/STL을 포함했다.
URDF에서 root=`base_link`, movable joint=`joint1`~`joint8`을 확인하고 런타임에도 파싱한다.
URDF는 S-V1.6-3 이후 공식 모델이며 실제 로봇 보정이나 펌웨어 확인을 대체하지 않는다.

공식 Gazebo launch도 조사했으나 이 환경에는 Gazebo가 없고 sudo 인증이 필요했다.
현재 호환되는 최소 구성은 **PyBullet DIRECT + TinyRenderer**다. URDF 로봇과 3D 테이블,
트레이·사과 mesh를 로드하고 perspective RGB와 동일 렌더의 z-buffer를 취득한다.
Depth는 `Z=far*near/(far-(far-near)*buffer)`로 미터 광축 깊이로 복원한다.
렌더러의 객체 ID/segmentation buffer는 비활성화하며 YOLO 대신 사용하지 않는다.
로봇은 고정 자세로 둔다. 하드웨어 드라이버, CAN, enable, 컨트롤러, MoveIt는 없다.
RViz만 띄우는 구성과 달리 이 launch는 실제 3D 센서 RGB-D를 계속 발행한다.

`src/asac_sim/config/scene.yaml`에서 테이블·트레이 크기/위치, 사과 개수·위치·반지름·색,
apple/sphere 형상, 조명, 카메라 높이/회전/해상도/FOV, joint 자세, 시간별 추가·이동·제거를
설정한다. 시뮬레이션 카메라 설치는 이 파일 한 곳에서 렌더와 TF에 함께 사용한다.
YAML을 복사한 뒤 `scene_file:=/절대/경로/scene.yaml`로 전달한다.
인식 노드는 이 파일이나 사과 생성 좌표를 읽지 않는다.

## 토픽·TF와 downstream 계약

| 토픽 | 타입 / 역할 |
|---|---|
| `/asac_sim/rgb/image` | `sensor_msgs/Image`, RGB8 렌더 |
| `/asac_sim/depth/image` | `sensor_msgs/Image`, RGB 정합 32FC1 광축 Z [m] |
| `/asac_sim/rgb/camera_info` | `sensor_msgs/CameraInfo`, 같은 stamp/해상도/optical frame |
| `/asac/observations` | `asac_interfaces/AppleObservationArray`, 품질분류 입력에 필요한 crop와 기하 |
| `/asac/observations_json` | `std_msgs/String`, RFC8259 metadata, 무효값 `null` |
| `/asac/planning_input` | `std_msgs/String`, 한 capture의 ID/base 중심/radius/diameter/validity/reasons |
| `/asac/overlay` | 원본과 구분된 BGR8 시각화 |
| `/asac/markers` | 중심/근사 구/ID/방향 MarkerArray; 유효 base 측정만 표시 |
| `/asac/status` | 입력 미수신, stale, 추론 실패 등 상태 JSON |
| `/asac_sim/joint_states` | 시뮬레이션 state만 발행; 물리 command 토픽과 분리 |
| `/asac_sim/scene_markers` | 테이블/트레이/카메라 표시 |
| `/asac_sim/evaluation_truth` | 평가 전용; 인식은 구독하지 않음 |
| `/clock` | 시뮬레이터가 단독 발행 |

```text
asac_sim_world
├─ base_link → 공식 PiPER link/joint TF (robot_state_publisher)
└─ asac_sim_camera_link → asac_sim_camera_optical_frame
```

world→base와 world→camera_link→optical static TF의 발행 주체는 scene 하나다.
PiPER joint TF의 발행 주체는 robot_state_publisher 하나다. 모든 sim 노드는
`use_sim_time=true`이며 scene의 steady timer가 /clock을 구동한다.
기본 optical +Z는 아래 방향이며 +X는 영상 오른쪽, +Y는 영상 아래다.
body frame(+X 전방)과 optical frame은 REP103 변환으로 구별했다.
기본 optical→base는 translation `[0.4,0,0.85]`, xyzw `[1,0,0,0]`다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic info /asac_sim/rgb/image --verbose
ros2 topic echo /asac_sim/rgb/camera_info --once
ros2 topic echo /asac/observations_json --once
ros2 interface show asac_interfaces/msg/AppleObservation
ros2 run tf2_ros tf2_echo base_link asac_sim_camera_optical_frame
ros2 run rqt_image_view rqt_image_view /asac/overlay
```

관측 배열과 각 crop의 stamp는 RGB capture stamp다. crop는 Apple ID가 들어 있는 메시지에
함께 담아 연결한다. bbox는 원본 영상의 YOLO xyxy이며 ROI는 bbox와 Mask를 모두 포함하는
정수 `[x_offset,y_offset,width,height]`다. crop 픽셀 좌표에 ROI offset을 더하면 원본 좌표다.
각 좌표는 PointStamped이며 surface와 sphere center를 별도로 제공한다.
apple orientation/Pose를 추정하지 않는다. `camera_pose_base`는 해당 capture의 TF다.
방향은 base에서 **사과→카메라** 단위벡터 `(camera_position-center)/norm(...)`다.
중심이 무효이면 방향도 무효다. 기하·사진·Mask·방향은 품질분류에 그대로 전달할 수 있고,
planning 입력의 다른 apple 항목들이 주변 사과 위치다. 목록은 현재 보이는 사과만 포함하며
미검출/가려진 장애물이 없다는 보장은 없다. 품질분류/경로계획 consumer는 이번에 시작하지 않는다.

ROS float/point 무효값은 NaN과 validity/reasons로 표현한다. JSON은 반드시 null을 사용한다.
불확실 원형도는 `circularity_top=null`, `circularity_valid=false`이며 계산 자체의 값은
`circularity_top_raw` 진단 필드에 남는다. 부정확한 숫자를 정상 원형도로 표시하지 않는다.
빈 배열은 성공적으로 추론했으나 검출이 없는 capture다. stale/추론 오류는 정상 배열을 만들지
않고 status로 알린다. downstream은 capture stamp, validity, timeout을 함께 검사해야 한다.

## 실제 카메라 입력

```bash
# 기존 ROS 카메라 드라이버를 재사용; 기본은 드라이버 시작 안 함
bash scripts/run_topview.sh real /절대/경로/segmentation.pt \
  params_file:=/절대/경로/topview_real.yaml
# RGB-only: Depth 없이 분할·crop·ID·2D 원형도만 출력. CameraInfo도 선택적이다.
bash scripts/run_topview.sh real models/yolov8n-seg.pt rgb_only:=true \
  params_file:=/절대/경로/topview_real.yaml
```

`src/asac_perception/config/topview_real.yaml`을 복사하여 camera_id, 실제 입력 토픽,
optical frame, image_state, Depth 계약을 설정한다. 기존 D455 입력 토픽을 기본으로 재사용한다.
D455 드라이버는 `align_depth.enable=true`와 `enable_sync=true`여야 한다.
드라이버를 새로 시작할 필요가 있으면 별도 설치 후 `start_camera:=true`를 사용할 수 있다.
기존 `d455_yolo.launch.py`에서 확인한 공식 RealSense `rs_launch.py`만 include한다.

RGB-D는 RGB/Depth/CameraInfo를 approximate synchronization으로 묶으며 기본 허용차 30 ms,
queue 5, best-effort/keep-last-1, 최대 추론 시작 5 Hz, 프레임 age 상한 2 s다.
추론 중에는 최신 input 한 개만 보존한다. YAML에서 QoS·주기·slop·queue·최신성·TF timeout을
변경할 수 있다. /clock이 멈추어도 미수신 watchdog은 steady clock으로 동작한다.
RGB-only는 RGB를 직접 받아 CameraInfo 없이도 2D를 수행한다. 이 모드의 모든 미터 기하는 무효다.

Depth는 동일 RGB 픽셀에 정합되고 optical-axis Z여야 한다. 프레임/해상도가 다르면 무효다.
단순 resize 정합은 수행하지 않는다. 정합이 없는 센서는 먼저 드라이버에서 정합해야 한다.
16UC1의 스케일은 센서 계약에 맞게 `depth_scale_16uc1`을 설정한다(D455 기본 mm→m=.001).
32FC1은 m만 허용한다. rectified 영상은 CameraInfo.P의 좌측 3x3을 사용하며 stereo P의
translation 항은 지원하지 않는다. raw 영상은 K와 Brown-Conrady/fisheye 역왜곡 광선을 쓴다.
보정되지 않은 ROI/binning, 미보정 내참수, 비유한 값, 지원되지 않는 왜곡은 기하를 무효 처리한다.

실카메라 base 출력은 검증된 **eye-to-hand 보정**과 외부 TF 발행이 필요하다.
기본 `base_frame=''`, `calibration_verified=false`이며 임의 static TF를 생성하지 않는다.
보정 파일의 실제 root/optical frame과 변환 방향을 확인한 뒤 YAML의 `base_frame`,
`expected_optical_frame`, `calibration_verified=true`를 설정하고 보정의 TF 발행 노드를 별도로
실행한다. 보정은 optical→base를 표현해야 하며 설치 body→optical 회전을 중복 적용하지 않는다.
드라이버가 이미 body→optical TF를 발행한다면 외부 보정 노드는 base→카메라 root/body를
연결하고, 합성된 base→optical이 검증된 보정과 일치하도록 한다. 같은 optical child의 TF를
두 발행 주체가 중복 발행하지 않는다.
시뮬레이션 설치값을 실카메라 보정에 복사하지 않는다. tf2 조회는 항상 capture 시각을 사용하며
최신 TF로 fallback하지 않는다. 보정이나 TF가 없어도 RGB 분할과 유효 camera 기하는 남는다.
실제 카메라/보정 파일은 이번 환경에서 확보되지 않아 물리 RGB/RGB-D 검증은 미완료다.

## 모델·기하·추적 설정

```bash
.venv/bin/python scripts/inspect_model.py models/yolov8n-seg.pt
.venv/bin/python scripts/download_topview_models.py --models yolov8n-seg.pt yolov8s-seg.pt
```

task를 checkpoint에서 검사하므로 detection 가중치는 오류로 종료한다. `target_classes`는
실제 model.names에 있는 클래스 이름(여러 개면 쉼표 구분)으로 지정한다. COCO apple ID=47은
metadata에서 찾은 결과일 뿐 하드코딩하지 않았다. 자동 가중치 다운로드나 허위 Mask fallback은 없다.
Ultralytics 8.3.40의 `retina_masks=true`로 letterbox를 제거하고 원본 해상도 Mask를 얻으며
출력 shape 계약을 검사한다. bbox를 Mask로 만들지 않는다.

실루엣은 최대 연결 성분만 사용하며 폐쇄된 8 px 이하 공극만 채운다. smoothing은 없다.
같은 실루엣의 foreground 픽셀 수 A와 모든 외곽/남은 공극 윤곽의 OpenCV chain perimeter P에서
`C=4πA/P²`, 픽셀 등가 반지름 `sqrt(A/π)`을 계산한다. raster perimeter 편향으로 원에 대해서도
C가 정확히 1이 되지 않고 작은 영상에서는 1을 약간 넘을 수 있다. clamp로 성능을 좋게 만들지
않는다. 경계 잘림, 100 px 미만, 겹침 cue, solidity<.88은 원형도를 무효/불확실로 둔다.
한 탑뷰만으로 가림을 확정할 수 없으며 기본 상태는 `unknown_single_view`다.

깊이는 Mask를 2 px erosion한 내부에서 0/NaN/Inf, .15~3 m 외의 값과 MAD 이상치를 제거한다.
유효 비율의 분모는 eroded Mask의 전체 픽셀 수다. 최소 80점/유효비율 .3을 요구한다.
surface는 점군의 좌표별 median에 가장 가까운 실제 관측 점이다.
최대 2,000점으로 4점 RANSAC(80회)+soft-L1 sphere fit을 한다. 최소 inlier비율 .7,
inlier threshold 3 mm, RMSE≤3 mm, radius .015~.09 m, Jacobian condition≤100,
cap height/diameter≥.12와 중심이 관측 표면 뒤에 있다는 조건을 검사한다.
구 피팅이 실패하면 center/radius/diameter는 null이며 surface를 중심으로 대체하지 않는다.
반지름은 사과의 **근사 구 형상** 값이다. apple mesh의 꼭짓점 radius와 다를 수 있다.
실제 형태/깊이 노이즈/부분 가림은 별도 오차를 발생시킨다. 단일 표면으로 plausible 단위 배율
오류를 항상 알아낼 수는 없다. 센서 단위 계약 확인과 허용 범위 설정이 필요하다.
필터·피팅 실제 thresholds도 `measurement_config`에 남는다.

Tracker는 Hungarian association을 사용한다. 같은 optical frame에서 양쪽 구 중심이 유효하면
3D 거리 gate .06 m와 IoU cost를, 3D가 무효이면 IoU gate .1을 사용한다. 중복 bbox IoU≥.85는
높은 confidence만 남긴다. ID는 임의 세션값+증가 counter이며 누락 트랙은 2 s 유지한다.
같은 frame의 재검출은 gate를 통과하면 ID를 유지하고, TTL 종료 후에는 새 ID다.
frame 변경/시간 역행 시 association을 비우되 counter는 재사용하지 않는다.
완전 가림, 교차, 비슷한 사과, 장시간 누락 후 재식별을 보장하지 않는다.
모든 파라미터는 시작 시 읽고 재시작 시 적용한다.

## 저장·재현·검증

`save_directory`를 설정하면 실행마다 별도 run 폴더에 RGB crop, binary Mask PNG, JSON을
최소 10 s 간격/최대 50 batch로 저장한다. 기본은 저장 비활성화다. `save_interval_sec`,
`save_max_batches`로 주기·양을 제한한다. 실행 간 폴더는 보존하므로 전체 저장 공간 관리는 별도다.

```bash
ros2 service call /asac/snapshot std_srvs/srv/Trigger '{}'
ros2 bag record --use-sim-time -o runs/topview/bag /asac_sim/rgb/image /asac_sim/depth/image \
  /asac_sim/rgb/camera_info /tf /tf_static /asac/observations /asac/observations_json
```

bag 재생은 scene을 동시에 실행하지 않고 아래처럼 단일 /clock을 사용한다.

```bash
ros2 run asac_perception topview --ros-args -r __ns:=/asac \
  --params-file src/asac_perception/config/topview_real.yaml \
  -p model_path:="$PWD/models/yolov8n-seg.pt" -p use_sim_time:=true -p input_mode:=sim \
  -p color_topic:=/asac_sim/rgb/image -p depth_topic:=/asac_sim/depth/image \
  -p camera_info_topic:=/asac_sim/rgb/camera_info -p image_state:=rectified \
  -p camera_id:=sim_top -p expected_optical_frame:=asac_sim_camera_optical_frame \
  -p base_frame:=base_link -p calibration_verified:=true
ros2 bag play runs/topview/bag --clock
```

```bash
source .venv/bin/activate
source install/setup.bash
python -m colcon test --event-handlers console_direct+
python -m colcon test-result --verbose
# ROS_DOMAIN_ID는 기존 실행과 겹치지 않게 선택한다.
ROS_DOMAIN_ID=73 python scripts/verify_topview_e2e.py --model models/yolov8n-seg.pt --seconds 10
# 단일 사과는 small 모델로 별도 시험; nano 결과와 합쳐 성공률을 만들지 않는다.
ROS_DOMAIN_ID=74 python scripts/verify_topview_e2e.py --model models/yolov8s-seg.pt \
  --cases single_apple --output runs/topview/e2e-small
```

E2E runner는 실제 launch, 실제 모델·rendered RGB-D, DDS crop/ROI, stamp TF, overlay/markers를
검사하고 모든 scene YAML·launch 로그·원본 crop/Mask·샘플 JSON·평가를 저장한다.
허용오차는 시험 전 고정하며 구/사과 mesh를 별도로 평가한다. 합성 단위 테스트나 분할 대역
ROS 테스트는 학습 모델의 E2E 성공으로 세지 않는다. 실패를 포함한 결과는
[검증 기록](TOPVIEW_VERIFICATION.md)에 있다. CI는 unit/sensor contract용이며 실제 모델을
다운로드하지 않는다. 원격 CI 실행 결과는 아직 확인하지 않았다.

현재 전체 nano scenario 검증은 단일 사과·구 검출 실패로 종료코드 1을 반환한다.
통과한 기본 경로만 재현하려면 `--cases baseline`을 사용한다. 실패 case는 summary에 남는다.

구현 구조: `segmentation.py`, `geometry.py`, `observations.py`, `tracking.py`, `snapshots.py`를
단일 `topview_node.py`에서 연결하고, `asac_sim`이 입력/TF와 평가 전용 truth를 생성한다.
`evaluate_node.py`만 정답을 비교한다. `scripts/verify_topview_e2e.py`는 독립 시험 도구다.

## 출처

- [PiPER 공식 Humble 코드](https://github.com/agilexrobotics/piper_ros/tree/humble), MIT;
  정확한 commit와 라이선스는 `src/asac_sim/assets/piper/SOURCE.md`/`LICENSE`.
- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/),
  [segmentation](https://docs.ultralytics.com/tasks/segment/),
  [8.3.40 segmentation predictor](https://github.com/ultralytics/ultralytics/blob/v8.3.40/ultralytics/models/yolo/segment/predict.py).
  checkpoint는 공식 assets v8.3.0; downloader에서 SHA256 검증. AGPL-3.0/Enterprise 조건.
- [Bullet 공식 저장소](https://github.com/bulletphysics/bullet3), zlib;
  [getCameraImage 공식 예제](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py).
- [ROS CameraInfo 필드](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html),
  [Humble tf2 Buffer 소스](https://github.com/ros2/geometry2/blob/humble/tf2_ros_py/tf2_ros/buffer.py).
  실제 설치된 Humble signature/source로 capture-time API도 확인했다.
- 테이블/트레이 primitive와 사과 OBJ mesh는 이 구현에서 생성한 자산(패키지 Apache-2.0).
  외부 texture나 상용 사과 mesh는 사용하지 않았다.
