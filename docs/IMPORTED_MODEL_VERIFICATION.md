# 전달받은 best.pt 검증 및 적용

2026-10-02에 `models/imported/apple-v2`의 best.pt, training_report.json,
results.csv, args.yaml, delivery_receipt.json을 확인했다. 네 원본 파일의 SHA256은
delivery_receipt.json과 일치하고 best.pt hash는 training_report.json과도 일치한다.
수신 파일을 수정하거나 재학습하지 않았다.

## 모델 및 학습 기록

- 모델: YOLOv8n, **detect**, 약 6.0 MiB, parameter 수 3,012,213.
- 클래스: Calyx / Disease / Intact / Miscolor / Misshape / Scar / Stem.
- 요청 100 epoch, 실제 완료 **71 epoch**. CSV의 1~71행과 checkpoint 내부 history가 일치.
- 최고 validation fitness는 epoch **51**, patience 20과 이후 20행 기록이 일치한다.
  이는 조기 종료 설정과 일관되며 제공되지 않은 원본 GPU 콘솔 로그까지 확인한 것은 아니다.
- 전체 train split fraction 1.0, imgsz 640, batch 8, seed 42, amp false.
- 전달된 GPU 환경 기록: Windows 11 / RX 9060 XT / Python 3.12.14 /
  PyTorch 2.12.0+rocm7.14.1 / HIP 7.14.60850 / Ultralytics 8.3.40.
- 수신 환경: Ubuntu 22.04 WSL2 / ROS 2 Humble / Python 3.10.12 /
  PyTorch 2.5.1+cpu / Ultralytics 8.3.40. 이 환경에서 로드와 실제 추론을 확인.

best.pt SHA256:

```text
5aadadabb7f5b60ccd9797a89691f3d916c499b0584d1b7ebcf928636b637717
```

## 로컬 test 366장 재평가

로컬 실제 data.yaml과 라벨/클래스 수를 전달 보고서와 대조했다.
train 1,102 / validation 366 / test 366, test 객체 1,163개가 일치한다.
Windows의 학습 경로를 재사용하지 않고 로컬 절대 경로 YAML을 생성해 평가했다.
`scripts/verify_imported_model.py --apply`를 실행했으며 CPU / imgsz 640 / batch 4 / workers 0이다.

| Metric | 로컬 재평가 | 전달 보고서 |
|---|---:|---:|
| Precision | 0.469043 | 0.467981 |
| Recall | 0.581794 | 0.581794 |
| mAP50 | 0.500384 | 0.499679 |
| mAP50–95 | 0.315757 | 0.315211 |

mAP50 차이는 약 0.000706이다. CPU/batch/runtime이 다르므로 동일 숫자를 강제하지 않았다.
평가기가 측정한 추론 시간은 **44.55 ms/image**이며 batch 4의 평가 결과다.
이는 카메라 취득/전송/이미지 디코딩/ROS 처리까지 포함한 실시간 end-to-end 주기가 아니다.
전체 test loop는 약 41초였다.

| 클래스 | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Calyx | .771 | .922 | .899 | .489 |
| Disease | .461 | .334 | .335 | .163 |
| Intact | .388 | .690 | .568 | .560 |
| Miscolor | .484 | .513 | .461 | .190 |
| Misshape | .458 | .917 | .606 | .604 |
| Scar | .086 | .020 | .041 | .011 |
| Stem | .635 | .677 | .592 | .193 |

Precision/recall은 평가기가 고른 운영점의 통계이며 런타임 confidence 0.5에서의 통계와
동일하다고 주장하지 않는다. 특히 **Scar와 Disease의 재현율이 낮다**.
파일 호환/실행 성공이 모든 결함의 신뢰할 만한 검출을 의미하지 않는다.

JSON 원본은 `docs/verification/imported-model.json`에 복사했다.
큰 평가 산출물은 `runs/import-verification/apple-v2-20261002T053304503224`에 있으며,
test 결과/PR/혼동행렬/예측 및 실제 test RGB 샘플 overlay 3장을 포함한다.

## 기본 모델 선택과 ROS 실행

`models/active_detector.json`은 전달받은 best.pt와 인접 training_report.json을 가리킨다.
기존 COCO 가중치와 탑뷰 segmentation 가중치는 보존한다. 전체 클래스를 선택할 때
ROS launch의 빈 `target_classes:=` 인자를 생략하도록 실행 스크립트를 수정했다.
첫 인자에 model.pt 대신 launch 인자를 전달하는 실행도 지원한다.

```bash
cd /home/dshine/yolov8apple
# 카메라 드라이버가 이미 실행 중이거나 외부 입력을 사용할 때
bash scripts/run_detector.sh start_camera:=false image_size:=640
# 특정 클래스만 볼 때
bash scripts/run_detector.sh start_camera:=false image_size:=640 target_classes:=Intact
# 명시적으로 이전 COCO 모델 선택
bash scripts/run_detector.sh models/yolov8n.pt start_camera:=false
```

입력은 `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`/camera/camera/color/camera_info`이며 출력은 `/asac/detections`, `/asac/annotated_image`이다.
이 모델은 결함/부위 detect 모델이다. 전체 사과 instance Mask나 탑뷰 구 피팅 모델을 대체하지 않으며,
실제 품질 등급·MoveIt·PiPER 동작을 시작하지 않는다.

재현 명령:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install --packages-select asac_perception
source install/local_setup.bash
python scripts/verify_imported_model.py --apply
ROS_DOMAIN_ID=187 python scripts/verify_imported_ros.py \
  --image datasets/apple-v2/test/images/003_JPG.rf.dd55138ed34db7171bf8a66081fa6034.jpg \
  --output /tmp/asac-import-ros.json
```

ROS replay는 실제 test RGB와 실제 best.pt를 사용한다. Depth는 의도적으로 전부 0이고
CameraInfo는 테스트 메타데이터이므로 **실제 Depth/카메라 보정/미터 좌표 측정 시험은 아니다**.
양성 샘플에서는 검출·동일 stamp overlay와 invalid depth/NaN 내부 좌표를 확인한다.
저장 JSON은 무효 좌표를 null로 쓴다.

첫 번째 test 샘플 `003_JPG.rf.0d706fc66647743d2d8601ca3c9b0a38.jpg`는 원본 RGB에서
Misshape confidence 약 .560이지만 최대 변 960px로 먼저 줄인 replay RGB에서는
confidence .5를 통과하지 못했다. 이 샘플의 빈 배열을 인식 성공으로 취급하지 않는다.
양성 메시지 시험에는 명시한 두 번째 실제 test 샘플을 사용하며 원래 confidence .5를 유지한다.
실제 카메라 영상의 해상도/촬영 조건 차이에 따른 검출 변화는 별도로 검증해야 한다.

현재 WSL에는 realsense2_camera 패키지와 접근 가능한 물리 카메라 입력이 없어
실카메라 시험은 미완료다. 카메라를 시작하지 않고 ROS 인식 노드만 검증했다.
현재 Python entry point로 패키지를 다시 빌드해 ROS detector가 프로젝트 venv에서 실행됨도 확인했다.
기존 단위/ROS/시뮬레이터 테스트는 **67개 통과**했다.
