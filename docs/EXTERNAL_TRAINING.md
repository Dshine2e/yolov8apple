# 다른 GPU 환경에서 학습하고 best.pt 전달

사용자는 2026-10-02에 다른 GPU 환경에서 학습 후 best.pt를 전달하는 방식을 선택했다.
현재 작업공간에서는 전체 CPU 학습을 시작하지 않았다. 아래 묶음과 명령은
ROS 없이 **같은 YOLOv8 코드와 실제 Roboflow 라벨**을 사용한다.

## 현재 확인 결과

- 받은 ZIP: `src/asac_perception/resource/Apple.v2i.yolov8.zip` (약 1.79 GiB).
- `datasets/apple-v2`로 압축 해제했다. CRC 검사 통과. 현재 확인된 사용 가능 데이터는
  이 압축 해제 디렉터리이며 학습 묶음도 이 파일들을 사용한다.
- train 1,102 / validation 366 / test 366, 합계 1,834장.
- 전체 이미지 헤더/파일 무결성 검사 통과. split 간 완전히 동일한 파일 hash 0개.
  원본 촬영 세션별 독립성을 증명한 검사는 아니므로 유사 영상 누출 가능성은 별도로 살펴야 한다.
- 검출 bbox 라벨, 7개 클래스, train 객체 3,829 / validation 1,264 / test 1,163.
- 클래스 순서: Calyx, Disease, Intact, Miscolor, Misshape, Scar, Stem.
  train의 background 이미지 1개, 누락 라벨 파일 0개.
- 실제 GPU: **AMD Radeon RX 9060 XT**, Windows 11 Home.
  현재 WSL의 PyTorch는 **2.5.1+cpu**, GPU 사용 불가. ROCm은 설치되어 있지 않다.

Windows에서 만든 YOLOv8 `.pt`는 이 Linux 워크스페이스에서도 로드할 수 있다.
OS의 차이보다 **모델 task/Ultralytics 호환성/실제 클래스/학습 라벨**이 중요하다.
이 데이터는 detect 전용이므로 새 best.pt는 기존 detector에 적용한다. 탑뷰 YOLOv8-seg에
삽입하거나 bbox를 mask로 만들지 않는다.

처음 조사한 ROCm 7.2.1 문서는 Windows의 ML 학습을 지원하지 않는다고 명시했으나,
그 제한을 모든 후속 버전에 적용해서는 안 된다. 이번에 받은 실제 보고서는 Windows의
RX 9060 XT에서 **PyTorch 2.12.0+rocm7.14.1 / Python 3.12.14 / Ultralytics 8.3.40**으로
71 epoch 학습 완료를 기록한다. 이 모델의 수신 hash와 로컬 CPU 로드·test 재평가는 확인했다.
학습 GPU 런타임을 이 WSL에서 재현한 시험은 아니며, 현재 WSL에는 ROCm이 없다.

## 전달용 묶음

`runs/handoff/asac-apple-v2-training.zip`에 압축 해제된 데이터셋, YOLOv8n 초기 pretrained 가중치,
학습 Python 코드, 설치 의존성, 이 안내, 라이선스를 함께 넣었다.
외부 GPU 머신의 로컬 폴더로 복사해 압축을 해제한다. `datasets/apple-v2/data.yaml`과
세 split이 바로 생기므로 Roboflow 재다운로드와 중첩 ZIP 압축 해제가 필요 없다.
메타데이터와 파일 hash는 같은 묶음의 `handoff-manifest.json`에 있다.

## Windows + NVIDIA GPU 예시

이 절은 **NVIDIA GPU가 있는 다른 Windows 머신**용이다. 현재 AMD PC에 CUDA wheel을
설치하는 명령이 아니다. Python 3.10과 NVIDIA 드라이버가 준비되어 있어야 한다.
[PyTorch 공식 2.5.1 설치 명령](https://pytorch.org/get-started/previous-versions/#v251)을 사용한다.
PowerShell에서 묶음을 푼 폴더로 이동한 뒤 실행한다.

```powershell
py -3.10 -m venv .venv-train
$ASAC_PYTHON = '.\.venv-train\Scripts\python.exe'
& $ASAC_PYTHON -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& $ASAC_PYTHON -m pip install -r requirements-training.txt
& $ASAC_PYTHON -c "import torch; assert torch.cuda.is_available(), 'GPU unavailable'; print(torch.cuda.get_device_name(0))"
& $ASAC_PYTHON scripts/train_export.py --data datasets/apple-v2/data.yaml --task detect --validate-only
& $ASAC_PYTHON scripts/train_export.py --data datasets/apple-v2/data.yaml --task detect --model models/yolov8n.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --workers 2 --seed 42 --project runs/apple --name yolov8n_roboflow_v2
```

Windows에서는 학습 entry point가 `if __name__ == '__main__'` 안에 있어 worker의
재귀 실행을 막는다. 메모리가 부족하면 batch 4/2로 낮춘다. GPU 미인식 시 device 0 실행은
실패해야 하며, CPU로 조용히 전환해 GPU 학습이라고 보고하지 않는다.

## Linux / GPU 클라우드

동일 코드를 실행할 수 있다. NVIDIA 머신에서는 GPU가 연결된 작업 세션에서 다음과 같이 한다.

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-training.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/train_export.py --data datasets/apple-v2/data.yaml --task detect \
  --model models/yolov8n.pt --epochs 100 --imgsz 640 --batch 8 --device 0 \
  --workers 2 --seed 42 --project runs/apple --name yolov8n_roboflow_v2
```

AMD ROCm 환경에서는 머신의 GPU/OS와 호환되는 공식 ROCm PyTorch/torchvision을 먼저 설치하고
GPU 연산과 backward를 확인한다. 위 CUDA wheel을 쓰지 않는다. 최신 torch와 이전
Ultralytics 조합을 임의로 섞으면 checkpoint 로딩/연산 호환성 문제가 생길 수 있으므로
실제로 성공한 버전을 함께 기록해 전달한다.

이 머신에서 Windows/GPU 학습 명령을 실행한 것은 아니다. 로컬에서는 동일 Python entry point의
실제 export 라벨 검증과 기존 테스트를 실행했다. 외부 환경의 GPU 학습 성공은 해당 로그로 확인한다.

## 학습 결과 전달

묶음의 데이터 출처/라이선스는 `datasets/apple-v2/README.dataset.txt`의 Rita / CC BY 4.0이며,
초기 YOLOv8n 가중치는 [Ultralytics 공식 assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0)의
COCO pretrained 모델이다. Ultralytics 코드·가중치의 AGPL-3.0/Enterprise 조건은
[공식 라이선스](https://www.ultralytics.com/license)를 확인한다.

학습이 끝나면 출력된 실제 run 디렉터리에서 다음 파일을 전달한다.

- 필수: `weights/best.pt`.
- 함께 권장: `training_report.json`, `results.csv`, `args.yaml`.

이 도구는 전체 train split을 사용하고 validation split으로 best를 고른 다음,
별도 test 366장으로 best.pt를 평가한다. test 결과는 학습 설정을 고르기 위한 반복 튜닝에
사용하지 않는다. 실카메라/탑뷰 시뮬레이션의 도메인 차이는 이 test metric만으로 검증되지 않는다.

최종 `.pt`는 현재 워크스페이스의 `models/imported/apple-v2/best.pt`에 놓으면 된다.
원본 가중치와 학습 기록을 보존한다. 파일을 받은 뒤 이 환경에서 `model.task`, `model.names`,
메타데이터, test 평가 및 실제 추론을 확인하고 기존 detector 실행 선택에 연결한다.
best.pt만 전달해도 모델 로드/클래스 확인/로컬 평가가 가능하지만 학습 epoch/중단 여부 등
학습 기록이 없는 정보는 확인된 것처럼 보고하지 않는다.

최초 shell 오류는 Python 문장(`from ...`, 함수 호출)을 Bash에 그대로 넣어서 발생했다.
Python 파일을 `python 파일.py`로 실행한다. 로컬 ZIP이 확보되어 이번 전달 과정에서는
Roboflow API 키가 필요하지 않다.
