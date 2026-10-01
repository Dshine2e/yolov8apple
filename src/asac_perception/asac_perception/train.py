"""Roboflow YOLOv8 export의 경로를 검증하고 사과 탐지 모델을 학습한다."""

import argparse
import math
from pathlib import Path

import yaml


def normalize_dataset(source: Path) -> dict:
    source = source.expanduser().resolve()
    with source.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get('names'), (list, dict)):
        raise ValueError('data.yaml must contain class names')
    names = data['names']
    if not names or any(not isinstance(name, str) or not name for name in (
        names.values() if isinstance(names, dict) else names
    )):
        raise ValueError('Dataset class names must be nonempty strings')
    if isinstance(names, dict) and sorted(names) != list(range(len(names))):
        raise ValueError('Class IDs must be consecutive integers starting at zero')
    if 'nc' in data and data['nc'] != len(names):
        raise ValueError('nc does not match the number of class names')
    if data.get('path') is not None and not isinstance(data['path'], str):
        raise ValueError('Dataset path must be a string')
    base = Path(data.get('path') or source.parent)
    if not base.is_absolute():
        base = source.parent / base
    normalized = {'path': str(source.parent), 'names': names, 'nc': len(names)}
    for split in ('train', 'val', 'test'):
        value = data.get(split)
        if not value and split == 'test':
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f'{split} must be an image directory path')
        path = Path(value)
        candidates = [path] if path.is_absolute() else [base / path, source.parent / path]
        # Roboflow export는 data.yaml 옆의 train/valid 폴더를 ../train 등으로 표기한다.
        # 원래 경로가 없을 때만 해당 export 내부 경로로 보정한다.
        if not path.is_absolute() and path.parts and path.parts[0] == '..':
            candidates.append(source.parent.joinpath(*path.parts[1:]))
        resolved = next((p.resolve() for p in candidates if p.is_dir()), None)
        if resolved is None:
            raise ValueError(f'{split} image directory not found: {value}')
        if not any(p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                   for p in resolved.rglob('*') if p.is_file()):
            raise ValueError(f'{split} has no supported images: {resolved}')
        label_dir = resolved.parent / 'labels'
        if not label_dir.is_dir():
            raise ValueError(f'{split} label directory not found: {label_dir}')
        normalized[split] = str(resolved)
    return normalized


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description='Train YOLOv8 on the Roboflow apple v2 export')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--imgsz', default=640, type=int)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fraction', default=1.0, type=float)
    parser.add_argument('--project', default=Path('runs/apple'), type=Path)
    parser.add_argument('--name', default='yolov8n_v2')
    parser.add_argument('--validate-only', action='store_true')
    options = parser.parse_args(args)
    if min(options.epochs, options.imgsz, options.batch) < 1 or options.workers < 0:
        parser.error('epochs, imgsz, batch must be positive; workers must be nonnegative')
    if not math.isfinite(options.fraction) or not 0 < options.fraction <= 1:
        parser.error('fraction must be in (0, 1]')
    if options.name in ('', '.', '..') or Path(options.name).name != options.name:
        parser.error('name must be a single directory name')
    try:
        data = normalize_dataset(options.data)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if options.validate_only:
        print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        return
    # 실행별 별도 YAML을 생성하고 원본 export와 기존 학습 결과는 보존한다.
    project = options.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', prefix='dataset_', dir=project, delete=False,
        encoding='utf-8',
    ) as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
        normalized_path = Path(stream.name)
    from ultralytics import YOLO
    model = YOLO(options.model, task='detect')
    model.train(
        data=str(normalized_path), epochs=options.epochs, imgsz=options.imgsz,
        batch=options.batch, device=options.device, workers=options.workers,
        seed=options.seed, deterministic=True, project=str(project), name=options.name,
        fraction=options.fraction, exist_ok=False,
    )
    print(f'Best weights: {Path(model.trainer.save_dir) / "weights/best.pt"}')


if __name__ == '__main__':
    main()
