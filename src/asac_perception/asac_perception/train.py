"""Validate Roboflow labels, train YOLOv8 and evaluate the actual best.pt."""

import argparse
import math
import json
from pathlib import Path

import yaml

from asac_perception.model_selection import activate_detector

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


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
        if not any(p.suffix.lower() in IMAGE_SUFFIXES
                   for p in resolved.rglob('*') if p.is_file()):
            raise ValueError(f'{split} has no supported images: {resolved}')
        label_dir = resolved.parent / 'labels'
        if not label_dir.is_dir():
            raise ValueError(f'{split} label directory not found: {label_dir}')
        normalized[split] = str(resolved)
    return normalized


def audit_dataset(data: dict) -> dict:
    """Read actual YOLO rows: boxes cannot become segmentation polygons."""
    formats = set()
    splits = {}
    for split in ('train', 'val', 'test'):
        if split not in data:
            continue
        directory = Path(data[split])
        counts = {'images': 0, 'instances': 0, 'background_images': 0,
                  'missing_label_files': 0, 'class_counts': [0] * data['nc']}
        for image in sorted(directory.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            counts['images'] += 1
            label = directory.parent / 'labels' / image.relative_to(directory).with_suffix('.txt')
            if not label.exists():
                counts['missing_label_files'] += 1
                counts['background_images'] += 1
                continue
            instances = 0
            for number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                context = f'{label}:{number}'
                try:
                    values = [float(v) for v in line.split()]
                except ValueError:
                    raise ValueError(f'{context}: nonnumeric label') from None
                if not values or not all(math.isfinite(v) for v in values):
                    raise ValueError(f'{context}: nonfinite label')
                cls, coordinates = values[0], values[1:]
                if not cls.is_integer() or not 0 <= cls < data['nc']:
                    raise ValueError(f'{context}: class index outside data.yaml names')
                if any(not 0 <= value <= 1 for value in coordinates):
                    raise ValueError(f'{context}: coordinates must be normalized to [0, 1]')
                if len(values) == 5:
                    if coordinates[2] <= 0 or coordinates[3] <= 0:
                        raise ValueError(f'{context}: bbox width/height must be positive')
                    formats.add('detect')
                elif len(values) >= 7 and len(coordinates) % 2 == 0:
                    points = list(zip(coordinates[::2], coordinates[1::2]))
                    area = abs(sum(x * points[(i + 1) % len(points)][1]
                                   - y * points[(i + 1) % len(points)][0]
                                   for i, (x, y) in enumerate(points))) / 2
                    if area <= 1e-12:
                        raise ValueError(f'{context}: polygon has zero area')
                    formats.add('segment')
                else:
                    raise ValueError(f'{context}: expected bbox or at least three polygon points')
                instances += 1
                counts['class_counts'][int(cls)] += 1
            counts['instances'] += instances
            counts['background_images'] += instances == 0
        if split in ('train', 'val') and not counts['instances']:
            raise ValueError(f'{split} has no annotated objects')
        splits[split] = counts
    if len(formats) != 1:
        raise ValueError('Mixed detection/segmentation labels are not supported')
    return {'task': formats.pop(), 'splits': splits}


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description='Train YOLOv8 on the Roboflow apple v2 export')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--model', help='Pretrained weights; defaults to the actual label task')
    parser.add_argument('--task', choices=('auto', 'detect', 'segment'), default='auto')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--imgsz', default=640, type=int)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fraction', default=1.0, type=float)
    parser.add_argument('--project', default=Path('runs/apple'), type=Path)
    parser.add_argument('--name', default='yolov8n_v2')
    parser.add_argument('--apply', action='store_true',
                        help='Select completed full-data detect best.pt for run_detector.sh')
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
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
        audit = audit_dataset(data)
        if options.task != 'auto' and options.task != audit['task']:
            raise ValueError(
                f"Dataset labels are {audit['task']}, requested task is {options.task}")
        if options.apply and (audit['task'] != 'detect' or options.fraction != 1.0
                              or options.epochs < 2):
            raise ValueError('--apply requires detection, full train split and at least 2 epochs; '
                             'segmentation keeps its separate topview configuration')
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if options.validate_only:
        print(yaml.safe_dump({'dataset': data, 'audit': audit},
                             allow_unicode=True, sort_keys=False))
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
    import ultralytics
    import torch
    import platform
    model_path = options.model
    if model_path is None:
        filename = 'yolov8n-seg.pt' if audit['task'] == 'segment' else 'yolov8n.pt'
        local = options.workspace / 'models' / filename
        model_path = str(local) if local.is_file() else filename
    model = YOLO(model_path)
    if model.task != audit['task']:
        parser.error(f'Model task {model.task} does not match dataset {audit["task"]}')
    model.train(
        data=str(normalized_path), epochs=options.epochs, imgsz=options.imgsz,
        batch=options.batch, device=options.device, workers=options.workers,
        seed=options.seed, deterministic=True, project=str(project), name=options.name,
        fraction=options.fraction, exist_ok=False,
    )
    directory = Path(model.trainer.save_dir)
    best = directory / 'weights/best.pt'
    if not best.is_file():
        raise RuntimeError('Training did not produce best.pt; detector selection was preserved')
    evaluated = YOLO(str(best))
    split = 'test' if 'test' in data and audit['splits']['test']['instances'] else 'val'
    metrics = evaluated.val(data=str(normalized_path), split=split, device=options.device,
                            imgsz=options.imgsz, batch=options.batch, workers=options.workers,
                            project=str(directory), name=f'best_{split}', exist_ok=False)
    names = evaluated.names
    report = {
        'status': 'completed', 'task': evaluated.task, 'names': names,
        'dataset_yaml': str(options.data.resolve()), 'normalized_yaml': str(normalized_path),
        'audit': audit, 'requested_epochs': options.epochs,
        'completed_epochs': model.trainer.epoch + 1, 'fraction': options.fraction,
        'best_pt': str(best.resolve()), 'evaluation_split': split,
        'metrics': {key: float(value) if math.isfinite(float(value)) else None
                    for key, value in metrics.results_dict.items()},
        'device': options.device, 'seed': options.seed,
        'versions': {'python': platform.python_version(), 'ultralytics': ultralytics.__version__,
                     'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()},
    }
    report_path = directory / 'training_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                           encoding='utf-8')
    print(f'Best weights: {best.resolve()}')
    print(f'Training/evaluation report: {report_path.resolve()}')
    if options.apply:
        selected = activate_detector(options.workspace, best, report)
        print(f'Detector model selected: {selected}; all model classes retain their actual names')


if __name__ == '__main__':
    main()
