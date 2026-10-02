"""Select a verified training artifact without changing pretrained checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def activate_detector(root: Path, checkpoint: Path, report: dict) -> Path:
    if report.get('task') != 'detect' or report.get('status') != 'completed':
        raise ValueError('Only completed detection training can select the detector model')
    if report.get('fraction') != 1.0 or report.get('completed_epochs', 0) < 2:
        raise ValueError('Smoke/subset training cannot replace the default detector model')
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.name != 'best.pt':
        raise ValueError('Expected an existing best.pt training artifact')
    directory = root.resolve() / 'models'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'active_detector.json'
    report_path = checkpoint.parent / 'training_report.json'
    if not report_path.is_file():
        report_path = checkpoint.parents[1] / 'training_report.json'
    selection = {
        'schema_version': 1, 'task': 'detect', 'model_path': str(checkpoint),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'target_classes': '', 'names': report['names'],
        'training_report': str(report_path),
        'selected_at': datetime.now(timezone.utc).isoformat(),
    }
    if target.exists():
        previous = directory / f'active_detector-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json'
        previous.write_bytes(target.read_bytes())
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False,
                                     encoding='utf-8') as stream:
        json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, target)
    return target


def detector_selection(root: Path) -> tuple:
    root = root.resolve()
    selected = root / 'models/active_detector.json'
    if not selected.exists():
        return root / 'models/yolov8n.pt', 'apple'
    data = json.loads(selected.read_text(encoding='utf-8'))
    model = Path(data['model_path'])
    if data.get('task') != 'detect' or not model.is_file():
        raise ValueError('Selected detector artifact is invalid; supply an explicit .pt path')
    if hashlib.sha256(model.read_bytes()).hexdigest() != data.get('sha256'):
        raise ValueError('Selected detector weights changed since training; select them again')
    return model, data.get('target_classes', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    options = parser.parse_args()
    try:
        model, classes = detector_selection(options.root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(model)
    print(classes)


if __name__ == '__main__':
    main()
