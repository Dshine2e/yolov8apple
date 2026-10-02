#!/usr/bin/env python3
"""Verify delivered YOLOv8 files, re-evaluate locally, and optionally select best.pt."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

ASAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASAC_ROOT / 'src/asac_perception'))
from asac_perception.model_selection import activate_detector  # noqa: E402
from asac_perception.train import audit_dataset, normalize_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=ASAC_ROOT / 'models/imported/apple-v2/best.pt')
    parser.add_argument('--data', type=Path, default=ASAC_ROOT / 'datasets/apple-v2/data.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--apply', action='store_true')
    options = parser.parse_args()
    model_path = options.model.expanduser().resolve()
    incoming = model_path.parent
    required = ['best.pt', 'training_report.json', 'args.yaml', 'results.csv']
    hashes = {name: hashlib.sha256((incoming / name).read_bytes()).hexdigest()
              for name in required}
    source_report = json.loads((incoming / 'training_report.json').read_text(encoding='utf-8'))
    receipt_path = incoming / 'delivery_receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if any(receipt['sha256'].get(name) != value for name, value in hashes.items()):
            raise ValueError('Delivered files do not match receipt SHA256 values')
    if hashes['best.pt'] != source_report['best_pt_sha256']:
        raise ValueError('best.pt hash differs from training report')
    source_args = yaml.safe_load((incoming / 'args.yaml').read_text(encoding='utf-8'))
    with (incoming / 'results.csv').open(encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    if len(history) != source_report['completed_epochs']:
        raise ValueError('results.csv epoch count differs from training report')
    if [int(float(row['epoch'])) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError('Training history has missing/duplicate epoch rows')
    if source_report['status'] != 'completed' or source_report['fraction'] != 1.0:
        raise ValueError('Expected completed full-dataset training')
    if source_args.get('fraction') != 1.0:
        raise ValueError('args.yaml records subset training')
    dataset = normalize_dataset(options.data)
    audit = audit_dataset(dataset)
    from ultralytics import YOLO
    import torch
    import ultralytics
    model = YOLO(str(model_path))
    names = {str(key): value for key, value in model.names.items()}
    expected_names = {str(key): value for key, value in enumerate(dataset['names'])}
    if model.task != 'detect' or model.task != audit['task']:
        raise ValueError('This import requires detection model and actual detection labels')
    if names != source_report['names'] or names != expected_names:
        raise ValueError('Checkpoint, training report and local dataset class names differ')
    source_splits = source_report['dataset_audit']['splits']
    for key, source_key in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        local = audit['splits'][key]
        supplied = source_splits[source_key]
        if local['images'] != supplied['images']:
            raise ValueError(f'{key} image count differs from delivered dataset audit')
        if dict(zip(dataset['names'], local['class_counts'])) != supplied['boxes']:
            raise ValueError(f'{key} object counts differ from delivered dataset audit')
    destination = options.output or ASAC_ROOT / 'runs/import-verification' / datetime.now(
        timezone.utc).strftime('apple-v2-%Y%m%dT%H%M%S%f')
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    data_path = destination / 'dataset.yaml'
    data_path.write_text(yaml.safe_dump(dataset, allow_unicode=True), encoding='utf-8')
    metrics = model.val(data=str(data_path), split='test', imgsz=options.imgsz,
                        batch=options.batch, device='cpu', workers=0,
                        project=str(destination), name='test', exist_ok=False,
                        save_json=True, plots=True)
    result = {
        'status': 'verified', 'task': model.task, 'names': model.names,
        'model_path': str(model_path), 'received_file_sha256': hashes,
        'completed_epochs': source_report['completed_epochs'],
        'requested_epochs': source_report['requested_epochs'],
        'local_dataset_audit': audit, 'evaluation_split': 'test',
        'evaluation_images': audit['splits']['test']['images'],
        'metrics': {key: float(value) for key, value in metrics.results_dict.items()},
        'reported_test_metrics': source_report['metrics'],
        'absolute_metric_delta': {
            key: abs(float(value) - source_report['metrics'][key])
            for key, value in metrics.results_dict.items() if key in source_report['metrics']
        },
        'speed_ms_per_image': metrics.speed,
        'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'input_imgsz': options.imgsz, 'batch': options.batch, 'device': 'cpu',
        'physical_camera_tested': False,
        'topview_segmentation_model_changed': False,
        'per_class': [
            {'name': name, 'precision': float(metrics.box.p[i]),
             'recall': float(metrics.box.r[i]), 'mAP50': float(metrics.box.ap50[i]),
             'mAP50_95': float(metrics.box.ap[i]),
             'test_instances': audit['splits']['test']['class_counts'][i]}
            for i, name in enumerate(dataset['names'])
        ],
    }
    for image in sorted(Path(dataset['test']).glob('*.jpg'))[:3]:
        prediction = model.predict(str(image), imgsz=640, device='cpu', conf=0.5,
                                   verbose=False, retina_masks=False)[0]
        prediction.save(filename=str(destination / f'sample-{image.stem}.jpg'))
    if options.apply:
        result['detector_selection'] = str(activate_detector(ASAC_ROOT, model_path, source_report))
    report = destination / 'verification.json'
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                      encoding='utf-8')
    print(f'Local verification report: {report}')
    print(json.dumps(result['metrics'], indent=2))
    if options.apply:
        print(f'Default detector selected: {model_path}')


if __name__ == '__main__':
    main()
