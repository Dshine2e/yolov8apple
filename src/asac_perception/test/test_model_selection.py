from pathlib import Path
import json

import pytest

from asac_perception.model_selection import activate_detector, detector_selection


def artifact(root: Path):
    weights = root / 'runs/apple/trial/weights/best.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'unit-test checkpoint bytes; not a usable model')
    return weights


def report(**overrides):
    return dict(status='completed', task='detect', names={0: 'Intact'},
                fraction=1.0, completed_epochs=100, **overrides)


def test_pretrained_fallback_preserved(tmp_path):
    model, classes = detector_selection(tmp_path)
    assert model == tmp_path / 'models/yolov8n.pt'
    assert classes == 'apple'


def test_selected_best_and_real_class_names(tmp_path):
    best = artifact(tmp_path)
    activate_detector(tmp_path, best, report())
    selected, classes = detector_selection(tmp_path)
    assert selected == best
    assert classes == ''  # All literal dataset classes; no invented apple label.
    before = (tmp_path / 'models/active_detector.json').read_bytes()
    activate_detector(tmp_path, best, report())
    backups = list((tmp_path / 'models').glob('active_detector-*.json'))
    assert len(backups) == 1 and backups[0].read_bytes() == before


def test_changed_weight_content_refused(tmp_path):
    best = artifact(tmp_path)
    activate_detector(tmp_path, best, report())
    best.write_bytes(b'changed checkpoint bytes')
    with pytest.raises(ValueError, match='weights changed'):
        detector_selection(tmp_path)


def test_imported_artifact_uses_adjacent_training_report(tmp_path):
    imported = tmp_path / 'models/imported/apple-v2'
    imported.mkdir(parents=True)
    best = imported / 'best.pt'
    best.write_bytes(b'unit-test checkpoint bytes')
    report_path = imported / 'training_report.json'
    report_path.write_text(json.dumps(report()))
    selected = activate_detector(tmp_path, best, report())
    assert json.loads(selected.read_text())['training_report'] == str(report_path)


@pytest.mark.parametrize('change', [
    {'task': 'segment'}, {'status': 'interrupted'},
    {'fraction': 0.05}, {'completed_epochs': 1},
])
def test_unfinished_or_incompatible_training_not_applied(tmp_path, change):
    best = artifact(tmp_path)
    data = report()
    data.update(change)
    with pytest.raises(ValueError):
        activate_detector(tmp_path, best, data)
    assert not (tmp_path / 'models/active_detector.json').exists()
