from pathlib import Path

import pytest
import yaml

from asac_perception.train import audit_dataset, main, normalize_dataset


def make_export(root: Path, train='../train/images') -> Path:
    for split in ('train', 'valid'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir()
        (root / split / 'images' / 'apple.jpg').touch()
    source = root / 'data.yaml'
    source.write_text(yaml.safe_dump({
        'train': train, 'val': '../valid/images', 'nc': 1, 'names': ['apple'],
        'roboflow': {'workspace': 'rita-kmex0', 'project': 'apple-fbyiy', 'version': 2},
    }), encoding='utf-8')
    return source


def test_roboflow_relative_paths_and_original_preserved(tmp_path):
    source = make_export(tmp_path)
    before = source.read_bytes()
    normalized = normalize_dataset(source)
    assert normalized['train'] == str(tmp_path / 'train/images')
    assert normalized['val'] == str(tmp_path / 'valid/images')
    assert normalized['names'] == ['apple']
    assert 'roboflow' not in normalized
    assert source.read_bytes() == before


def test_export_local_paths(tmp_path):
    normalized = normalize_dataset(make_export(tmp_path, train='train/images'))
    assert normalized['train'] == str(tmp_path / 'train/images')


def test_missing_images_rejected(tmp_path):
    source = make_export(tmp_path)
    (tmp_path / 'train/images/apple.jpg').unlink()
    with pytest.raises(ValueError, match='no supported images'):
        normalize_dataset(source)


def test_nc_mismatch_rejected(tmp_path):
    source = make_export(tmp_path)
    data = yaml.safe_load(source.read_text())
    data['nc'] = 2
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='nc'):
        normalize_dataset(source)


def annotated_export(root, row='0 0.5 0.5 0.3 0.4'):
    source = make_export(root)
    for split in ('train', 'valid'):
        (root / split / 'labels/apple.txt').write_text(row, encoding='utf-8')
    return source


def test_actual_box_labels_detected(tmp_path):
    result = audit_dataset(normalize_dataset(annotated_export(tmp_path)))
    assert result['task'] == 'detect'
    assert result['splits']['train']['class_counts'] == [1]


def test_actual_polygon_labels_detected(tmp_path):
    result = audit_dataset(normalize_dataset(
        annotated_export(tmp_path, '0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8')))
    assert result['task'] == 'segment'


@pytest.mark.parametrize('row, reason', [
    ('0 nan 0.5 0.3 0.4', 'nonfinite'),
    ('0 1.1 0.5 0.3 0.4', 'normalized'),
    ('1 0.5 0.5 0.3 0.4', 'class index'),
    ('0 0.5 0.5 0 0.4', 'positive'),
    ('0 0.2 0.2 0.5 0.5 0.8 0.8', 'zero area'),
    ('0 0.5 0.5', 'expected bbox'),
])
def test_invalid_labels_rejected(tmp_path, row, reason):
    with pytest.raises(ValueError, match=reason):
        audit_dataset(normalize_dataset(annotated_export(tmp_path, row)))


def test_mixed_boxes_and_polygons_rejected(tmp_path):
    source = annotated_export(tmp_path)
    (tmp_path / 'valid/labels/apple.txt').write_text('0 0.2 0.2 0.8 0.2 0.8 0.8')
    with pytest.raises(ValueError, match='Mixed'):
        audit_dataset(normalize_dataset(source))


def test_background_missing_labels_counted(tmp_path):
    source = annotated_export(tmp_path)
    (tmp_path / 'train/images/background.png').touch()
    result = audit_dataset(normalize_dataset(source))
    assert result['splits']['train']['images'] == 2
    assert result['splits']['train']['background_images'] == 1
    assert result['splits']['train']['missing_label_files'] == 1


def test_box_dataset_cannot_request_segmentation(tmp_path, capsys):
    source = annotated_export(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(['--data', str(source), '--task', 'segment', '--validate-only'])
    assert error.value.code == 2
    assert 'Dataset labels are detect' in capsys.readouterr().err


def test_subset_cannot_apply_default_model(tmp_path, capsys):
    source = annotated_export(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(['--data', str(source), '--fraction', '0.05', '--apply', '--validate-only'])
    assert error.value.code == 2
    assert 'full train split' in capsys.readouterr().err
