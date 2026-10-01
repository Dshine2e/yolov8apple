from pathlib import Path

import pytest
import yaml

from asac_perception.train import normalize_dataset


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
