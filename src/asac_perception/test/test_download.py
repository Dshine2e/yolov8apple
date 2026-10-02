"""다운로드 CLI의 인증정보 노출과 기존 export 덮어쓰기를 검증한다."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def downloader(monkeypatch):
    script = Path(__file__).resolve().parents[3] / 'scripts/download_dataset.py'
    spec = importlib.util.spec_from_file_location('asac_download', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ROBOFLOW_API_KEY', 'test-secret-key')
    return module


def test_sdk_errors_and_logs_redact_key(downloader, monkeypatch, tmp_path, capsys):
    def fail_sdk(api_key):
        print(f'HTTP SDK log key={api_key}')
        print(f'HTTP SDK stderr key={api_key}', file=sys.stderr)
        raise RuntimeError(f'HTTP authentication failed: api_key={api_key}')

    monkeypatch.setitem(sys.modules, 'roboflow', SimpleNamespace(Roboflow=fail_sdk))
    monkeypatch.setattr(sys, 'argv', ['download', '--output', str(tmp_path / 'data')])
    with pytest.raises(SystemExit) as error:
        downloader.main()
    assert '[REDACTED]' in str(error.value)
    captured = capsys.readouterr()
    assert 'test-secret-key' not in captured.out + captured.err + str(error.value)


def test_existing_export_is_preserved(downloader, monkeypatch, tmp_path):
    source = tmp_path / 'data.yaml'
    source.write_text('original export', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['download', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        downloader.main()
    assert error.value.code == 2
    assert source.read_text(encoding='utf-8') == 'original export'


def test_dotenv_is_data_and_never_executed(downloader, tmp_path):
    credentials = tmp_path / '.env'
    sentinel = tmp_path / 'executed'
    credentials.write_text(
        f'ROBOFLOW_API_KEY="$(touch {sentinel})"\n'
        'export ROBOFLOW_DATASET_URL="https://app.roboflow.com/ds/example?key=secret" # note\n'
        'UNRELATED_VARIABLE=ignored\n', encoding='utf-8',
    )
    values = downloader.read_credentials(credentials)
    assert values['ROBOFLOW_API_KEY'] == f'$(touch {sentinel})'
    assert not sentinel.exists()
    assert 'UNRELATED_VARIABLE' not in values


@pytest.mark.parametrize('member', ['../outside', '/absolute', r'..\outside'])
def test_zip_traversal_rejected(downloader, tmp_path, member):
    archive = tmp_path / 'export.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('data.yaml', 'names: [apple]')
        source.writestr(member, 'unsafe')
    with pytest.raises(ValueError, match='unsafe'):
        downloader.extract_dataset(archive, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_local_zip_requires_no_key_and_preserves_archive(downloader, monkeypatch, tmp_path):
    monkeypatch.delenv('ROBOFLOW_API_KEY')
    archive = tmp_path / 'export.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('data.yaml', 'names: [apple]')
        source.writestr('train/labels/apple.txt', '0 0.5 0.5 0.2 0.2')
    before = archive.read_bytes()
    output = tmp_path / 'data'
    output.mkdir()  # An empty destination must not make SDK-style downloads silently skip.
    arguments = ['download', '--zip', str(archive), '--output', str(output),
                 '--env-file', str(tmp_path / 'absent.env')]
    monkeypatch.setattr(sys, 'argv', arguments)
    downloader.main()
    assert (output / 'data.yaml').is_file()
    assert (output / 'train/labels/apple.txt').is_file()
    assert archive.read_bytes() == before


def test_incomplete_zip_not_promoted(downloader, monkeypatch, tmp_path):
    archive = tmp_path / 'export.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('train/labels/apple.txt', '0 0.5 0.5 0.2 0.2')
    output = tmp_path / 'data'
    monkeypatch.setattr(sys, 'argv', ['download', '--zip', str(archive), '--output', str(output)])
    with pytest.raises(SystemExit, match='data.yaml'):
        downloader.main()
    assert not output.exists()
