"""다운로드 CLI의 인증정보 노출과 기존 export 덮어쓰기를 검증한다."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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
