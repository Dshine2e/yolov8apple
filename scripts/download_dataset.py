"""환경변수로 인증하여 지정한 Roboflow YOLOv8 데이터셋을 다운로드한다."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import shlex
import stat
import tempfile
from urllib.parse import quote
from urllib.request import urlopen
import zipfile


def read_credentials(path: Path) -> dict:
    """Read selected dotenv values as data; never source/execute user text."""
    values = {}
    if not path.is_file():
        return values
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        name, separator, value = line.partition('=')
        if not separator or name.strip() not in ('ROBOFLOW_API_KEY', 'ROBOFLOW_DATASET_URL'):
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f'Invalid credential quoting on line {number}') from None
        if not parts:
            values[name.strip()] = ''
            continue
        if len(parts) != 1:
            raise ValueError(f'Credential must be one value on line {number}')
        values[name.strip()] = parts[0]
    return values


def extract_dataset(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('ZIP contains an unsafe member path')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('ZIP contains a symbolic link')
        if source.testzip() is not None:
            raise ValueError('Dataset ZIP failed its CRC check')
        source.extractall(destination)
    if not (destination / 'data.yaml').is_file():
        raise ValueError('Export must contain data.yaml at the ZIP root')


def main() -> None:
    parser = argparse.ArgumentParser(description='Download apple-fbyiy version 2 in YOLOv8 format')
    parser.add_argument('--output', type=Path, default=Path('datasets/apple-v2'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--zip', type=Path, help='Use an already downloaded Roboflow export')
    parser.add_argument('--workspace', default='rita-kmex0')
    parser.add_argument('--project', default='apple-fbyiy')
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--format', default='yolov8')
    options = parser.parse_args()
    output = options.output.expanduser().resolve()
    if output.exists() and not output.is_dir():
        parser.error(f'Output path is not a directory: {output}')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}')
    try:
        credentials = read_credentials(options.env_file.expanduser())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    key = os.environ.get('ROBOFLOW_API_KEY', credentials.get('ROBOFLOW_API_KEY', '')).strip()
    url = os.environ.get(
        'ROBOFLOW_DATASET_URL', credentials.get('ROBOFLOW_DATASET_URL', '')).strip()
    if not options.zip and not url and not key:
        parser.error('Set ROBOFLOW_API_KEY or ROBOFLOW_DATASET_URL in the environment/.env, '
                     'or supply --zip; do not put credentials in source code')
    if url and not url.startswith('https://'):
        parser.error('ROBOFLOW_DATASET_URL must use HTTPS')
    if options.version < 1:
        parser.error('version must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-download-', dir=output.parent))
    export = staging / 'export'
    # SDK의 HTTP 로그/오류에 인증 URL이 포함될 수 있으므로 직접 노출하지 않는다.
    try:
        if options.zip:
            extract_dataset(options.zip.expanduser().resolve(), export)
        elif url:
            archive = staging / 'roboflow.zip'
            with urlopen(url, timeout=60) as response, archive.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            extract_dataset(archive, export)
        else:
            from roboflow import Roboflow
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rf = Roboflow(api_key=key)
                project = rf.workspace(options.workspace).project(options.project)
                project.version(options.version).download(options.format, location=str(export))
        if not (export / 'data.yaml').is_file():
            raise ValueError('Download did not produce data.yaml; incomplete exports cannot train')
        if output.exists():
            output.rmdir()  # Only the prechecked empty directory can be replaced.
        export.rename(output)
    except KeyboardInterrupt:
        raise SystemExit(f'Download interrupted; incomplete files retained at {staging}') from None
    except Exception as error:
        # 외부 SDK 예외의 종류가 다양하므로 CLI 경계에서만 처리하고 실패를 반환한다.
        message = str(error)
        for secret in (key, url):
            if secret:
                message = message.replace(secret, '[REDACTED]').replace(quote(secret), '[REDACTED]')
        # Some HTTP exceptions contain only a shortened signed URL. Hide URL exceptions entirely.
        if url:
            message = f'{type(error).__name__}; check export URL access and ZIP integrity'
        raise SystemExit(
            f'Roboflow download failed: {message}; incomplete files: {staging}') from None
    import shutil
    shutil.rmtree(staging)
    print(f'Dataset downloaded: {output}')
    print(f'Dataset YAML: {output / "data.yaml"}')


if __name__ == '__main__':
    main()
