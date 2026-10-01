"""환경변수로 인증하여 지정한 Roboflow YOLOv8 데이터셋을 다운로드한다."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from urllib.parse import quote


def main() -> None:
    parser = argparse.ArgumentParser(description='Download apple-fbyiy version 2 in YOLOv8 format')
    parser.add_argument('--output', type=Path, default=Path('datasets/apple-v2'))
    options = parser.parse_args()
    key = os.environ.get('ROBOFLOW_API_KEY', '').strip()
    if not key:
        parser.error('Set ROBOFLOW_API_KEY in the environment; do not put the key in source code')
    output = options.output.expanduser().resolve()
    if output.exists() and not output.is_dir():
        parser.error(f'Output path is not a directory: {output}')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    from roboflow import Roboflow
    # SDK의 HTTP 로그/오류에 인증 URL이 포함될 수 있으므로 직접 노출하지 않는다.
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rf = Roboflow(api_key=key)
            project = rf.workspace('rita-kmex0').project('apple-fbyiy')
            dataset = project.version(2).download('yolov8', location=str(output))
    except KeyboardInterrupt:
        raise SystemExit('Download interrupted; keep the incomplete directory separate') from None
    except Exception as error:
        # 외부 SDK 예외의 종류가 다양하므로 CLI 경계에서만 처리하고 실패를 반환한다.
        message = str(error).replace(key, '[REDACTED]').replace(quote(key), '[REDACTED]')
        raise SystemExit(f'Roboflow download failed: {message}') from None
    print(f'Dataset downloaded: {dataset.location}')
    print(f'Dataset YAML: {output / "data.yaml"}')


if __name__ == '__main__':
    main()
