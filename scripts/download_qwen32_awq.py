"""Resumable direct download of the pinned Qwen2.5 32B AWQ model files."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import subprocess


REVISION = '5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c'
REPO = 'Qwen/Qwen2.5-32B-Instruct-AWQ'
DEST = Path('vendor/models/Qwen2.5-32B-Instruct-AWQ')
SIZES = {
    '.gitattributes': 1518,
    'LICENSE': 11343,
    'README.md': 6372,
    'config.json': 841,
    'generation_config.json': 243,
    'merges.txt': 1671839,
    'model-00001-of-00005.safetensors': 3943575336,
    'model-00002-of-00005.safetensors': 3980134208,
    'model-00003-of-00005.safetensors': 3947411392,
    'model-00004-of-00005.safetensors': 3980134240,
    'model-00005-of-00005.safetensors': 3477738728,
    'model.safetensors.index.json': 143195,
    'tokenizer.json': 7031645,
    'tokenizer_config.json': 7305,
    'vocab.json': 2776833,
}
WEIGHT_SHA256 = {
    'model-00001-of-00005.safetensors': '548f5c7078e297088c74bec4443bfe9e3b4183ee7457f328107c37a5eb861ea1',
    'model-00002-of-00005.safetensors': '735d1b5aa8ef01420f2079b355d84cfa7a4b37571d44715276b3b903c06d65d8',
    'model-00003-of-00005.safetensors': 'f158db037c405677ea6ca21a5cb67a800ddc256217c722c0dee0eb31f3f75fb8',
    'model-00004-of-00005.safetensors': 'ece82d2cd5ecc9691572aae55cf2d66fa52edd2c199d5a6dccb8182145bae59c',
    'model-00005-of-00005.safetensors': '1af3bf12ce80cf2f85bacb57a7d2fd584712a945f481cf394f0c7b17983269e5',
}


def download(item):
    name, size = item
    target = DEST / name
    if target.exists() and target.stat().st_size == size:
        return name, 'cached'
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
    subprocess.run([
        'curl', '--fail', '--location', '--silent', '--show-error',
        '--continue-at', '-', '--retry', '15', '--retry-all-errors',
        '--retry-delay', '5', '--connect-timeout', '30',
        '--output', str(target), url,
    ], check=True)
    actual = target.stat().st_size
    if actual != size:
        raise ValueError(f'{name}: expected {size} bytes, found {actual}')
    return name, 'downloaded'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download, item) for item in SIZES.items()]
        for future in as_completed(futures):
            name, state = future.result()
            print(name, state, flush=True)
    for name, expected in WEIGHT_SHA256.items():
        digest = hashlib.sha256()
        with (DEST / name).open('rb') as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'SHA-256 mismatch: {name}')
        print(name, 'sha256 verified', flush=True)
    print('complete', REVISION, DEST.resolve(), flush=True)


if __name__ == '__main__':
    main()
