"""创建检查包，不使用 git 文件列表，不修改已有数据或实验产物。"""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'Project1_review.zip'
LISTING = ROOT / 'Project1_review_files.txt'
EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', 'mpl_cache', '.pytest_cache',
                 '.mypy_cache', '.ruff_cache', 'feature_cache', 'feature_caches'}
SOURCE_EXTENSIONS = {'.py', '.ipynb', '.ps1', '.sh', '.bat', '.cmd', '.r', '.R'}
RESULT_EXTENSIONS = {'.csv', '.json', '.jsonl', '.log', '.txt', '.md', '.png', '.svg', '.jpg', '.jpeg', '.pdf',
                     '.yaml', '.yml', '.toml', '.ini', '.cfg'}
ROOT_FILES = {'README.md', 'requirements.txt', '.gitignore', 'TUNING.md',
              'train_data.csv', 'test_data_unlabeled.csv', 'prediction.csv',
              'predictions.csv', '实验一要求.pdf'}
EXCLUDED_EXTENSIONS = {'.zip', '.joblib', '.pkl', '.pickle', '.pt', '.pth', '.onnx',
                       '.npy', '.npz', '.h5', '.hdf5', '.bin', '.pyc', '.pyo'}

def selected(path):
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
        return False
    if path.suffix.lower() in EXCLUDED_EXTENSIONS or path.is_symlink():
        return False
    if path.suffix in SOURCE_EXTENSIONS:
        return True
    if len(relative.parts) == 1:
        return path.name in ROOT_FILES
    # 同时保留正式结果、审计失败记录/快照与旧解析结果，便于检查修正过程。
    return relative.parts[0] in {'results', 'results_initial_parser'} and path.suffix.lower() in RESULT_EXTENSIONS

def main():
    if ARCHIVE.exists() or LISTING.exists():
        raise FileExistsError('检查包或清单已存在；为保留现有文件，不自动覆盖。')
    files = sorted((p for p in ROOT.rglob('*') if p.is_file() and selected(p)),
                   key=lambda p: p.relative_to(ROOT).as_posix())
    names = [p.relative_to(ROOT).as_posix() for p in files]
    assert ROOT_FILES.issubset(set(names))
    assert {'main.py', 'leakage_audit.py', 'verify_results.py', 'pack_review.py',
            'results/leakage_audit.md', 'results/experiments.csv',
            'results/best_validation_predictions.csv'}.issubset(set(names))
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in zip(names, files)}
    with zipfile.ZipFile(ARCHIVE, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, name in zip(files, names):
            z.write(path, arcname=name)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.namelist() == names
        assert z.testzip() is None
        for name, path in zip(names, files):
            assert hashlib.sha256(z.read(name)).hexdigest() == before[name]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before[name], f'原文件发生变化：{name}'
        total = sum(i.file_size for i in z.infolist())
        lines = [f'Archive: {ARCHIVE}', f'Archive bytes: {ARCHIVE.stat().st_size}',
                 f'Files: {len(names)}', f'Uncompressed bytes: {total}',
                 'Filter: explicit root files + project source extensions + results evidence extensions.',
                 'Excluded: environments, Python/Git/font/feature caches, archives, serialized models and array caches.',
                 'Original files verified unchanged; every archive entry SHA256 matches its source.',
                 '', 'Relative path\tUncompressed bytes\tSHA256']
        lines += [f'{i.filename}\t{i.file_size}\t{before[i.filename]}' for i in z.infolist()]
    with LISTING.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Archive: {ARCHIVE}\nBytes: {ARCHIVE.stat().st_size}\nMiB: {ARCHIVE.stat().st_size / 1048576:.2f}\nFiles: {len(names)}\nListing: {LISTING}')

if __name__ == '__main__':
    main()
