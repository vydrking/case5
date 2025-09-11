import os
import zipfile
import tarfile
import shutil
import subprocess
from patoolib import extract_archive

def extract_any_archive(src_path: str, out_dir: str) -> None:
    lower = src_path.lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(src_path) as zf:
            zf.extractall(out_dir)
        return
    if lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')):
        mode = 'r'
        if lower.endswith(('.tar.gz', '.tgz')):
            mode = 'r:gz'
        elif lower.endswith(('.tar.bz2', '.tbz2')):
            mode = 'r:bz2'
        elif lower.endswith(('.tar.xz', '.txz')):
            mode = 'r:xz'
        with tarfile.open(src_path, mode) as tf:
            tf.extractall(out_dir)
        return
    if lower.endswith('.7z') and shutil.which('7z'):
        subprocess.run(['7z', 'x', '-y', f'-o{out_dir}', src_path], check=True)
        return
    if lower.endswith('.rar'):
        if shutil.which('unrar'):
            subprocess.run(['unrar', 'x', '-y', src_path, out_dir], check=True)
            return
        if shutil.which('unar'):
            subprocess.run(['unar', '-quiet', '-output-directory', out_dir, src_path], check=True)
            return
    extract_archive(src_path, outdir=out_dir)


