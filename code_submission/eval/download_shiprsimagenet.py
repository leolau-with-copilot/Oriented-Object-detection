# 下载shiprsimagenet数据集
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

REPO = 'insomnia7/ShipRSImageNet'
FILE = 'ShipRSImageNet_V1.zip'
DEST = os.path.join(paths.DATA, 'ShipRSImageNet')          # extraction dir

print(f'downloading {FILE} (~4.6 GB, resumable)...', flush=True)
zip_path = hf_hub_download(repo_id=REPO, filename=FILE, repo_type='dataset')
print(f'downloaded to: {zip_path}', flush=True)

print(f'extracting to {DEST}/ ...', flush=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DEST)
print('done. To reclaim space you can delete the HF cache copy:')
print(f'   rm "{zip_path}"')
