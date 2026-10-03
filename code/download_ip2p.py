import os
import tempfile

# redirect temp dir to home2 which has plenty of space
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HOME'] = '/home2/muskan.singh/hf_cache'
os.environ['TMPDIR'] = '/home2/muskan.singh/tmp'
tempfile.tempdir = '/home2/muskan.singh/tmp'

# create tmp dir
os.makedirs('/home2/muskan.singh/tmp', exist_ok=True)

from huggingface_hub import snapshot_download

print("Downloading instruct-pix2pix...")
path = snapshot_download(
    repo_id="timbrooks/instruct-pix2pix",
    cache_dir="/home2/muskan.singh/hf_cache",
    ignore_patterns=["*.msgpack", "*.h5"],
)
print(f"Downloaded to: {path}")
