uv venv --python 3.8 scripts/libero/.venv
source scripts/libero/.venv/bin/activate
uv pip sync scripts/libero/requirements.txt third_party/openpi/third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e third_party/openpi/packages/openpi-client
uv pip install -e third_party/openpi/third_party/libero
uv pip install robosuite==1.4.1
uv pip install easydict gym gsutil