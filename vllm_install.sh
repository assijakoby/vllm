export HTTP_PROXY=http://proxy-dmz.intel.com:912
export HTTPS_PROXY=http://proxy-dmz.intel.com:912
export NO_PROXY=localhost,127.0.0.1,.svc,.cluster.local,10.0.0.0/8,habana-labs.com,.habana-labs.com,intel.com,.intel.com 
apt update
apt install -y git curl
conda install -y git
VLLM_USE_PRECOMPILED=1 pip install --editable .