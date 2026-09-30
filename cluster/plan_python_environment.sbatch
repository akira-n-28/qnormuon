#!/bin/bash
# Environment planning only: no environment creation or package installation.
# Submit from /home/prignano/qnormuon.
#SBATCH --job-name=qso-python-plan
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/plan_python_environment-%j.log
#SBATCH --error=cluster/plan_python_environment-%j.log

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
hostname
date --iso-8601=seconds
for key in SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$key" "${!key-<unset>}"
done
pwd
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /home /
df -i /home
findmnt -T /home/prignano
readlink -f /usr/local/cuda
/usr/local/cuda-12.6/bin/nvcc --version
/home/prignano/.local/bin/uv --version

/home/prignano/.venv/bin/python -B - <<'PY'
import email, hashlib, html.parser, importlib.metadata, json, os, pathlib
import platform, ssl, sys, tempfile, urllib.parse, urllib.request

print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Prefix:', sys.prefix, 'base prefix:', sys.base_prefix)
print('Architecture/libc:', platform.machine(), platform.libc_ver())
print('TLS:', ssl.OPENSSL_VERSION)
print('Existing distributions:', sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
for path in ['/home/prignano', '/home/prignano/.venv', '/home/prignano/.cache']:
    p = pathlib.Path(path)
    print('Storage path:', path, 'exists:', p.exists(), 'writable:', os.access(p, os.W_OK))

# Only transient, tiny files; no cache or virtual environment is created.
with tempfile.TemporaryDirectory(prefix='.qso-storage-probe-', dir='/home/prignano') as tmp:
    p = pathlib.Path(tmp) / 'probe'
    p.write_bytes(b'qso home write test\n')
    with p.open('rb') as f: os.fsync(f.fileno())
    assert p.read_bytes() == b'qso home write test\n'
    print('HOME_WRITE_READ_OK:', p)
    linked = pathlib.Path(tmp) / 'hardlink'
    try:
        os.link(p, linked)
        print('HOME_HARDLINK_OK:', p.stat().st_ino == linked.stat().st_ino)
    except OSError as e:
        print('HOME_HARDLINK_FAILED:', type(e).__name__, str(e))
print('Temporary probe removed')

print('Quota-related xattrs (absence or permission failure does not establish no quota):')
for key in ['trusted.glusterfs.quota.limit-set', 'trusted.glusterfs.quota.size']:
    try: print(key, os.getxattr('/home/prignano', key).hex())
    except OSError as e: print(key, type(e).__name__, str(e))

# HTTPS metadata only. Never GET a wheel/archive. Bound each response body and
# total payload; use default certificate verification and HTTPS-only redirects.
class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise RuntimeError('Refusing non-HTTPS redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)
opener = urllib.request.build_opener(HTTPSOnly())
total_bytes = 0
def metadata(url, method='GET'):
    global total_bytes
    if method == 'GET' and not (url.endswith('/') or url.endswith('/json') or url.endswith('.metadata')):
        raise RuntimeError('Only metadata GETs permitted')
    req = urllib.request.Request(url, method=method, headers={'User-Agent': 'qnormuon-environment-plan/1.0', 'Accept-Encoding': 'identity'})
    with opener.open(req, timeout=12) as r:
        length = r.headers.get('Content-Length')
        print('HTTPS:', method, url, 'status:', r.status, 'final:', r.url, 'content-length:', length)
        if method == 'HEAD': return b''
        cap = min(262144, 1048576-total_bytes)
        if length is not None and int(length) > cap:
            raise RuntimeError('Metadata length exceeds payload budget')
        data = r.read(cap + 1)
        total_bytes += len(data)
        if len(data) > cap: raise RuntimeError('Metadata body exceeds payload budget')
        print('Metadata bytes read:', len(data))
        return data

class Links(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'a': self.links.append(dict(attrs))

try:
    index = 'https://download.pytorch.org/whl/cu126/torch/'
    parser = Links()
    parser.feed(metadata(index).decode())
    filename = 'torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl'
    matches = [a for a in parser.links if urllib.parse.unquote(urllib.parse.urlsplit(a.get('href','')).path).endswith('/'+filename)]
    if len(matches) != 1: raise RuntimeError(f'Expected exactly one matching torch wheel; found {len(matches)}')
    link = urllib.parse.urljoin(index, matches[0]['href'])
    print('SELECTED_TORCH_WHEEL:', link)
    print('TORCH_INDEX_ATTRIBUTES:', matches[0])
    wheel, _ = urllib.parse.urldefrag(link)
    metadata(wheel, 'HEAD')  # headers only, no binary body
    if matches[0].get('data-dist-info-metadata') or matches[0].get('data-core-metadata'):
        raw = metadata(wheel + '.metadata')
        print('Torch metadata SHA256:', hashlib.sha256(raw).hexdigest())
        msg = email.message_from_bytes(raw)
        for key in ['Name','Version','Requires-Python','Requires-Dist']:
            for value in msg.get_all(key, []): print('TORCH_METADATA:', key, value)
    else:
        print('No metadata sidecar advertised; wheel was NOT downloaded')
except Exception as e:
    print('TORCH_METADATA_CHECK_FAILED:', type(e).__name__, str(e))

for name, version in [('numpy','2.2.6'), ('pytest','9.0.2')]:
    try:
        obj = json.loads(metadata(f'https://pypi.org/pypi/{name}/{version}/json'))
        print('PYPI:', name, version, 'Requires-Python:', obj['info']['requires_python'])
        for f in obj['urls']:
            fn = f['filename']
            suitable = ('cp310-cp310-manylinux' in fn and 'x86_64' in fn) if name == 'numpy' else fn.endswith('py3-none-any.whl')
            if suitable:
                print('PYPI_WHEEL:', fn, 'size:', f['size'], 'sha256:', f['digests']['sha256'], 'url:', f['url'])
                metadata(f['url'], 'HEAD')
    except Exception as e:
        print('PYPI_METADATA_CHECK_FAILED:', name, type(e).__name__, str(e))
print('TOTAL_METADATA_BODY_BYTES:', total_bytes)
PY
date --iso-8601=seconds
