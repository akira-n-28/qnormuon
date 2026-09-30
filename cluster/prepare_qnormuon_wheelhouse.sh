#!/usr/bin/env bash
# External Ubuntu x86_64 preparation ONLY. No sudo or system pip is used.
# Usage: bash prepare_qnormuon_wheelhouse.sh
# Read-only package verification on Lagrange, INSIDE SLURM:
#   bash prepare_qnormuon_wheelhouse.sh --verify-only /home/prignano/qnormuon-wheelhouse
# This file is self-contained; it does not require a repository checkout.
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# pip is executed from the wheel as zipimport code; it is never installed on
# the preparation PC. The actual interpreter is 3.10.20, including for marker
# evaluation. -I ignores system PYTHONPATH and the user site.
wheel_pip() {
    "$target_python" -I - "$wheelhouse/pip-25.3-py3-none-any.whl" "$@" <<'PY'
import os, runpy, sys
if os.environ.get('QSO_OFFLINE_CHECK') == '1':
    def prohibit_network(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',
                     'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg'}:
            raise RuntimeError('Offline completeness check blocked a network operation: ' + event)
    sys.addaudithook(prohibit_network)
sys.path.insert(0, sys.argv.pop(1))
sys.argv[0] = 'pip'
runpy.run_module('pip', run_name='__main__')
PY
}

verify_wheelhouse() {
    [[ -d "$wheelhouse" ]] || fail "Wheelhouse absent: $wheelhouse"
    [[ -x "$target_python" ]] || fail "Python 3.10.20 interpreter absent: $target_python"
    # Full byte hashes are checked BEFORE executing bootstrap pip.
    "$target_python" -I - "$wheelhouse" <<'PY'
import email, hashlib, pathlib, re, sys, zipfile
def fail(message): raise SystemExit('ERROR: ' + message)
def canon(name): return re.sub(r'[-_.]+', '-', name).lower()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
if sys.version_info[:3] != (3, 10, 20): fail('Verification must use Python 3.10.20, not host Python 3.12')
root = pathlib.Path(sys.argv[1])
allowed = {'requirements-offline.txt', 'requirements.lock', 'SHA256SUMS'}
for path in root.iterdir():
    if path.is_symlink() or not path.is_file() or (path.suffix != '.whl' and path.name not in allowed):
        fail('Unexpected wheelhouse entry: ' + path.name)
for name in allowed:
    if not (root / name).is_file(): fail('Missing ' + name)
if (root/'requirements.lock').read_bytes() != (root/'requirements-offline.txt').read_bytes():
    fail('Requirements files differ')
pins = {}
for line in (root/'requirements-offline.txt').read_text().splitlines():
    match = re.fullmatch(r'([a-z0-9][a-z0-9-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*) --hash=sha256:([0-9a-f]{64})', line)
    if not match: fail('Requirements must contain only exact pins and SHA256 hashes')
    name, version, digest = match.groups()
    if name in pins: fail('Duplicate pin: ' + name)
    pins[name] = (version, digest)
for name, version in {'pip':'25.3', 'torch':'2.10.0+cu126', 'numpy':'2.2.6', 'pytest':'9.0.2'}.items():
    if name not in pins or pins[name][0] != version: fail('Incorrect or missing pin: ' + name)
if 'setuptools' not in pins or 'wheel' not in pins: fail('Project build requirements are missing')
manifest = {}
for line in (root/'SHA256SUMS').read_text().splitlines():
    match = re.fullmatch(r'([0-9a-f]{64})  ([^/\\]+)', line)
    if not match: fail('Invalid SHA256SUMS line')
    digest, name = match.groups()
    if name in manifest: fail('Duplicate checksum: ' + name)
    manifest[name] = digest
expected_files = {p.name for p in root.iterdir()} - {'SHA256SUMS'}
if set(manifest) != expected_files: fail('Checksum manifest does not cover exactly all wheels and both requirements files')
seen = set()
for name, digest in sorted(manifest.items()):
    path = root/name
    if sha(path) != digest: fail('Checksum mismatch: ' + name)
    print('SHA256 OK:', name, flush=True)
    if path.suffix != '.whl': continue
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if i.filename.endswith('.dist-info/METADATA')]
        if len(entries) != 1 or entries[0].file_size > 1024*1024: fail('Invalid wheel metadata: ' + name)
        meta = email.message_from_bytes(archive.read(entries[0]))
    package = canon(meta['Name'])
    if package in seen or pins.get(package) != (meta['Version'], digest): fail('Wheel/pin mismatch: ' + name)
    seen.add(package)
    for req in meta.get_all('Requires-Dist', []):
        if '@' in req or '://' in req: fail('Remote dependency reference in ' + name)
if seen != set(pins): fail('Missing wheels: ' + ', '.join(sorted(set(pins)-seen)))
for name in ['pip-25.3-py3-none-any.whl', 'torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl']:
    if not (root/name).is_file(): fail('Required exact wheel missing: ' + name)
PY
    # No download, installation, or index access. Empty cache, ignore-installed,
    # local find-links, mandatory hashes, and a socket audit guard prevent a
    # warmed cache, installed package, or working Internet from masking gaps.
    QSO_OFFLINE_CHECK=1 wheel_pip install --dry-run --ignore-installed \
        --no-index --find-links "$wheelhouse" --no-cache-dir \
        --only-binary=:all: --require-hashes --disable-pip-version-check \
        --no-input -r "$wheelhouse/requirements-offline.txt"
    printf '\nOFFLINE COMPLETENESS CHECK PASSED (resolution only; no packages installed).\n'
    printf 'Exact pins, including all CUDA and Python dependencies:\n'
    sed 's/ --hash=.*//' "$wheelhouse/requirements-offline.txt"
    printf 'Target Python: CPython 3.10.20; ABI cp310 (plus compatible abi3/pure wheels)\n'
    printf 'Target platform: Linux x86_64, glibc <= 2.39 wheel baseline\n'
    printf 'PyTorch CUDA build: 2.10.0+cu126; official index https://download.pytorch.org/whl/cu126\n'
    du -sh -- "$wheelhouse"
    "$target_python" -I - "$wheelhouse" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print('Wheel files:', len(list(root.glob('*.whl'))))
print('Total file bytes:', sum(p.stat().st_size for p in root.iterdir()))
PY
}

for tool in python3 uname hostname realpath mkdir mktemp sed du; do
    command -v "$tool" >/dev/null || fail "Required tool missing: $tool"
done
for key in ${!PIP_@} ${!UV_@}; do unset "$key"; done
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ${1:-} == --verify-only ]]; then
    [[ $# == 2 ]] || fail 'Usage: --verify-only ABSOLUTE_WHEELHOUSE_PATH'
    wheelhouse=$(realpath -e -- "$2")
    target_python=/home/prignano/.venv/bin/python
    [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'On Lagrange, run verification inside srun/sbatch, not on the frontend.'
    grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup || fail 'Not inside the SLURM job cgroup.'
    [[ $(hostname -s) != lagrangectl ]] || fail 'Do not hash multi-GB wheels on the frontend.'
    export PIP_CACHE_DIR=/home/prignano/.cache/qnormuon/pip
    export TMPDIR=/home/prignano/.cache/qnormuon/tmp
    for path in "$wheelhouse" "$PIP_CACHE_DIR" "$TMPDIR"; do
        [[ $(realpath -m -- "$path") == /home/* ]] || fail "Verification storage must stay on /home: $path"
    done
    mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
    for path in "$wheelhouse" "$TMPDIR" "$PIP_CACHE_DIR"; do
        [[ $(findmnt -n -o TARGET -T "$path") == /home ]] || fail "Not the shared /home mount: $path"
    done
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    verify_wheelhouse
    exit 0
fi
[[ $# == 0 ]] || fail 'Usage: bash prepare_qnormuon_wheelhouse.sh (on the temporary PC)'
case $(hostname -s) in lagrangectl*|lagrange[0-9]*) fail 'Preparation downloads are forbidden on Lagrange; use the temporary Internet-connected PC.';; esac
[[ -z ${SLURM_JOB_ID:-} ]] || fail 'Preparation is for the external PC, not a SLURM job.'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Preparation requires Linux x86_64.'
for tool in wget tar sha256sum; do command -v "$tool" >/dev/null || fail "Required tool missing: $tool"; done
wheelhouse="$PWD/qnormuon-wheelhouse"
[[ ! -e "$wheelhouse" ]] || fail 'qnormuon-wheelhouse already exists; choose a fresh working directory to avoid mixing runs.'
work=$(mktemp -d "$PWD/.qnormuon-wheelhouse-work.XXXXXXXX")
mkdir "$wheelhouse"
trap 'printf "ERROR: preparation failed at line %s. Do not transfer this as a complete wheelhouse. Work retained at %s\n" "$LINENO" "$work" >&2' ERR
export UV_CACHE_DIR="$work/cache" UV_PYTHON_INSTALL_DIR="$work/python"
export UV_PYTHON_BIN_DIR="$work/bin" UV_NO_CONFIG=1
export TMPDIR="$work/tmp" PIP_CACHE_DIR="$work/pip-cache"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$work/bin"
printf 'Private bootstrap/cache: %s\nOutput: %s\n' "$work" "$wheelhouse"

# Pin uv; verify its official release checksum before extracting. No installer
# script, shell profile change, sudo, or system package manager is involved.
uv_version=0.10.11
archive=uv-x86_64-unknown-linux-gnu.tar.gz
release="https://github.com/astral-sh/uv/releases/download/$uv_version"
wget --https-only --timeout=60 --tries=3 -O "$work/$archive" "$release/$archive"
wget --https-only --timeout=60 --tries=3 -O "$work/$archive.sha256" "$release/$archive.sha256"
python3 -I - "$work/$archive" "$work/$archive.sha256" <<'PY'
import hashlib, pathlib, re, sys
archive, checksum = map(pathlib.Path, sys.argv[1:])
expected = checksum.read_text().split()[0].lower()
if not re.fullmatch('[0-9a-f]{64}', expected): raise SystemExit('Invalid uv checksum response')
h = hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
if h.hexdigest() != expected: raise SystemExit('uv archive SHA256 mismatch')
PY
tar -xzf "$work/$archive" -C "$work/bin" --strip-components=1
uv="$work/bin/uv"
"$uv" --version
"$uv" python install --no-config --no-bin 3.10.20
target_python=$("$uv" python find --no-config --managed-python --no-python-downloads 3.10.20)
"$target_python" -I - <<'PY'
import platform, sys
if sys.version_info[:3] != (3,10,20) or platform.machine() != 'x86_64':
    raise SystemExit('Refusing to use a non-target interpreter')
print('Private target interpreter:', sys.executable, sys.version)
print('Host libc:', platform.libc_ver())
PY

# Obtain only the universal pip wheel via small PyPI metadata and HTTPS wget.
# System Python supplies stdlib only. No system pip or package installation.
python3 -I - "$wheelhouse" <<'PY'
import hashlib, json, pathlib, subprocess, sys, urllib.request, urllib.parse
root = pathlib.Path(sys.argv[1])
with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=60) as r:
    data = json.loads(r.read(1024*1024))
items = [f for f in data['urls'] if f['filename'] == 'pip-25.3-py3-none-any.whl']
if len(items) != 1: raise SystemExit('Exact pip bootstrap wheel not found')
item = items[0]
url = urllib.parse.urlsplit(item['url'])
if url.scheme != 'https' or url.hostname != 'files.pythonhosted.org':
    raise SystemExit('Unexpected pip wheel source')
path = root/item['filename']
subprocess.run(['wget','--https-only','--timeout=60','--tries=3','-O',str(path),item['url']], check=True)
if hashlib.sha256(path.read_bytes()).hexdigest() != item['digests']['sha256']:
    raise SystemExit('pip wheel SHA256 mismatch')
PY

# Explicit wheel tags target Lagrange's glibc 2.39, regardless of the temporary
# machine's system Python. Running pip under real 3.10.20 also makes conditional
# Requires-Dist markers use Python 3.10 (cross-tag flags alone are insufficient).
target_flags=(--python-version 3.10.20 --implementation cp --abi cp310 --abi abi3 --abi none)
for minor in {39..5}; do target_flags+=(--platform "manylinux_2_${minor}_x86_64"); done
target_flags+=(--platform manylinux2014_x86_64 --platform manylinux2010_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64)
wheel_pip download --no-deps --only-binary=:all: "${target_flags[@]}" \
    --index-url https://download.pytorch.org/whl/cu126 \
    --dest "$wheelhouse" 'torch==2.10.0+cu126'
[[ -f "$wheelhouse/torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl" ]] || fail 'Wrong torch artifact; exact cp310/cu126 wheel required.'

# Local torch and pip references preserve their sources. Resolve EVERY runtime
# dependency, including NVIDIA libraries and Triton, as well as project build
# requirements. Do not use --no-deps for this operation.
wheel_pip download --only-binary=:all: "${target_flags[@]}" \
    --index-url https://pypi.org/simple --find-links "$wheelhouse" --dest "$wheelhouse" \
    "$wheelhouse/torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl" \
    "$wheelhouse/pip-25.3-py3-none-any.whl" \
    'numpy==2.2.6' 'pytest==9.0.2' 'setuptools>=68' wheel

# Write one exact pin/hash per wheel. Both requirements names are intentionally
# identical so the existing Lagrange installer can continue using its lock name.
"$target_python" -I - "$wheelhouse" <<'PY'
import email, hashlib, pathlib, re, sys, zipfile
root = pathlib.Path(sys.argv[1])
entries, sums = {}, {}
for path in sorted(root.glob('*.whl')):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1: raise SystemExit('Ambiguous wheel metadata: ' + path.name)
        meta = email.message_from_bytes(archive.read(names[0]))
    name = re.sub(r'[-_.]+', '-', meta['Name']).lower()
    if name in entries: raise SystemExit('Duplicate package wheel: ' + name)
    for requirement in meta.get_all('Requires-Dist', []):
        if '@' in requirement or '://' in requirement: raise SystemExit('Remote dependency reference: ' + path.name)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    sums[path.name] = h.hexdigest()
    entries[name] = f"{name}=={meta['Version']} --hash=sha256:{h.hexdigest()}"
payload = '\n'.join(entries[name] for name in sorted(entries)) + '\n'
for name in ['requirements-offline.txt', 'requirements.lock']:
    (root/name).write_text(payload)
    sums[name] = hashlib.sha256(payload.encode()).hexdigest()
(root/'SHA256SUMS').write_text(''.join(f'{sums[name]}  {name}\n' for name in sorted(sums)))
PY
verify_wheelhouse
printf '\nREADY TO TRANSFER: %s\n' "$wheelhouse"
printf 'Keep the temporary PC powered on until transferred files pass --verify-only on Lagrange.\n'
printf 'Private bootstrap files remain at %s; transfer only qnormuon-wheelhouse/.\n' "$work"
