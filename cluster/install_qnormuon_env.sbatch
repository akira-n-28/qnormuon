#!/bin/bash
# Submit only after staging the complete wheelhouse and assessing quota.
#SBATCH --job-name=qso-offline-install
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/install_qnormuon_env-%j.log
#SBATCH --error=cluster/install_qnormuon_env-%j.log

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Submit this script with sbatch.'
grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup || fail 'Not inside the SLURM job cgroup; refusing frontend execution.'
[[ $(hostname -s) != lagrangectl ]] || fail 'Installation on the frontend is forbidden.'

export QSO_PYTHON=/home/prignano/.venv/bin/python
export QSO_ENV=/home/prignano/.venv
: "${QSO_WHEELHOUSE:?Set QSO_WHEELHOUSE to the absolute shared /home wheelhouse directory}"
[[ "$QSO_WHEELHOUSE" == /home/* && -d "$QSO_WHEELHOUSE" ]] || fail "Wheelhouse absent or not an absolute /home directory: $QSO_WHEELHOUSE"
QSO_WHEELHOUSE=$(realpath -e -- "$QSO_WHEELHOUSE")
export QSO_WHEELHOUSE
[[ "$QSO_WHEELHOUSE" == /home/* ]] || fail 'Wheelhouse resolves outside /home.'
[[ -x "$QSO_PYTHON" ]] || fail "Documented interpreter missing: $QSO_PYTHON"

# Discard inherited pip options, including extra indexes, remote find-links,
# constraints, and target directories. /dev/null disables all pip config files.
for key in ${!PIP_@}; do unset "$key"; done
export PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/home/prignano/.cache/qnormuon
export PIP_CACHE_DIR=/home/prignano/.cache/qnormuon/pip
export TMPDIR=/home/prignano/.cache/qnormuon/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

# Check real paths before creating cache directories (no /dev/shm symlinks).
for path in "$QSO_ENV" "$PIP_CACHE_DIR" "$TMPDIR"; do
    resolved=$(realpath -m -- "$path")
    [[ "$resolved" == /home/* ]] || fail "Storage path resolves outside /home: $path"
done
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
for path in "$QSO_ENV" "$QSO_WHEELHOUSE" "$PIP_CACHE_DIR" "$TMPDIR"; do
    [[ $(findmnt -n -o TARGET -T "$path") == /home ]] || fail "Path is not on the shared /home mount: $path"
done
exec 9>"$XDG_CACHE_HOME/environment.lock"
flock -n 9 || fail 'Another environment install or smoke job holds the environment lock.'
trap 'printf "ERROR: offline installation failed at line %s; inspect this log. No online fallback.\n" "$LINENO" >&2' ERR

hostname
date --iso-8601=seconds
printf 'SLURM_JOB_ID=%s SLURM_JOB_NODELIST=%s CPUS=%s MEMORY_MB=%s\n' "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
pwd
printf 'Python=%s\nWheelhouse=%s\nCache=%s\nTemp=%s\n' "$QSO_PYTHON" "$QSO_WHEELHOUSE" "$PIP_CACHE_DIR" "$TMPDIR"
df -h /home
du -sh -- "$QSO_WHEELHOUSE"
printf 'Quota remains an external capacity prerequisite; df is not a personal quota.\n'

# Validate the complete, flat wheelhouse using stdlib only. Hash every wheel
# before importing the bootstrap pip. No package or project code runs here.
"$QSO_PYTHON" -I - <<'PY'
import email, hashlib, importlib.metadata, os, pathlib, re, sys, zipfile

def fail(message): raise SystemExit('ERROR: ' + message)
def canon(name): return re.sub(r'[-_.]+', '-', name).lower()
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

if sys.version_info[:3] != (3, 10, 20) or sys.prefix != '/home/prignano/.venv':
    fail(f'Expected documented Python 3.10.20 venv; got {sys.version} at {sys.prefix}')
root = pathlib.Path(os.environ['QSO_WHEELHOUSE'])
lock = root / 'requirements.lock'
if not lock.is_file() or lock.is_symlink(): fail('Missing regular requirements.lock in wheelhouse')
alias = root / 'requirements-offline.txt'
if alias.exists():
    if not alias.is_file() or alias.is_symlink() or alias.read_bytes() != lock.read_bytes():
        fail('requirements-offline.txt must be a regular, identical copy of requirements.lock')
pins = {}
pattern = r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)\s+--hash=sha256:([0-9a-fA-F]{64})'
for number, raw in enumerate(lock.read_text().splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith('#'): continue
    match = re.fullmatch(pattern, line)
    if not match: fail(f'Lock line {number}: require one name==version --hash=sha256:HEX; URLs/options/includes/continuations forbidden')
    name, version, sha = match.groups()
    name = canon(name)
    if name in pins: fail(f'Duplicate lock entry: {name}')
    pins[name] = (version, sha.lower())
required = {'pip':'25.3', 'torch':'2.10.0+cu126', 'numpy':'2.2.6', 'pytest':'9.0.2'}
for name, version in required.items():
    if name not in pins or pins[name][0] != version: fail(f'Lock must require {name}=={version}')
print('Required packages, including all locked transitive dependencies:', flush=True)
for name, (version, _) in sorted(pins.items()): print(f'  {name}=={version}', flush=True)

wheels, hashes = {}, {}
for path in sorted(root.iterdir()):
    if path.name in ('requirements.lock', 'requirements-offline.txt', 'SHA256SUMS'): continue
    if path.suffix != '.whl' or not path.is_file() or path.is_symlink():
        fail(f'Wheelhouse must contain only regular wheels, requirements files, and optional SHA256SUMS: {path.name}')
    with zipfile.ZipFile(path) as z:
        entries = [i for i in z.infolist() if i.filename.endswith('.dist-info/METADATA')]
        if len(entries) != 1 or entries[0].file_size > 1024 * 1024: fail(f'Invalid METADATA in {path.name}')
        meta = email.message_from_bytes(z.read(entries[0]))
    name, version = canon(meta['Name']), meta['Version']
    if name in wheels: fail(f'Keep exactly one wheel per distribution: {name}')
    if name not in pins or version != pins[name][0]: fail(f'Wheel not matched by lock: {path.name}')
    for requirement in meta.get_all('Requires-Dist', []):
        if '@' in requirement or '://' in requirement:
            fail(f'Direct URL dependency forbidden in {path.name}: {requirement}')
    sha = digest(path)
    if sha != pins[name][1]: fail(f'Lock SHA256 mismatch: {path.name}')
    wheels[name], hashes[path.name] = path, sha
    print(f'WHEEL_OK {path.name} bytes={path.stat().st_size}', flush=True)
missing = sorted(set(pins) - set(wheels))
if missing: fail('Required wheels missing: ' + ', '.join(missing))
if wheels['pip'].name != 'pip-25.3-py3-none-any.whl': fail('Expected universal bootstrap wheel pip-25.3-py3-none-any.whl')
if wheels['torch'].name != 'torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl':
    fail('Expected the documented CPython 3.10/Linux x86_64/cu126 torch wheel')

manifest = root / 'SHA256SUMS'
if manifest.exists():
    if not manifest.is_file() or manifest.is_symlink(): fail('SHA256SUMS must be a regular file')
    checked = set()
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *]([^/\\]+)', line)
        if not match: fail('Invalid SHA256SUMS line; use sha256sum output with plain basenames')
        sha, name = match.groups()
        if name in checked or name not in {*hashes, 'requirements.lock', 'requirements-offline.txt'}: fail(f'Unexpected/duplicate manifest entry: {name}')
        actual = hashes.get(name) or digest(root / name)
        if sha.lower() != actual: fail(f'SHA256SUMS mismatch: {name}')
        checked.add(name)
    if not set(hashes).issubset(checked): fail('SHA256SUMS must cover every wheel when present')
    print('SHA256SUMS: verified', flush=True)
else:
    print('SHA256SUMS: absent; all wheels verified against mandatory lock hashes', flush=True)

# Do not overwrite an environment that has been repurposed for other work.
for dist in importlib.metadata.distributions():
    name = canon(dist.metadata['Name'])
    if name not in pins or dist.version != pins[name][0]:
        fail(f'Existing environment differs from lock: {name}=={dist.version}; review before changing it')
print('Wheelhouse verification complete; pip will check compatible tags and dependency closure.', flush=True)
PY

# Run pip directly from its verified universal wheel; no ensurepip, uv, or
# network bootstrap is needed even when the target environment has no pip.
offline_pip() {
    "$QSO_PYTHON" -I - "$QSO_WHEELHOUSE/pip-25.3-py3-none-any.whl" "$@" <<'PY'
import runpy, sys
sys.path.insert(0, sys.argv.pop(1))
sys.argv[0] = 'pip'
runpy.run_module('pip', run_name='__main__')
PY
}
pip_args=(install --no-index --find-links "$QSO_WHEELHOUSE" --only-binary=:all:
    --require-hashes --disable-pip-version-check --no-input
    --cache-dir "$PIP_CACHE_DIR" -r "$QSO_WHEELHOUSE/requirements.lock")
if ! offline_pip "${pip_args[@]}" --dry-run --ignore-installed; then
    fail 'Offline dependency preflight failed: missing/incompatible wheel, incomplete lock, or conflicting versions. Installation was not started.'
fi

# A failed installation must not leave an old success marker for smoke tests.
"$QSO_PYTHON" -I - <<'PY'
from pathlib import Path
Path('/home/prignano/.venv/qnormuon-offline-install.json').unlink(missing_ok=True)
PY
offline_pip "${pip_args[@]}"
"$QSO_PYTHON" -I -m pip --disable-pip-version-check check
"$QSO_PYTHON" -I - <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, re, sys
root = pathlib.Path(os.environ['QSO_WHEELHOUSE'])
versions = {re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version for d in importlib.metadata.distributions()}
for line in (root / 'requirements.lock').read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'): continue
    name, version = line.split()[0].split('==')
    name = re.sub(r'[-_.]+', '-', name).lower()
    if versions.get(name) != version: raise SystemExit(f'ERROR: installed version mismatch: {name}')
print('FINAL INSTALLED VERSIONS:', json.dumps(versions, sort_keys=True, indent=2))
record = {'python': sys.version.split()[0], 'environment': sys.prefix,
          'slurm_job_id': os.environ['SLURM_JOB_ID'], 'wheelhouse': str(root),
          'lock_sha256': hashlib.sha256((root / 'requirements.lock').read_bytes()).hexdigest(),
          'versions': versions}
marker = pathlib.Path(sys.prefix) / 'qnormuon-offline-install.json'
temporary = marker.with_suffix('.tmp')
temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
temporary.replace(marker)
print('INSTALLATION_SUCCESS:', marker)
PY
date --iso-8601=seconds
