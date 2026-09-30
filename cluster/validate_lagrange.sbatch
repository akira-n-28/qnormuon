#!/bin/bash
# Submit from the project root: sbatch cluster/validate_lagrange.sbatch
# Requires cluster/frontend_visibility_probe.txt created on the frontend.
# Read-only environment checks; no installs, downloads, tests, or tensor work.
#SBATCH --job-name=qso-env-validation
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/validate_lagrange-%j.log
#SBATCH --error=cluster/validate_lagrange-%j.log

set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

section() { printf '\n## %s\n' "$1"; }
check() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    timeout 20s "$@"
    local result=$?
    printf '[exit=%s]\n' "$result"
}

section 'Identity and allocation'
check hostname
check date --iso-8601=seconds
for key in SLURM_JOB_ID SLURM_JOB_NODELIST CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_MEM_PER_NODE SLURM_SUBMIT_DIR; do
    printf '%s=%s\n' "$key" "${!key-<unset>}"
done
check scontrol show job "$SLURM_JOB_ID"

section 'GPU inventory (management visibility may exceed allocation)'
check nvidia-smi
check nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv

section 'CPU and RAM visibility (host totals are not allocation limits)'
check nproc
check nproc --all
check free -h
check bash -c 'grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status; cat /proc/self/cgroup'
check python3 -B - <<'PY'
from pathlib import Path
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _, controllers, relative = line.split(':', 2)
    bases = [Path('/sys/fs/cgroup')] if not controllers else [Path('/sys/fs/cgroup') / c for c in controllers.split(',')]
    for base in bases:
        path = base / relative.lstrip('/')
        while path == base or base in path.parents:
            for name in ('memory.max', 'memory.high', 'memory.limit_in_bytes', 'cpuset.cpus.effective', 'cpuset.cpus'):
                f = path / name
                if f.is_file():
                    try: print(f'{f}: {f.read_text().strip()}')
                    except OSError as e: print(f'{f}: {e}')
            if path == base: break
            path = path.parent
PY

section 'Project and frontend-created file'
check pwd
project="${SLURM_SUBMIT_DIR:?Submit from the project root}"
check ls -ld "$project" "$project/AGENTS.md" "$project/pyproject.toml"
probe="$project/cluster/frontend_visibility_probe.txt"
probe_ok=0
if [[ -r "$probe" ]]; then
    check cat "$probe"
    check sha256sum "$probe"
    probe_ok=1
else
    printf 'FAIL: frontend probe is not readable: %s\n' "$probe"
fi
check sha256sum "$project/cluster/validate_lagrange.sbatch"

section 'Filesystems and quota queries'
check df -h /home /
check df -hT
check df -i /home /
check findmnt -T /home
check findmnt -T /
check ls -ld /scratch /data /datasets /checkpoints
if command -v quota >/dev/null 2>&1; then
    check quota -s
    check quota -v
else
    printf 'quota command unavailable\n'
fi
for key in QSO_DATA_ROOT QSO_OUTPUT_ROOT QSO_CACHE_ROOT QSO_CHECKPOINT_ROOT; do
    printf '%s=%s\n' "$key" "${!key-<unset>}"
done

section 'Environment and module systems'
for tool in module conda micromamba uv apptainer singularity nvcc python python3; do
    type -a "$tool" 2>&1 || true
done
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/modules/init/bash; do
        if [[ -r "$init" ]]; then
            printf 'Initializing existing module system: %s\n' "$init"
            source "$init"
            break
        fi
    done
fi
if type module >/dev/null 2>&1; then
    module --version 2>&1
    module list 2>&1
    module avail 2>&1
else
    printf 'module unavailable, including standard initialization paths\n'
fi
if command -v conda >/dev/null 2>&1; then check conda env list; fi
if command -v micromamba >/dev/null 2>&1; then check micromamba env list; fi
if command -v uv >/dev/null 2>&1; then check uv --version; fi

section 'CUDA tools and runtime libraries'
check nvcc --version
for compiler in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/cuda/bin/nvcc; do
    if [[ -x "$compiler" ]]; then check "$compiler" --version; fi
done
check bash -c 'ldconfig -p 2>/dev/null | grep -E "libcuda\.so|libcudart\.so|libnvidia-ml\.so"'

section 'Existing Python environments and package imports'
# Bounded discovery: PATH, current venv, known home environments, common
# admin environment roots, and direct environment paths in conda registry.
# Never invoke uv run, create an environment, or resolve/install dependencies.
check python3 -B - <<'PY'
import glob, os, pathlib, shutil, subprocess
home = pathlib.Path.home()
candidates = [shutil.which('python3'), shutil.which('python')]
patterns = [
    str(home / '.venv/bin/python'), str(pathlib.Path.cwd() / '.venv/bin/python'),
    str(home / 'venvs/*/bin/python'), str(home / '.virtualenvs/*/bin/python'),
    str(home / 'miniconda3/bin/python'), str(home / 'anaconda3/bin/python'),
    str(home / 'miniconda3/envs/*/bin/python'), str(home / 'anaconda3/envs/*/bin/python'),
    str(home / '.conda/envs/*/bin/python'), str(home / 'micromamba/envs/*/bin/python'),
    str(home / '.local/share/uv/python/*/bin/python3'),
    '/opt/*/bin/python', '/opt/*/envs/*/bin/python', '/opt/venvs/*/bin/python',
    '/usr/local/bin/python3', '/usr/bin/python3',
]
for pattern in patterns: candidates.extend(sorted(glob.glob(pattern)))
for name in ('VIRTUAL_ENV', 'CONDA_PREFIX'):
    if os.environ.get(name): candidates.append(os.environ[name] + '/bin/python')
registry = home / '.conda/environments.txt'
if registry.is_file():
    candidates.extend(p + '/bin/python' for p in registry.read_text().splitlines() if p)
probe = '''import sys, importlib.util, importlib.metadata as md
print('executable:', sys.executable)
print('version:', sys.version.replace('\\n', ' '))
print('prefix:', sys.prefix)
for name in ('torch', 'numpy', 'pytest'):
    try:
        spec = importlib.util.find_spec(name)
        print(name, 'module:', None if spec is None else spec.origin)
        if spec is None: continue
        try: print(name, 'distribution:', md.version(name))
        except md.PackageNotFoundError: print(name, 'distribution metadata absent')
        module = __import__(name)
        print(name, 'import OK; version:', getattr(module, '__version__', '<unknown>'))
        if name == 'torch': print('torch.version.cuda:', module.version.cuda)
    except Exception as e: print(name, 'ERROR:', type(e).__name__, str(e))
'''
seen = set()
for candidate in candidates:
    if not candidate or not os.access(candidate, os.X_OK): continue
    p = pathlib.Path(candidate)
    # Preserve venv identity even if its executable symlinks to system Python.
    identity = str(p.parent.parent) if (p.parent.parent / 'pyvenv.cfg').exists() else str(p.resolve())
    if identity in seen: continue
    seen.add(identity)
    if len(seen) > 12:
        print('Discovery capped at 12 interpreters; further candidates unresolved')
        break
    print('\nINTERPRETER:', candidate, flush=True)
    try:
        result = subprocess.run([candidate, '-B', '-c', probe], timeout=15, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end='')
        print('exit:', result.returncode)
    except Exception as e: print(type(e).__name__, str(e))
PY

section 'Completion'
printf 'Frontend probe readable: %s\n' "$probe_ok"
date --iso-8601=seconds
[[ "$probe_ok" == 1 ]]
