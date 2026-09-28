# Offline Python installation and H100 smoke checks

These are **prepared instructions, not executed results**. The installer and
smoke scripts have only been syntax-checked. No package was installed, wheel
downloaded, job submitted, or optimizer code changed as part of this task.

This procedure implements the offline plan in
[LAGRANGE_PYTHON_ENV.md](LAGRANGE_PYTHON_ENV.md) using **pip**, as now requested.
It supersedes that plan's proposed uv installation commands. It does not create
a virtual environment or install the repository as a package.

## Fixed environment and storage

| Purpose | Path / value |
|---|---|
| Existing interpreter | `/home/prignano/.venv/bin/python`, exactly Python 3.10.20 |
| Target environment | `/home/prignano/.venv` |
| Required configuration | `QSO_WHEELHOUSE`: absolute path on shared `/home` |
| Suggested wheelhouse | `/home/prignano/qso-wheelhouse/py310-cu126` |
| pip cache | `/home/prignano/.cache/qnormuon/pip` |
| Temporary extraction | `/home/prignano/.cache/qnormuon/tmp` |
| CUDA/Torch/Triton caches | `/home/prignano/.cache/qnormuon/{cuda,torch,triton}` |
| Environment lock | `/home/prignano/.cache/qnormuon/environment.lock` |
| Successful-install record | `/home/prignano/.venv/qnormuon-offline-install.json` |

Both scripts require a numeric SLURM job ID and the observed Lagrange job cgroup
layout, and explicitly reject `lagrangectl`. They check resolved storage paths
and the `/home` mount. Run them with `sbatch` from the project root, where the
`cluster/` log directory already exists. Do not run them as frontend shell jobs.

The installer requests **2 CPUs, 16 GiB, 20 minutes, no GPU**: local wheel
installation and metadata checks do not require a GPU. The smoke script requests
**1 GPU, 2 CPUs, 4 GiB, 5 minutes**, all on partition `mat`.

Quota remains unknown from the earlier inspection. Before staging multi-GB
payloads, establish sufficient user/project capacity for the wheelhouse,
environment, and temporary space. Filesystem-wide free space does not establish
personal quota. Unlike the earlier uv proposal, pip installation generally
requires an installed copy in addition to retained compressed wheel files; this
procedure makes no hard-link storage-saving assumption. No quotas are invented
or changed by these scripts.

## Required wheelhouse contents

Use a dedicated **flat directory**, containing only regular `.whl` files,
`requirements.lock`, an optional identical `requirements-offline.txt`, and
optional `SHA256SUMS`. No subdirectories, wheel symlinks,
source archives, HTML find-links pages, or unrelated files are accepted.

Required top-level pins:

```text
pip==25.3
torch==2.10.0+cu126
numpy==2.2.6
pytest==9.0.2
```

The exact bootstrap wheel is `pip-25.3-py3-none-any.whl`. The existing environment
has no pip; the script executes this verified wheel directly with the documented
Python interpreter, using zip import, then installs pip with the other locked
packages. It does not use online bootstrap scripts, ensurepip, or uv downloads.

The exact torch artifact must be:

```text
torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
```

Also supply **every transitive dependency**, including the NVIDIA runtime
packages, Triton, and Python dependencies selected by this torch release and
the NumPy/pytest/pip pins. The installed system CUDA toolkit does not replace
these dependencies. Let a resolver on the matching preparation platform determine
the actual names and versions; do not substitute a guessed dependency list.
Keep exactly one compatible wheel for each distribution.

`requirements.lock` must contain the complete resolved set, including pip and
all four top-level pins, in this deliberately restricted format:

```text
distribution-name==exact-version --hash=sha256:ACTUAL_64_HEX_DIGIT_WHEEL_HASH
```

Use one package and one selected wheel hash per physical line. Blank lines and
whole-line `#` comments are allowed. URLs, direct references, markers, extras,
included requirements files, pip options, editable installs, and line
continuations are not accepted. This is a target-specific lock for Linux
x86_64/CPython 3.10, not a universal lock across platforms. The example hash above
is a format description; generate real hashes from the prepared wheel files.

The installer prints all locked requirements, reports `du -sh` and `df -h`, and
verifies every wheel against its lock hash before executing bootstrap pip. It
reads wheel metadata to reject direct-URL dependencies. If `SHA256SUMS` exists,
it must use standard `sha256sum` format with plain filenames, cover every wheel,
and may also cover either requirements filename. Any mismatch fails. Without that optional
file, mandatory per-wheel lock hashes are still checked.

Hashes detect corruption relative to a trusted manifest; a manifest bundled with
untrusted wheels does not itself establish provenance. Stage artifacts from the
selected official sources through an approved process and retain their provenance.

## Expected external preparation (operator procedure, not executed here)

For the temporary Ubuntu 24.04 x86_64 PC with Python 3.12 and restricted system
pip, use the self-contained
[prepare_qnormuon_wheelhouse.sh](../cluster/prepare_qnormuon_wheelhouse.sh).
Run `bash prepare_qnormuon_wheelhouse.sh` in a fresh working directory on that PC.
It uses a checksum-verified uv 0.10.11 bootstrap to unpack private CPython 3.10.20,
then runs a verified pip 25.3 wheel without installing pip or touching system
Python. Package markers are evaluated by actual Python 3.10.20, and wheel tags
are explicitly constrained to CPython 3.10/Linux x86_64 up to glibc 2.39.
Neither a GPU nor CUDA installation is needed on the preparation PC.

The script downloads the complete dependency graph plus the project's declared
build requirements (`setuptools>=68`, `wheel`), which have no pre-existing exact
pins. All selected versions are frozen into both requirements filenames. It
creates `qnormuon-wheelhouse/` in the current directory and refuses to merge into
an existing directory. Private uv/Python/cache files remain separately under a
`.qnormuon-wheelhouse-work.*` directory and are not part of the transfer.

Its offline completeness check hashes every wheel, validates the lock, and runs
pip resolution with `--dry-run --ignore-installed --no-index --find-links`,
mandatory hashes, no cache, and a Python audit hook that rejects network socket
operations. No torch installation or import occurs. Successful resolution is
not an H100 execution test. Preparation has not been executed by Codex.

From the temporary PC, fetch the small preparation script, run it, then transfer
only the completed wheelhouse after `OFFLINE COMPLETENESS CHECK PASSED`:

```bash
scp prignano@lagrangectl.mat.uniroma1.it:/home/prignano/qnormuon/cluster/prepare_qnormuon_wheelhouse.sh .
bash prepare_qnormuon_wheelhouse.sh
scp -r qnormuon-wheelhouse prignano@lagrangectl.mat.uniroma1.it:/home/prignano/
```

Use a fresh destination on Lagrange; do not merge independent wheelhouse runs.
Check that `/home/prignano/qnormuon-wheelhouse` does not already exist before
starting a new transfer. Confirm capacity before transferring multi-GB assets.

**Verify the transferred wheelhouse on Lagrange before powering off the ephemeral
PC.** Run the following from that PC; it uses a CPU SLURM allocation to avoid
hashing multi-GB payloads on the frontend. It downloads and installs nothing:

```bash
ssh prignano@lagrangectl.mat.uniroma1.it \
  'srun --partition=mat --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:20:00 bash /home/prignano/qnormuon/cluster/prepare_qnormuon_wheelhouse.sh --verify-only /home/prignano/qnormuon-wheelhouse'
```

Require exit status 0, a `SHA256 OK` for every wheel and both requirements files,
and `OFFLINE COMPLETENESS CHECK PASSED`. The printed wheel count and total file
bytes must match the preparation output. If any check fails, keep the PC running,
correct/retransfer the files, and repeat verification. The target environment
remains unmodified. For a later separately authorized installation, set
`QSO_WHEELHOUSE=/home/prignano/qnormuon-wheelhouse`.

The manual alternative below assumes an already available matching preparation
interpreter; it is not required when using the supplied script.

Use an approved connected preparation environment with **CPython 3.10, Linux
x86_64, and compatible glibc**, or an administrator-managed equivalent. Availability
of such a machine or transfer mechanism is not assumed. Do not use the Lagrange
frontend for large downloads. Coordinate staging onto shared `/home` after the
quota/capacity check; no transfer is initiated by these instructions.

In an isolated preparation environment with modern pip, choose a new, empty
wheelhouse directory. Example acquisition commands for that external environment:

```bash
# Preparation system only. WHEELHOUSE must be explicitly selected there.
: "${WHEELHOUSE:?Choose an empty preparation wheelhouse directory}"
mkdir -p "$WHEELHOUSE"

# Obtain the exact CUDA variant from its official index, without dependencies.
python3.10 -m pip download --only-binary=:all: --no-deps \
  --index-url https://download.pytorch.org/whl/cu126 \
  --dest "$WHEELHOUSE" 'torch==2.10.0+cu126'

# Resolve the complete graph using that local torch wheel and PyPI for others.
python3.10 -m pip download --only-binary=:all: \
  --index-url https://pypi.org/simple --find-links "$WHEELHOUSE" \
  --dest "$WHEELHOUSE" \
  'pip==25.3' 'torch==2.10.0+cu126' 'numpy==2.2.6' 'pytest==9.0.2'
```

Use a clean preparation configuration; do not inject extra indexes or requirements
through environment variables or pip configuration. Never execute these download
commands on the Lagrange frontend or inside either supplied job script.

Generate the restricted lock and optional checksum file on the preparation
system. This example reads metadata and streams hashes without extracting wheels:

```python
import email
import hashlib
import os
from pathlib import Path
import re
import zipfile

root = Path(os.environ["WHEELHOUSE"])
entries, checksums = {}, {}
for path in sorted(root.glob("*.whl")):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise RuntimeError(f"Ambiguous metadata: {path.name}")
        meta = email.message_from_bytes(archive.read(names[0]))
    name = re.sub(r"[-_.]+", "-", meta["Name"]).lower()
    if name in entries:
        raise RuntimeError(f"More than one wheel for {name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    checksums[path.name] = digest.hexdigest()
    entries[name] = f"{name}=={meta['Version']} --hash=sha256:{digest.hexdigest()}"
lock = root / "requirements.lock"
lock.write_text("\n".join(entries[k] for k in sorted(entries)) + "\n")
checksums[lock.name] = hashlib.sha256(lock.read_bytes()).hexdigest()
(root / "SHA256SUMS").write_text(
    "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums))
)
```

Confirm the complete wheelhouse installs with no network in a disposable matching
preparation environment before staging. Preserve exact filenames, the lock,
and checksums. A missing binary wheel must be resolved during preparation, not by
allowing source builds or online fallback on Lagrange.

## Offline installation on Lagrange (future execution only)

After the wheelhouse is actually present and capacity assessed:

```bash
cd /home/prignano/qnormuon
export QSO_WHEELHOUSE=/home/prignano/qso-wheelhouse/py310-cu126
test -d "$QSO_WHEELHOUSE"
test -s "$QSO_WHEELHOUSE/requirements.lock"
sbatch --export=ALL,QSO_WHEELHOUSE cluster/install_qnormuon_env.sbatch
```

The script fails before bootstrap pip runs if the wheelhouse is absent,
malformed, missing a locked wheel, or fails hash checks. It checks Python version
and refuses to overwrite an environment containing packages that differ from the
lock. It uses pip with **`--no-index --find-links "$QSO_WHEELHOUSE"`**,
`--only-binary=:all:` and `--require-hashes` for both dependency preflight and
installation.

Preflight adds `--dry-run --ignore-installed`, so existing packages cannot hide an
incomplete wheelhouse. Missing transitive wheels, unsupported wheel tags,
incomplete locks, and dependency conflicts cause a clear nonzero exit before
package installation starts. Cache metadata and temporary files may still be
created during preflight.

After installation, `pip check` must pass, all locked versions are compared with
installed metadata, and the script prints the final package inventory. Only then
does it atomically write the successful-install record. No GPU workload or full
mathematical suite runs in this job. Inspect its log and accounting state before
proceeding; a submitted job is not evidence of successful installation.

## H100 smoke procedure (future execution only)

Submit after a successful install, or use a SLURM `afterok` dependency on the
actual installation job ID:

```bash
cd /home/prignano/qnormuon
sbatch cluster/smoke_h100.sbatch
# Alternatively, replace INSTALL_JOB_ID with the actual numeric job ID:
# sbatch --dependency=afterok:INSTALL_JOB_ID cluster/smoke_h100.sbatch
```

The smoke job requires the successful-install record and compares its package
inventory with the current environment before importing torch. A shared lock
prevents it from racing the installer. It reports torch version, built CUDA
version, CUDA availability, device name/properties, compiled architectures, and
provenance. It requires an H100 and the documented CUDA 12.6 torch build.

It then checks a tiny CUDA allocation, matrix multiplication for each requested
dtype, and native `torch.linalg.svd(..., full_matrices=False)` independently for
**float64, float32, and bfloat16**. Each operation logs `PASS` or `FAIL`, with the
exception or measured reconstruction error. SVD input is never silently cast to
a different dtype. Float64 casts used only for diagnostics are explicit.

The polar check uses `U @ Vh` from the same default reduced-SVD call as
`experiments/dual_solver.py`, on a deterministic `(2, 16, 8)` full-column-rank
pair in float64 and float32. It checks finiteness, Gram error, and the nuclear
support identity. These are full-rank backend checks, not a coupled LMO solve or
a rank-deficient selection test. No claim about zero momentum or quotient
certification follows from them.

Baseline success requires allocation, all three matrix-multiply checks, and the
float64/float32 SVD and polar checks. **Bfloat16 SVD is a capability probe**:
its failure is printed explicitly and does not alone fail the baseline job.
The concluding line distinguishes `BFLOAT16_NATIVE_SVD_FAILED` from
`BFLOAT16_NATIVE_SVD_PASSED`. Inspect the exact exception: the script does not
automatically classify every BF16 failure as expected unsupported behavior.
Any required check failing produces a nonzero job exit. Matrix multiplication
success never implies bfloat16 SVD support.

## Failure recovery and no-network behavior

- **Absent/incomplete wheelhouse:** stage the missing compatible wheels through
  the approved preparation process, regenerate the complete lock/manifest, then
  resubmit. There is no online retry.
- **Hash mismatch:** stop and verify the staged payload against the trusted
  source. Do not disable hashes merely to proceed.
- **Existing environment differs:** inspect its use and versions. The installer
  does not delete or automatically reset it. Restore deliberately in a separate
  maintenance task or choose a separately reviewed environment strategy.
- **Quota, disk-full, or interrupted installation:** obtain capacity and inspect
  the log. pip installation is not transactional. The success marker is removed
  immediately before mutation, so an incomplete install cannot authorize the
  smoke job. A rerun can complete a partial environment whose installed packages
  still match the lock; conflicts require review.
- **Lock busy:** wait for the other environment job; do not bypass locking.
- **CUDA or required numerical failure:** retain logs and inspect the allocation,
  driver/build versions, and exact error. Do not run training or the full suite
  on the strength of this infrastructure alone.
- **Different cgroup layout:** the scripts fail closed. Reinspect the scheduler
  layout before changing the guard; never remove it to run on the frontend.

The installation path has no package-network fallback: inherited `PIP_*`
variables are cleared; `PIP_CONFIG_FILE=/dev/null` disables pip configuration;
index access and version checks are disabled; find-links is a verified local
directory; locks and wheel dependency metadata cannot contain remote references;
and source builds are forbidden. Python runs with `-I` to ignore `PYTHONPATH` and
user site packages. The scripts contain no package download commands, and the
smoke script uses only installed packages and local tensors.

This is an application-level no-network installation policy, not an OS network
namespace or protection against malicious third-party wheel code. Only trusted
prepared wheels should be executed. No network access is needed for normal
installation or smoke execution, and no undocumented large cache, frontend
temporary directory, or RAM-backed `/dev/shm` path is used.
