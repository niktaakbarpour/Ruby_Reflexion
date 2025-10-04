#!/bin/bash
#SBATCH --job-name=lantern-run
#SBATCH --account=rrg-fard_gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:h100
#SBATCH --output=lantern_output.txt
#SBATCH --error=lantern_error.txt

set -e  # Exit on error

# === Load modules ===
module purge
module load gcc/13.3
module load cuda/12.6
module load python/3.11
module load qt/5.15.11
module load arrow/19.0.1
module load apptainer
module load rust/1.85.0
module load nodejs
module load java
module load go
module list

# sanity checks
which rustc && rustc --version
which python && python -V

# === Define paths ===
PROJECT_DIR=/project/def-fard/niktakbr/LANTERN
VENV_DIR=$PROJECT_DIR/.venv
REQUIREMENTS=$PROJECT_DIR/requirements.txt
WHEELHOUSE=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
PYTHON_PKGS=/project/def-fard/niktakbr/python_pkgs
PYARROW_WHLS=$PROJECT_DIR/pyarrow_pkg

cd $PROJECT_DIR

# === Create virtual environment if it doesn't exist ===
if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
fi

# === Activate venv ===
source $VENV_DIR/bin/activate


# Remove any pip-installed version so it doesn't shadow your local one
pip uninstall -y promptsource || true

# Force Python to import our local repo first
export PYTHONPATH="$PROJECT_DIR/promptsource:${PYTHONPATH:+$PYTHONPATH}"


# === Install specific packages ===
pip install \
  streamlit==1.28.2+computecanada \
  pillow==9.5.0 \
  watchdog==2.1.5 \
  --no-index \
  -f $PYTHON_PKGS \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/ \
  | tee install_basic.log

pip install --no-index --no-deps \
  -f $PYTHON_PKGS \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4 \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  accelerate==0.30.1+computecanada | tee install_accelerate.log

# === Install main requirements ===
pip install -r $REQUIREMENTS \
  --no-index \
  --find-links=$WHEELHOUSE \
  --find-links=$PYTHON_PKGS \
  | tee install_requirements.log

# === Ensure pynvml is present (needed by main.py) ===
pip install pynvml \
  --no-index \
  -f $WHEELHOUSE \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/ \
  | tee install_pynvml.log

# Before starting engine / shims
mkdir -p /project/def-fard/niktakbr/tmp_root \
         /project/def-fard/niktakbr/tmp_root/code_store \
         "$PROJECT_DIR/shims"


# === Create a prlimit shim that ignores --rss and never aborts ===
SHIMS_DIR="$PROJECT_DIR/shims"
mkdir -p "$SHIMS_DIR"

cat > "$SHIMS_DIR/prlimit" <<'EOF'
#!/usr/bin/env bash
# prlimit shim for Apptainer:
# - drop any --rss=... flags (Linux/containers won't allow RLIMIT_RSS)
# - if payload present, run real prlimit with filtered args and the payload
# - if real prlimit fails or is missing, run the payload directly

filtered=()
payload=()
seen_dashes=0
for arg in "$@"; do
  if [[ $seen_dashes -eq 0 ]]; then
    if [[ "$arg" == "--" ]]; then
      seen_dashes=1
      continue
    fi
    # Drop any --rss or --rss=...
    if [[ "$arg" == --rss ]] || [[ "$arg" == --rss=* ]]; then
      continue
    fi
    filtered+=("$arg")
  else
    payload+=("$arg")
  fi
done

REAL="/mnt/shims/prlimit.real"

if [[ -x "$REAL" ]]; then
  if [[ ${#payload[@]} -gt 0 ]]; then
    "$REAL" "${filtered[@]}" -- "${payload[@]}" 2>/dev/null
    rc=$?
    # If prlimit failed (caps missing / unsupported), just run the payload directly
    if [[ $rc -ne 0 ]]; then
      exec "${payload[@]}"
    else
      exit 0
    fi
  else
    "$REAL" "${filtered[@]}" 2>/dev/null
    exit $?
  fi
fi

# Fallbacks: if real prlimit missing or failed, run payload directly (or succeed if none)
if [[ ${#payload[@]} -gt 0 ]]; then
  exec "${payload[@]}"
else
  exit 0
fi
EOF
chmod +x "$SHIMS_DIR/prlimit"


# === Start execution engine in background inside Apptainer ===
apptainer exec --fakeroot --nv -C -e \
  -B /project \
  -B /scratch \
  -B /project/def-fard/niktakbr/ExecEval/execution_engine:/mnt/execution_engine \
  -B /project/def-fard/niktakbr/tmp_root/code_store:/code_store \
  -B $PROJECT_DIR/promptsource:/mnt/promptsource \
  -B $PROJECT_DIR/shims:/mnt/shims \
  -B /usr/bin/prlimit:/mnt/shims/prlimit.real \
  -W /project/def-fard/niktakbr/tmp_root \
  --env NUM_WORKERS=37 \
  --env GOPATH="$GOPATH" \
  --env GOCACHE="$GOCACHE" \
  --env GRADLE_USER_HOME="$GRADLE_USER_HOME" \
  --env PATH="/mnt/shims:${TOOL_PATHS:-/usr/local/bin:/usr/bin:/bin}" \
  /project/def-fard/niktakbr/ExecEval/docker.sif \
    bash -lc '
      echo "[engine preflight] PATH=$PATH"
      for cmd in go node kotlinc rustc cargo gcc g++ clang++-14 csc mono ruby php pypy3 python3 javac java; do     
        printf "[engine preflight] %-12s : " "$cmd"; command -v "$cmd" || echo "MISSING"
      done
      cd /mnt/execution_engine && bash -x start_engine.sh
    ' \
  > "$PROJECT_DIR/engine.out" 2> "$PROJECT_DIR/engine.err" &

# === Poll for readiness up to 120s (accept 200/204/405/404) ===
echo "[engine] waiting for readiness..."
for i in $(seq 1 120); do
  code=$(apptainer exec --fakeroot -C -e \
           -B /project -B /scratch \
           --env PATH="/mnt/shims:${TOOL_PATHS:-/usr/local/bin:/usr/bin:/bin}" \
           /project/def-fard/niktakbr/ExecEval/docker.sif \
           bash -lc 'curl -s -o /dev/null -w "%{http_code}" -X OPTIONS http://127.0.0.1:5000/api/execute_code || curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/' || true)

  if [[ "$code" == "200" || "$code" == "204" || "$code" == "405" || "$code" == "404" ]]; then
    echo "[engine] healthy after ${i}s (code=$code)"
    break
  fi

  sleep 1
  if (( i % 20 == 0 )); then
    echo "[engine] still waiting... (${i}s, last code=${code}) tailing logs:"
    tail -n 80 "$PROJECT_DIR/engine.err" || true
  fi

  if [[ $i -eq 120 ]]; then
    echo "[engine] FAILED to become ready (last code=${code}). Dumping logs:"
    tail -n +1 "$PROJECT_DIR/engine.out" || true
    tail -n +1 "$PROJECT_DIR/engine.err" || true
    exit 1
  fi
done



# === prlimit sanity check (uses same PATH) ===
apptainer exec --fakeroot -C -e \
  -B /project -B /scratch \
  -B $PROJECT_DIR/shims:/mnt/shims \
  -B /usr/bin/prlimit:/mnt/shims/prlimit.real \
  --env PATH="/mnt/shims:${TOOL_PATHS:-/usr/local/bin:/usr/bin:/bin}" \
  /project/def-fard/niktakbr/ExecEval/docker.sif \
  bash -lc 'echo "which prlimit:"; which -a prlimit; \
    prlimit --rss=1M --cpu=10 --as=4G -- bash -lc "python -c '\''import time; print(\"hello\"); time.sleep(0.1)'\''"; \
    echo "exit:$?"' | tee "$PROJECT_DIR/prlimit_sanity.log"


export EXEC_ENGINE_URL="http://127.0.0.1:5000"
echo "[engine] EXEC_ENGINE_URL=${EXEC_ENGINE_URL}"


# === Wait for engine to be ready ===
which python
python -c "import sys; print('PYTHONPATH=', repr(sys.path))"
python -c "import pyarrow; print('pyarrow OK:', pyarrow.__version__)"

# === Run main logic ===
source $VENV_DIR/bin/activate
# keep whatever the module put in PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/promptsource:${PYTHONPATH:+$PYTHONPATH}"

python $PROJECT_DIR/main.py --config $PROJECT_DIR/config/trans.yaml