#!/bin/bash
#SBATCH --job-name=self-colab-run
#SBATCH --account=def-fard_gpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100
#SBATCH --output=self_colab_output.txt
#SBATCH --error=self_colab_error.txt

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
VENV_DIR=/project/def-fard/niktakbr/Self-collaboration-Code-Generation/.venv
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


# Force Python to import our local repo first
export PYTHONPATH="$PROJECT_DIR/promptsource:${PYTHONPATH:+$PYTHONPATH}"

pip install pyyaml \
  --no-index \
  -f $PYTHON_PKGS \
  -f $WHEELHOUSE \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/

pip install \
  pyyaml jinja2 \
  --no-index \
  -f "$PYTHON_PKGS" \
  -f "$WHEELHOUSE" \
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/


# Sanity check to make sure we're using the right one
python - <<'PY'
import promptsource
print("Using promptsource from:", promptsource.__file__)
from promptsource.templates import Template
import inspect
print("Template.__init__ signature:", inspect.signature(Template.__init__))
PY


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



source $VENV_DIR/bin/activate
# keep whatever the module put in PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/promptsource:${PYTHONPATH:+$PYTHONPATH}"

bash /project/def-fard/niktakbr/Self-collaboration-Code-Generation/run.sh
