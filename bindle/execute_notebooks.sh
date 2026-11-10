#!/bin/bash

################################################################################
echo
echo "running execute_notebooks.sh"
echo "---------------------------------------------"
################################################################################

# fail on error
set -e

################################################################################
echo
echo "other initialization"
echo "--------------------"
################################################################################

# adapted from https://stackoverflow.com/a/24114056
script_dir="$(realpath "$(dirname -- "$BASH_SOURCE")")"
echo "script_dir ${script_dir}"

################################################################################
echo
echo "execute marimo notebooks in current directory"
echo "---------------------------------------------"
################################################################################

# cd into script_dir so teeplot outputs land under bindle/teeplots/
# (unlike jupyter nbconvert, `marimo export ipynb` uses the caller's cwd
# rather than the notebook's directory)
cd "${script_dir}"

shopt -s nullglob

if [ $# -gt 0 ]; then
  notebooks=("$@")
else
  notebooks=(*.py)
fi

for notebook in "${notebooks[@]}"; do
  echo "notebook ${notebook}"
  export NOTEBOOK_NAME="$(basename "${notebook%.*}")"
  export NOTEBOOK_PATH="$(realpath "${notebook}")"
  output_ipynb="${NOTEBOOK_NAME}.ipynb"
  # export marimo notebook to jupyter notebook with executed outputs
  # adapted from https://docs.marimo.io/guides/exporting/
  # `marimo export ipynb` exits non-zero when any cell errored, but only
  # prints "Export was successful, but some cells failed to execute." with
  # no traceback. Capture the status, then dump cell tracebacks from the
  # produced ipynb so CI logs show what actually failed.
  set +e
  marimo export ipynb \
    --include-outputs \
    --sort topological \
    -f \
    "${notebook}" \
    -o "${output_ipynb}"
  marimo_status=$?
  set -e
  if [ "${marimo_status}" -ne 0 ]; then
    echo
    echo "marimo export failed for ${notebook} (exit ${marimo_status})"
    echo "---------------------------------------------"
    if [ -f "${output_ipynb}" ]; then
      jq '.cells[].outputs[]? | select(.output_type=="error")' "${output_ipynb}"
      jq -r '.. | select(type == "object" and .output_type == "error") | .traceback[]' "${output_ipynb}"
    else
      echo "(${output_ipynb} was not produced)"
    fi
    exit "${marimo_status}"
  fi
done

shopt -u nullglob

################################################################################
echo
echo "recurse to subdirectories"
echo "-------------------------"
################################################################################

shopt -s nullglob

for script in "${script_dir}/"*/execute_notebooks.sh; do
  "${script}" "$@"
done

shopt -u nullglob
