#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/c3vd/endomd_slam_c3vdv2.py}"
SEQUENCES="${SEQUENCES:-configs/c3vd/c3vdv2_degradation_sequences.txt}"
BASEDIR="${BASEDIR:-./data/C3VDv2}"
WORKDIR="${WORKDIR:-./outputs/endomd_slam_c3vdv2}"
PYTHON="${PYTHON:-python}"

while IFS= read -r sequence; do
  if [[ -z "${sequence}" || "${sequence}" == \#* ]]; then
    continue
  fi
  "${PYTHON}" scripts/run_endomd_slam.py "${CONFIG}" \
    --basedir "${BASEDIR}" \
    --sequence "${sequence}" \
    --workdir "${WORKDIR}" \
    --run_name "${sequence}"
done < "${SEQUENCES}"
