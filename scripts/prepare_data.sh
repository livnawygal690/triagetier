#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/raw}"
mkdir -p "${DATA_ROOT}"

cat <<'EOF'
Place the three evaluation datasets under the data root:

  ${DATA_ROOT}/ratic/         RSNA 2023 Abdominal Trauma Detection (Kaggle)
                              build test_manifest.csv with columns:
                              study_id,series_path,liver,spleen,kidney,bowel,extravasation,mechanism

  ${DATA_ROOT}/cq500/         qure.ai head-CT release
                              build reads.csv with columns:
                              name,series_path,intraparenchymal,intraventricular,subdural,
                              extradural,subarachnoid,calvarial_fracture

  ${DATA_ROOT}/mimic-iv-ed/   PhysioNet mimic-iv-ed v2.2 (credentialed access)
                              edstays.csv, triage.csv, diagnosis.csv

RATIC and CQ500 volumes are expected as NIfTI; convert DICOM series first.
EOF
