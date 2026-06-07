#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git git-lfs
fi

git lfs install --force
git lfs pull

python3 -V
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

python3 -u -m unittest tests.test_prepare_geo_datasets tests.test_apps_pipeline tests.test_generalization_protocol
python3 -m compileall -q apps scripts main.py tests

python3 -u main.py
python3 -u -m scripts.evaluation.nested_internal_validation
python3 -u -m scripts.evaluation.loco_validation

git config user.name "cnb-bot"
git config user.email "cnb-bot@local"
git add -f results/
git commit -m "pipeline-results [skip ci]" || true
git push origin HEAD:develop
