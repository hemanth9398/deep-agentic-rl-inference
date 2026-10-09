#!/bin/bash
# Harbor verifier entry: overlay hidden tests, run pytest, write a 0/1 reward.
set -u
cd "${WORKDIR:-/workspace}"
cp -r /tests/overlay/. .
mkdir -p /logs/verifier
if python -m pytest -q -p no:cacheprovider tests/test_core.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
