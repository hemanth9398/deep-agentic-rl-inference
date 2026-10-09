#!/bin/bash
set -e
cd "${WORKDIR:-/workspace}"
git apply /solution/solve.patch
