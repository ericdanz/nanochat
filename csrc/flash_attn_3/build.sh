#!/bin/bash
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
python -m pip install ninja packaging 2>/dev/null
python -m pip install -e . --no-build-isolation
python -c "import flash_attn_3; print('flash_attn_3 build OK')"
