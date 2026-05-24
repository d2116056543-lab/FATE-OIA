#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/SNNA_repro/SNNA
echo "---ROOT---"
pwd
echo "---FILES MATCHING VIS/AUDIO TERMS---"
find . -maxdepth 2 -type f | sed 's#^\./##' | sort | grep -Ei 'heat|attn|attention|vis|visual|grad|cam|saliency|plot|demo|infer|test|noise|mask' || true
echo "---PYTHON GREP---"
grep -RInE 'attention|attn|grad|cam|heat|visual|noise|mask|register_forward|hook|return_attention|get_last_selfattention' -- *.py 2>/dev/null | head -160 || true
