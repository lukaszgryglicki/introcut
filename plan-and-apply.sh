#!/bin/bash
time python3 introcut.py --dry-run '/data/hevc_*' --save-plan plan.json
cat plan.json
echo "execute to apply:"
echo "time python3 introcut.py --overwrite '/data/hevc_*' --apply-plan plan.json"
