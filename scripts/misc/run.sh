#!/bin/bash
# Usage (from WSL, in this folder):  bash run.sh test.py
# Uses rfd3_env's python + the rfd4 packages (atomworks, rfproteina) on PYTHONPATH.
PYTHONPATH=/home/yunisoo/binder/envs/rfd4_env/lib/python3.12/site-packages \
/home/yunisoo/binder/envs/rfd3_env/bin/python "$@"
