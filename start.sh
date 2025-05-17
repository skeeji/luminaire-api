#!/bin/bash

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=false
export CUDA_VISIBLE_DEVICES=-1

echo "V?rification des fichiers:"
ls -la

echo "D?marrage de Gunicorn:"
gunicorn --workers=1 --timeout=120 --threads=4 --worker-class=gthread wsgi:app
