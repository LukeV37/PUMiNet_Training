#!/bin/bash
WORK_DIR=$(pwd)
if [ ! -f ./venv/PUMiNet/bin/activate ]; then
    cd venv
    python3 -m venv PUMiNet
    cd PUMiNet
    source ./bin/activate
    pip install --upgrade pip
    pip install -r ../pip_requirements.txt
else
    source ./venv/PUMiNet/bin/activate
fi
cd $WORK_DIR
