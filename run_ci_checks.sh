#!/bin/bash
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1

./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=notebooks
pytest tests/
pytest notebooks/ --nbmake --nbmake-timeout=120
