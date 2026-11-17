#!/bin/bash
set -eu
export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=work/.conda/envs/py310/lib
exec work/venvs/cyclea-dea/bin/python work/cyclea-h12/tools/h12_dea_block.py dea_scale69_s4243 --cells generic_prefix:42,generic_prefix:43,pii_leading:42 --segment dea_scale69_s4243_v1a
