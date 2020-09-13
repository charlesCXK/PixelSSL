#!/usr/bin/env bash

source activate ssc
export NGPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi

python -m script.deeplabv2_pascalvoc_1-8_sslgct_cxk