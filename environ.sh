#!/usr/bin/env bash
source ~/bin/ch.gcc-4.8.4.sh
export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,nvcc.fastmath=True,scan.allow_gc=True,allow_gc=True
export PYTHONPATH=/home/cyc/Documents/workspace:$PYTHONPATH
export OMP_NUM_THREADS=4