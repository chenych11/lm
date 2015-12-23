#!/usr/bin/env bash
source ../environ.sh

models_dir="../data/models/lang"
log_dir="../logs"

python run_nce0.py --lr 0.04 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.04.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.04.log
python run_nce0.py --lr 0.02 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.02.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.02.log
python run_nce0.py --lr 0.01 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.01.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.01.log
python run_nce0.py --lr 0.005 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.005.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.005.log

python run_nce0.py --lr 0.04 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.04-d.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.04-d.log -d --lr-min 0.005
python run_nce0.py --lr 0.02 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.02-d.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.02-d.log -d --lr-min 0.005
python run_nce0.py --lr 0.01 -C 128 -E 128 --save ${models_dir}/nce0-lstm-c128-e128-neg50-lr0.01-d.pkl \
                   --log-file ${log_dir}/nce0-lstm-c128-e128-neg50-lr0.01-d.log -d --lr-min 0.005