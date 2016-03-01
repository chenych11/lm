#!/usr/bin/env bash
if [ "x$1" = "x--dry-run" ]; then
    command_prefix="echo "
else
    command_prefix=
fi

export PYTHONPATH="${PWD}/../..:${PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python_command="python main_lblv2.py"
coding_file="../data/sparse/total-app-a0.1-b0.1-w1-0.1-15000.pkl"
embed_file="../data/models/embeddings/rw2vec_embeddings-size200.pkl"
data_file="../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2"
context_size=5
embed_size=200

# test different vocab size
lr='0.004'
lr_min='0.002'
gamma='0.03'
nb_neg=50
for ((nb_vocab=10000; nb_vocab<30000; nb_vocab+=2000)); do
    log_file="../logs/main-lblv2-C${context_size}-E${embed_size}-lr${lr}-lr_min${lr_min}-g${gamma}-V${nb_vocab}-N${nb_neg}.log"
    command_line_="$python_command -C ${context_size} -E ${embed_size} \
       --lr=${lr} --lr-min=${lr_min} \
       -d --gamma=${gamma} -N ${nb_neg} \
       -S $coding_file -e $embed_file --log-file $log_file \
       -D $data_file -V $nb_vocab "
    command_line=`echo "$command_line_" | tr -s " "`
    ${command_prefix} nohup sh -c "$command_line" &
    sleep 40
done


#for ((nb_vocab=30000; nb_vocab<=50000; nb_vocab+=2000)); do
#    log_file="../logs/main-lblv2-C${context_size}-E${embed_size}-lr${lr}-lr_min${lr_min}-g${gamma}-V${nb_vocab}-N${nb_neg}.log"
#    command_line_="$python_command -C ${context_size} -E ${embed_size} \
#       --lr=${lr} --lr-min=${lr_min} \
#       -d --gamma=${gamma} -N ${nb_neg} \
#       -S $coding_file -e $embed_file --log-file $log_file \
#       -D $data_file -V $nb_vocab "
#    command_line=`echo "$command_line_" | tr -s " "`
#    ${command_prefix} nohup sh -c "$command_line" &
#    sleep 40
#done

# test different lr:
#lr_min='0.002'
#gamma='0.003'
#nb_neg=50
#nb_vocab=30000
#for lr in 0.04 0.03 0.02; do #0.01 0.008 0.006; do
#    log_file="../logs/main-lblv2-C${context_size}-E${embed_size}-lr${lr}-lr_min${lr_min}-g${gamma}-V${nb_vocab}-N${nb_neg}.log"
#    command_line_="$python_command -C ${context_size} -E ${embed_size} \
#       --lr=${lr} --lr-min=${lr_min} \
#       -d --gamma=${gamma} -N ${nb_neg} \
#       -S $coding_file -e $embed_file --log-file $log_file \
#       -D $data_file -V $nb_vocab "
#    command_line=`echo "$command_line_" | tr -s " "`
#    ${command_prefix} nohup sh -c "$command_line" &
#    sleep 40
#done

#
#lr='0.01'
#for gamma in 0.001 0.002 0.004; do
#    log_file="../logs/main-lblv2-C${context_size}-E${embed_size}-lr${lr}-lr_min${lr_min}-g${gamma}-V${nb_vocab}-N${nb_neg}.log"
#    command_line_="$python_command -C ${context_size} -E ${embed_size} \
#       --lr=${lr} --lr-min=${lr_min} \
#       -d --gamma=${gamma} -N ${nb_neg} \
#       -S $coding_file -e $embed_file --log-file $log_file \
#       -D $data_file -V $nb_vocab "
#    command_line=`echo "$command_line_" | tr -s " "`
#    ${command_prefix} nohup sh -c "$command_line" &
#    sleep 40
#done
