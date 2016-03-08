#!/usr/bin/env bash
if [ "x$1" = "x--dry-run" ]; then
    command_prefix="echo "
else
    command_prefix=
fi

export PYTHONPATH="${PWD}/../..:${PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python_command="python main_nce8.py"
coding_file="../data/sparse/total-app-a0.1-b0.1-w1-0.1-15000.pkl"
embed_file="../data/models/embeddings/rw2vec_embeddings-size200.pkl"
data_file="../data/corpus/wiki-sg-norm-lc-drop-bin-sample.bz2"
context_size=200
embed_size=200

# test different vocab size
lr='0.002'
nb_neg=50
for ((nb_vocab=10000; nb_vocab<16000; nb_vocab+=2000)); do
    log_file="../logs/main-nce8-C${context_size}-E${embed_size}-lr${lr}-V${nb_vocab}-N${nb_neg}.log"
    command_line_="$python_command -V $nb_vocab -C ${context_size} -E ${embed_size} \
        --lr=${lr}\
        -N ${nb_neg} \
        -S $coding_file -e $embed_file --log-file $log_file \
        -D $data_file "
    command_line=`echo "$command_line_" | tr -s " "`
    ${command_prefix} sh -c "$command_line" &
    sleep 120
done

