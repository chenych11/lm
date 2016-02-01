#!/usr/bin/env bash

DATA_DIR="../../data/corpus/sri"

for (( nb_vocab=10; nb_vocab <=50; nb_vocab+= 2 )); do
    TEXTFILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k.txt"
    TEST_DATA="${DATA_DIR}/wiki-val-R5m-V${nb_vocab}k.txt"
    for (( kn = 1; kn <=4; kn += 1 )); do
        COUNT_FILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-kn${kn}.count"
        LM="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-kn${kn}.lm"
        echo "Begin: Test order 4, kn=${kn}, vocab: ${nb_vocab}"
        ngram-count -text $TEXTFILE -kndiscount $kn -order 4 -write-binary $COUNT_FILE
        ngram-count -read $COUNT_FILE -kn-counts-modified -write-binary-lm -lm $LM
        ngram -lm ${LM} -ppl ${TEST_DATA}
        echo "END: Test order 4, kn=${kn}"
    done
done

for (( nb_vocab=10; nb_vocab <=50; nb_vocab+= 2 )); do
    TEXTFILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k.txt"
    TEST_DATA="${DATA_DIR}/wiki-val-R5m-V${nb_vocab}k.txt"
    for (( gt = 1; gt <=4; gt += 1 )); do
        COUNT_FILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-gt${gt}.count"
        LM="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-gt${gt}.lm"
        PAR="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-gt${gt}.gt"
        echo "Begin: Test order 4, gt=${gt}, vocab: ${nb_vocab}"
        ngram-count -text $TEXTFILE  -order 4 -gt${gt} ${PAR} -write-binary $COUNT_FILE
        ngram-count -read $COUNT_FILE -write-binary-lm -lm ${LM} -gt${gt} ${PAR}
        ngram -lm ${LM} -ppl ${TEST_DATA}
        echo "END: Test order 4, gt=${gt}, vocab: ${nb_vocab}"
    done
done