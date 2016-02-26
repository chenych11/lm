#!/usr/bin/env bash

DATA_DIR="../../data/corpus/sri"
TEXTFILE="${DATA_DIR}/wiki-trn-R100m-V100k.txt"
TEST_DATA="${DATA_DIR}/wiki-val-R5m-V100k.txt"
COUNT_FILE="${DATA_DIR}/wiki-trn-R100m-order4-gt1-3.count"
OOV="900000"

ngram-count -text $TEXTFILE -order 4 -write-binary $COUNT_FILE \
    -gt1 ${DATA_DIR}/gt1.params \
    -gt2 ${DATA_DIR}/gt2.params \
    -gt3 ${DATA_DIR}/gt3.params

for (( nb_vocab=10; nb_vocab <=50; nb_vocab+= 2 )); do
    LM="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-gt1-3.lm"
    echo "Begin: Test order 4, gt=${gt}, vocab: ${nb_vocab}"

    ngram-count -read $COUNT_FILE -vocab ${nb_vocab}k.vocab -unk -map-unk $OOV \ 
        -order 4 -write-binary-lm -lm $LM \
        -gt1 ${DATA_DIR}/gt1.params \
        -gt2 ${DATA_DIR}/gt2.params \
        -gt3 ${DATA_DIR}/gt3.params
    ngram -unk -map-unk $OOV -lm ${LM} -ppl ${TEST_DATA}

    echo "END: Test order 4, gt=${gt}, vocab: ${nb_vocab}"
done

# for (( nb_vocab=10; nb_vocab <=50; nb_vocab+= 2 )); do
#     TEXTFILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k.txt"
#     TEST_DATA="${DATA_DIR}/wiki-val-R5m-V${nb_vocab}k.txt"
#     for (( kn = 1; kn <=4; kn += 1 )); do
#         COUNT_FILE="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-kn${kn}.count"
#         LM="${DATA_DIR}/wiki-trn-R100m-V${nb_vocab}k-order4-kn${kn}.lm"
#         echo "Begin: Test order 4, kn=${kn}, vocab: ${nb_vocab}"
#         ngram-count -text $TEXTFILE -kndiscount $kn -order 4 -write-binary $COUNT_FILE
#         ngram-count -read $COUNT_FILE -kn-counts-modified -write-binary-lm -lm $LM
#         ngram -lm ${LM} -ppl ${TEST_DATA}
#         echo "END: Test order 4, kn=${kn}"
#     done
# done
