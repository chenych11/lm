#!/usr/bin/env bash
# Usage: prog logfile
DATA_DIR="../../data/corpus/sri"
TEXTFILE="$DATA_DIR/wiki-trn-R100m.txt"
TESTFILE="$DATA_DIR/wiki-val-R5m.txt"

KnParams="-kn1 ${DATA_DIR}/wiki-kn1.param -kn2 ${DATA_DIR}/wiki-kn2.param -kn3 ${DATA_DIR}/wiki-kn3.param -kn4 ${DATA_DIR}/wiki-kn4.param"
CommParams="-unk -map-unk 10000000"
ngram-count -order 4 -text $TEXTFILE $KnParams

for (( nb_vocab=10; nb_vocab<=50; nb_vocab+=2 )); do
	NGRAMS=$DATA_DIR/wiki-V${nb_vocab}k.4grams
	VOCAB=$DATA_DIR/${nb_vocab}k.vocab

	ngram-count -order 4 -text $TEXTFILE $CommParams -vocab $VOCAB -write-binary $NGRAMS
	for order in 2 3 4; do
		LM=$DATA_DIR/wiki-V${nb_vocab}k-order${order}.lm
		ngram-count -order $order -read $NGRAMS $CommParams \
		    -kndiscount${order} -write-binary-lm -lm $LM -vocab $VOCAB $KnParams

		echo "PPL Results for V=$nb_vocab and order=$order: " | tee -a $1
		ngram -lm $LM -ppl $TESTFILE $CommParams | tee -a $1
	done
done