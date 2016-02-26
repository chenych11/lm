#!/usr/bin/bash
for (( nb_vocab=10000; nb_vocab<52000; nb_vocab+=2000 )); do
    let file_name=${nb_vocab}/1000
    for (( i=0; i<$nb_vocab; ++i )); do
	echo $i 
    done > ${file_name}k.vocab 
done

