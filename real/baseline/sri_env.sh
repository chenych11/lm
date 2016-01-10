#!/usr/bin/env bash
export SRILM="$( cd "../../$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/srilm-1.7.1
export PATH=$PATH:$SRILM/bin/i686-m64
export MANPATH=$MANPATH:$SRILM/man
export LC_NUMERIC=C