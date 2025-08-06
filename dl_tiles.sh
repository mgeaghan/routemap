#!/bin/bash

if [ -z "${1}" ]; then exit 1; fi
if [ ! -f "${1}" ]; then exit 1; fi

N=32
(
    while read URL
    do
        ZOOM=$(echo $URL | cut -d '/' -f 4)
        SUBDIR=$(echo $URL | cut -d '/' -f 5)
        BN=$(basename $URL)
        TILEDIR="data/tiles/$ZOOM/$SUBDIR"
        TILEPNG="$TILEDIR/$BN"
        if [ -s "$TILEPNG" ]; then continue; fi
        mkdir -p "$TILEDIR"
        ((I=I%N))
        ((I++==0)) && wait
        wget -O "$TILEPNG" $URL &
    done < ${1}
    wait
)
