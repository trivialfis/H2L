#!/bin/sh
rm *.aux *.tex *.log *.pdf
DUMP=../../dump
if [ -d $DUMP ]
then
    rm ${DUMP}/*
fi
