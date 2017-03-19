#!/bin/bash

for prefix in {a..r}
do
    for i in {1..7}
    do
	mkdir ${prefix}0${i}
	mv ${prefix}0${i}-*.xml ./${prefix}0${i}/
    done
done
find ./ -empty -type d -delete
