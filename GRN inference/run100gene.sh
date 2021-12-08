#!/bin/bash
for i in `seq 1 5`;
do
    echo "net$i"
    for sizem in 40
    do
        echo "sizem $sizem"
        for sizen in 200 300 400 600 700 800 900
        do
            echo "sizen $sizen"
            infile=/home/linaiqi/Lab/data/gene/mf100net$i.txt
            outfile=/home/linaiqi/Lab/data/gene/tmp/gpgene_mf100net$i\_n$sizen\_m$sizem.txt
            echo $outfile
            python main.py --infile $infile --outfile $outfile --sizen $sizen --sizem $sizem --gene 100 --iter 12000
        done
    done
done