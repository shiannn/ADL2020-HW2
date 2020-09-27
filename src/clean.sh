#! /bin/bash

if [ -e pre.json ] ; then
    rm pre.json
fi
if [ -e pre-test.json ] ; then
    rm pre-test.json
fi
if [ -e datasets/train.pkl ] ; then
    rm datasets/train.pkl
fi
if [ -e datasets/valid.pkl ] ; then
    rm datasets/valid.pkl
fi

for i in 1 3 5 7 9
do
    if [ -e results/pre$i.json ] ; then
        rm results/pre$i.json
    fi
done

if [ -e ansLength.npy ] ; then
    rm ansLength.npy
fi