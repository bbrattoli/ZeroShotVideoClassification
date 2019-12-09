#!/usr/bin/env bash
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -O /workplace/GoogleNews-vectors-negative300.bin.gz
gunzip -c /workplace/GoogleNews-vectors-negative300.bin.gz > /workplace/GoogleNews-vectors-negative300.bin