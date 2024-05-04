#!/bin/bash
source ~/.bashrc

cd STS

# make necessary multilingual STSBenchmark directories
mkdir STSBenchmarkES
mkdir STSBenchmarkFR
mkdir STSBenchmarkIT
mkdir STSBenchmarkPT

# copy multilingual datasets into their respective directories
# this assumes that the user has already cloned the sts-multi-mt respository into their home directory

# for Spanish
cp $HOME/stsb-multi-mt/data/stsb-es-dev.csv ./STSBenchmarkES
cp $HOME/stsb-multi-mt/data/stsb-es-test.csv ./STSBenchmarkES
cp $HOME/stsb-multi-mt/data/stsb-es-train.csv ./STSBenchmarkES

# for French
cp $HOME/stsb-multi-mt/data/stsb-fr-dev.csv ./STSBenchmarkFR
cp $HOME/stsb-multi-mt/data/stsb-fr-test.csv ./STSBenchmarkFR
cp $HOME/stsb-multi-mt/data/stsb-fr-train.csv ./STSBenchmarkFR

# for Italian
cp $HOME/stsb-multi-mt/data/stsb-it-dev.csv ./STSBenchmarkIT
cp $HOME/stsb-multi-mt/data/stsb-it-test.csv ./STSBenchmarkIT
cp $HOME/stsb-multi-mt/data/stsb-it-train.csv ./STSBenchmarkIT

# for Portuguese
cp $HOME/stsb-multi-mt/data/stsb-pt-dev.csv ./STSBenchmarkPT
cp $HOME/stsb-multi-mt/data/stsb-pt-test.csv ./STSBenchmarkPT
cp $HOME/stsb-multi-mt/data/stsb-pt-train.csv ./STSBenchmarkPT
