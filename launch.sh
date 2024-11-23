
#! /bin/bash

for wd in 0.1 0.01 0.001 0.0001; do
  for ht in 0 1 2; do
    for prob in 0 1; do
      sbatch --export=WD=$wd,HT=$ht,P=$prob hyper_search.sbatch
    done
  done
done
