#!/bin/bash

MODELS=("INI" "ER" "OL" "ICP" "ICT" "ER-ICP" "ER-ICT" "OL-ICP" "OL-ICT")
BEHAVIORS=("CONV" "DIV")

# Ring: n=3..10, m=2..10
for n in {3..10}; do
  for m in {2..10}; do
    for behavior in "${BEHAVIORS[@]}"; do
      for model in "${MODELS[@]}"; do
        python3 GraphSolver.py ring $n $m $behavior $model
      done
    done
  done
done

# Chain: n=3..10, m=2..10
for n in {3..10}; do
  for m in {2..10}; do
    for behavior in "${BEHAVIORS[@]}"; do
      for model in "${MODELS[@]}"; do
        python3 GraphSolver.py chain $n $m $behavior $model
      done
    done
  done
done

# Star: n=3..5, m=2..5
for n in {3..5}; do
  for m in {2..5}; do
    for behavior in "${BEHAVIORS[@]}"; do
      for model in "${MODELS[@]}"; do
        python3 GraphSolver.py star $n $m $behavior $model
      done
    done
  done
done
