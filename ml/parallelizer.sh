#!/bin/bash

values=(1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
