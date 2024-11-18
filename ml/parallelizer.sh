#!/bin/bash

values=(1000 2000 5000 10000 20000 30000 50000 100000 200000 500000 1000000)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
