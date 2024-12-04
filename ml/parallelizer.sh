#!/bin/bash

values=(100 1000 10000 100000 1000000)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
