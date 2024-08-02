#!/bin/bash

values=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
