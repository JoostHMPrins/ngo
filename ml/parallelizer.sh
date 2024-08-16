#!/bin/bash

values=(2 4 6 8 10 12 14 16)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
