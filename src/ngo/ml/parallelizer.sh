#!/bin/bash

values=(6 8 9)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
