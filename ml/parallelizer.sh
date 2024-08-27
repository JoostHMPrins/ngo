#!/bin/bash

values=(2 5 20)

for value in "${values[@]}"
do
    python trainparallel.py --arg "$value" &
done

wait  # Wait for all background processes to complete
