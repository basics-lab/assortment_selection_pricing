#!/bin/bash

RUNS=10
T_LIST=(125 250 500 1000 2000)

echo "Please choose an experiment setup (1, 2, or 3):"
read XP

if [ $XP -eq 1 ]; then
    K=5
    N=5
    D=5
    L0=0.5
elif [ $XP -eq 2 ]; then
    K=5
    N=100
    D=10
    L0=0.5
elif [ $XP -eq 3 ]; then
    K=10
    N=100
    D=10
    L0=0.1
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Store PIDs of background jobs
pids=()

# Get the current date and time in YYYYMMDD_HHMMSS format
current_datetime=$(date +"%Y%m%d_%H%M%S")

for T in "${T_LIST[@]}"; do
    for i in $(seq 1 $RUNS); do
        python experiments.py -K $K -N $N -d $D -L0 $L0 -T $T -result_dir "results/$current_datetime" &
        pids+=($!)  # Store the PID of the last background process
    done
done

# Function to kill all jobs
cleanup() {
    echo "Terminating all jobs..."
    for pid in "${pids[@]}"; do
        kill $pid 2>/dev/null
    done
    exit 1
}

# Trap Ctrl+C (SIGINT) and SIGTERM to clean up jobs
trap cleanup SIGINT SIGTERM

# Wait for all background jobs to finish
wait