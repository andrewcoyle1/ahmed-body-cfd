#!/usr/bin/env bash
# Waits for des_case_044 to finish, then resumes local sampling.
# Usage: nohup bash watch_and_resume.sh > watch_and_resume.log 2>&1 &

CASE_DIR="des_cases/des_case_044"
COEFF_FILE_A="${CASE_DIR}/postProcessing/forceCoeffs/0/coefficient.dat"
COEFF_FILE_B="${CASE_DIR}/postProcessing/forceCoeffs/0/forceCoeffs.dat"
LOG="watch_and_resume.log"
POLL_INTERVAL=60  # seconds between checks

echo "[$(date '+%H:%M:%S')]  Watcher started — waiting for des_case_044 to complete"

while true; do
    # Check if any container is still mounting des_case_044
    CONTAINER=$(docker ps --format '{{.Names}}' 2>/dev/null | while read name; do
        docker inspect "$name" --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null \
            | grep -q "des_case_044$" && echo "$name"
    done)

    if [ -z "$CONTAINER" ]; then
        echo "[$(date '+%H:%M:%S')]  des_case_044 container stopped."

        # Verify force output exists
        if [ -f "$COEFF_FILE_A" ] || [ -f "$COEFF_FILE_B" ]; then
            echo "[$(date '+%H:%M:%S')]  Force coefficients found — resuming local sampling."
            cd "$(dirname "$0")" || exit 1
            # Check no other pipeline process is running
            if [ -f ".pipeline.lock" ]; then
                PID=$(cat .pipeline.lock)
                if kill -0 "$PID" 2>/dev/null; then
                    echo "[$(date '+%H:%M:%S')]  Pipeline lock held by PID $PID — aborting auto-resume."
                    exit 1
                fi
            fi
            nohup python3 run_local_sampling.py >> local_sampling.log 2>&1 &
            echo "[$(date '+%H:%M:%S')]  run_local_sampling.py launched (PID $!)"
            exit 0
        else
            echo "[$(date '+%H:%M:%S')]  WARNING: container stopped but no force file found."
            echo "[$(date '+%H:%M:%S')]  Check des_case_044 manually before resuming."
            exit 1
        fi
    fi

    # Still running — log progress
    SIMTIME=$(grep "^Time = " "${CASE_DIR}/log.pimpleFoam" 2>/dev/null | tail -1 | awk '{print $3}')
    echo "[$(date '+%H:%M:%S')]  des_case_044 running in ${CONTAINER} — sim time ${SIMTIME:-unknown}"
    sleep $POLL_INTERVAL
done
