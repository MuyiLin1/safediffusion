#!/usr/bin/env bash
# Status of run_theory_experiments.sh.
#   bash progress_theory.sh              one-shot
#   watch -n 30 bash progress_theory.sh  live (Ctrl-C to leave; jobs keep running)
cd "$(dirname "$0")"
ROOT="${ROOT:-data/theory}"

now=$(date +%s)
if [[ -f "$ROOT/.started" ]]; then
    el=$((now - $(cat "$ROOT/.started")))
    printf "elapsed since launch: %dh %02dm\n" $((el / 3600)) $((el % 3600 / 60))
fi
printf "load average: %s\n\n" "$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"

printf "%-26s %7s %7s %7s %7s\n" stage done failed running total
for d in "$ROOT"/stage0_scalar "$ROOT"/stage[1-9]*_*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    if [[ "$name" == stage0_scalar ]]; then
        if [[ -s "$d/lower_bound.csv" ]]; then s="done"; else s="pending"; fi
        printf "%-26s %7s\n" "$name" "$s"
        continue
    fi
    total=$(wc -l < "$d/jobs.txt" 2>/dev/null || echo 0)
    ndone=$(find "$d/parts" -name '*.done' 2>/dev/null | wc -l)
    nfail=$(find "$d/parts" -name '*.failed' 2>/dev/null | wc -l)
    nrun=$(ps -eo args | grep -F "exp_theory" | grep -F -- "$ROOT/$name/parts" | grep -vc grep)
    printf "%-26s %7d %7d %7d %7d\n" "$name" "$ndone" "$nfail" "$nrun" "$total"
done
[[ -d figures/theory && -f figures/theory/summary.txt ]] && echo "stage8_report             done (figures/theory/summary.txt)"

echo
echo "running jobs:"
ps -eo pid,etime,args | grep -F "sindy.exp_theory" | grep -v grep \
    | sed -E 's/.*(stage[0-9]_[a-z_]+)\/parts\/([^ ]+)\.csv.*/\1\/\2/' | sed 's/^/  /'
ps -eo pid,etime,args | grep -F "run_theory_experiments.sh" | grep -vq grep \
    || echo "  (driver script is NOT running)"

echo
echo "newest log lines:"
for f in $(ls -t "$ROOT"/stage*/logs/*.log 2>/dev/null | head -n 3); do
    echo "  -- $f"
    grep -v '^\s*$' "$f" | grep -E "safe_box|delta_c|saved|Error|error|Traceback|Epoch [0-9]+:" \
        | sed -n '$p' | cut -c1-160 | sed 's/^/     /'
done

fails=$(find "$ROOT" -name '*.failed' 2>/dev/null)
if [[ -n "$fails" ]]; then
    echo
    echo "FAILED jobs (log = same name under logs/):"
    echo "$fails" | sed 's/^/  /'
fi
