#!/bin/bash
#SBATCH -N 1 -n 1 --gpus-per-node=1 -c 4
#SBATCH -A ACD115083
#SBATCH -t 60
#SBATCH -J micro_all_resolutions
#SBATCH -o ../slurm-micro-all-%j.out
#SBATCH -e ../slurm-micro-all-%j.err

# Purpose: Benchmark microoptimizations (Fused vs Prefetch vs BytePtr)
#          across all 6 standard resolutions to show consistency
# Output: CSV file with columns: resolution, method, dp_time_ms, speedup_vs_fused

set -e

cd /home/u2713124/NTUPDP2026

# Binaries
FUSED="./cuda/seam_carve_fused"
PREFETCH="./cuda/seam_carve_v4"
BYTEPTR="./cuda/seam_carve_v5"

# Test data (all 6 resolutions)
declare -A images
images[ctrl]="data/ctrl/broadway_tower_ctrl_960x540.png"
images[1080p]="data/1080p/desert_mesa_1080p_1920x1080.png"
images[2k]="data/2k/forest_pano_2k_2560x1440.png"
images[4k]="data/4k/desert_mesa_4k_3840x2160.png"
images[6k]="data/6k/forest_pano_6k_6144x3456.png"
images[8k]="data/8k/desert_mesa_8k_7680x4320.png"

# Remove fewer seams to speed up benchmark (20 instead of 50)
SEAMS=20

OUTPUT_CSV="results/micro_all_resolutions.csv"
mkdir -p results
echo "resolution,method,dp_time_ms,speedup_vs_fused" > "$OUTPUT_CSV"

for res in ctrl 1080p 2k 4k 6k 8k; do
    image="${images[$res]}"
    
    if [[ ! -f "$image" ]]; then
        echo "SKIP: $res ($image not found)" >&2
        continue
    fi
    
    echo "=== Benchmarking $res: $image ===" >&2
    
    # Run each method 2 times, take best (lowest) time
    declare -A times
    
    for method_name in "Fused" "Prefetch" "BytePtr"; do
        case "$method_name" in
            Fused)    binary="$FUSED" ;;
            Prefetch) binary="$PREFETCH" ;;
            BytePtr)  binary="$BYTEPTR" ;;
        esac
        
        best_time=99999.0
        
        for iter in 1 2; do
            # Run and extract DP kernel time from output
            # seam_carve binaries print: "GPU carving time: X.XXX ms  (Y.YYY ms/seam)"
            output=$("$binary" "$image" "$SEAMS" /tmp/micro_bench.png 2>&1 || true)
            dp_time=$(echo "$output" | grep -oP 'GPU carving time: \K[0-9.]+' | head -1)
            
            if [[ -z "$dp_time" ]]; then
                echo "  ERROR: Could not parse time from $method_name iteration $iter" >&2
                continue
            fi
            
            # Keep best (minimum) time
            if (( $(echo "$dp_time < $best_time" | bc -l) )); then
                best_time=$dp_time
            fi
        done
        
        times[$method_name]=$best_time
        echo "  $method_name: ${times[$method_name]} ms" >&2
    done
    
    # Compute speedups relative to Fused
    fused_ms=${times[Fused]}
    prefetch_ms=${times[Prefetch]}
    byteptr_ms=${times[BytePtr]}
    
    if [[ -z "$fused_ms" ]] || (( $(echo "$fused_ms == 0" | bc -l) )); then
        echo "  SKIP: Fused baseline failed for $res" >&2
        continue
    fi
    
    prefetch_speedup=$(echo "scale=2; $fused_ms / $prefetch_ms" | bc -l)
    byteptr_speedup=$(echo "scale=2; $fused_ms / $byteptr_ms" | bc -l)
    
    # Write results
    echo "$res,Fused,$fused_ms,1.00" >> "$OUTPUT_CSV"
    echo "$res,Prefetch,$prefetch_ms,$prefetch_speedup" >> "$OUTPUT_CSV"
    echo "$res,BytePtr,$byteptr_ms,$byteptr_speedup" >> "$OUTPUT_CSV"
done

echo "Results saved to $OUTPUT_CSV" >&2
cat "$OUTPUT_CSV"
