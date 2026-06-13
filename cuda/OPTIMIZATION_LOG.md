# CUDA Seam Carving — 優化歷程

垂直 seam carving 的純 CUDA standalone 實作優化紀錄。每一版都用 Nsight Compute
量測、依數據決定下一步。

## 環境與設定

- **GPU**:NVIDIA Tesla V100-SXM2-32GB(`sm_70`,80 SMs,每 block 上限 1024 threads,
  dynamic shared memory 上限 96 KB via opt-in)
- **測資**:`Broadway_tower_edit.jpg`,1440 × 968,移除 10 條 seam
- **Build**:`make`(raw `.cu` + Makefile,image I/O 用 vendored stb_image,不依賴 torch/python)
- **Pipeline**(每移除一條 seam 跑一輪,4 顆 kernel):
  1. `grayscale_kernel` — RGB → 灰階(0.299R + 0.587G + 0.114B)
  2. `energy_kernel` — gradient magnitude `|dx| + |dy|`,邊界 clamp
  3. **`seam_dp_*`** — cumulative-cost DP 找最小成本 seam(瓶頸)
  4. `remove_seam_kernel` — 移除該 seam,輸出寬度 −1 的影像

DP 遞迴:`cost[i][j] = energy[i][j] + min(cost[i-1][j-1], cost[i-1][j], cost[i-1][j+1])`,
回溯紀錄 argmin 欄;tie-break 偏好較小欄(left→center→right,嚴格 `<`)以對齊 torch 參考答案。

## Profiling 方法

```bash
# 1. 產生報告(ncu 慢,srun -t 給大一點)
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 30 \
    ncu --set full -o report_vX ./seam_carve_vX ../Broadway_tower_edit.jpg 10 outX.png

# 2. 印出每顆 kernel 的 Dur / SM% / Mem% / Occ% + top warp-stall reasons
python3 ncu_table.py report_vX.ncu-rep
```

`ncu_table.py` 自己呼叫 ncu 轉 CSV 再挑欄位,避免手解 800 欄的原始 CSV。

> ⚠️ **單位陷阱**:ncu 的 `gpu__time_duration` 在不同報告會自動縮放成 ms 或 µs,跨報告比較
> 前先用一顆穩定的 kernel(如 grayscale ≈ 0.031 ms = 31 µs)對齊單位,以免誤判 1000x。

---

## 各版本

### v2 — baseline:單 block 共享記憶體 DP
`cuda/seam_carve.cu` → `./seam_carve`

- **做法**:把整張圖的 DP 融進**一個 block**,blockDim.x 個 thread(一欄一個,grid-stride),
  由上往下走每一列,`prev`/`curr` 兩列 cost 放 dynamic shared memory ping-pong,列間
  `__syncthreads()`;最後一列 argmin + 回溯由 thread 0 做。把原本「每列一顆小 kernel」收成
  「每條 seam 一顆 kernel」。
- **Profiling(第一條 seam)**:

  | kernel | Duration | 占比 |
  |---|---|---|
  | grayscale | 0.031 ms | 2% |
  | energy | 0.016 ms | 1% |
  | **seam_dp** | **1.460 ms** | **93%** |
  | remove_seam | 0.059 ms | 4% |

- **發現**:DP 一顆吃掉 93%。SM throughput 0.35% / memory 0.79%(幾乎全閒置),但**achieved
  occupancy 其實有 ~42%**(warp 都在,只是 stall)。最大 warp stall = **`long_scoreboard` 9.24**
  > barrier 7.08 → **卡在等 global memory 的讀取**:968 列鎖步前進,每列所有 warp 同時停下來等
  該列的 `energy` load,序列依賴讓延遲藏不住。**結論:global-load-latency bound,不是 barrier bound。**

### v3 — cooperative groups 跨 SM(❌ 廢案)
`cuda/seam_carve_v3.cu` → `./seam_carve_v3`

- **動機**:DP 只用 1 個 block = 1/80 SM,直覺想攤到所有 SM。
- **做法**:`seam_dp_coop_kernel`,cost 兩列改放 global ping-pong buffer,列間用 device-wide
  `grid.sync()` 取代 `__syncthreads()`,`cudaLaunchCooperativeKernel` 啟動(需 `-rdc=true`)。
- **結果**:**DP 變慢到 1.657 ms(0.88x,退步)**,occupancy 只到 6.2%,最大 stall =
  **`barrier` 20.12** 獨大。
- **發現/教訓**:`grid.sync()`(跨 SM 全域 barrier)比 `__syncthreads()`(block 內)貴得多;
  DP 有 968 列、每列一次 barrier,換成昂貴的全域 barrier 後 80 個 SM 大多時間都在等 barrier。
  **「每列做一次 grid.sync」是死路,放棄。** 保留檔案僅供存證。
- **關鍵體悟**:H = 968 列的序列依賴**改不掉**(要對精確答案),所以唯一的施力點是「讓每一步更便宜」,
  而不是增加平行度。

### v4 — software-pipelined energy prefetch(✅ 主要突破)
`cuda/seam_carve_v4.cu` → `./seam_carve_v4`

- **動機**:回到 v2 的單 block(便宜的 `__syncthreads`,不碰 grid.sync),直接打 v2 的 bottleneck
  `long_scoreboard`。`energy[row]` 與 DP 的 cost 無關,可以提前載入。
- **做法**:`seam_dp_prefetch_kernel`,算第 `i` 列時就先發射第 `i+1` 列的 `energy` load 進暫存器
  (`e0`/`e1`),等下一圈要用時延遲已被當圈的算 + sync 蓋掉。每個 thread 顧最多 2 欄
  (`c0=tid`, `c1=tid+blockDim.x`),支援寬度 ≤ 2048(host 檢查);`dp_cell` 為兩欄共用的 device helper。
- **結果**:**DP 1.460 → 0.872 ms(1.67x)**。
- **發現**:`long_scoreboard` 9.24 → 掉出前四(< 1.84),**被 prefetch 殺掉了**。新的 stall 排行
  barrier 2.56 / not_selected 2.23 / wait 2.08 / math 1.82,全部變小且分散。

### v5 — int8 相對 back-pointer(✅ 小贏,收尾)
`cuda/seam_carve_v5.cu` → `./seam_carve_v5`

- **動機**:DP 每列都寫 W 個 back-pointer 到 global(968 列 × 1440 × 4 byte ≈ 5.5 MB/seam)。
  減少寫回流量,釋放記憶體頻寬給 energy prefetch。
- **做法**:`back` 從 `int`(絕對欄,4 byte)改成 `signed char`(相對偏移 −1/0/+1,1 byte),
  寫回流量 ÷4;回溯時 `best += back[...]`。DP 數學不變,輸出 bit-identical。
- **結果**:**DP 0.872 → 0.790 ms(再 9%)**。
- **發現**:stall 維持攤平(barrier 2.47 / not_selected 2.21 / wait 2.00 / math 1.82),
  沒有 store throttle。代表 store 流量確實有一點影響,但已不是關鍵路徑。

### v6 — 任意寬度泛化(templated cols/thread)
`cuda/seam_carve_v6.cu` → `./seam_carve_v6`

- **動機**:v4/v5 把「每個 thread 顧 2 欄」寫死(`c0=tid`, `c1=tid+blockDim.x`),寬度被卡在
  `2×1024 = 2048`,連 3840×2160 這種大圖都進不來。但 2048 是人為限制,不是硬體限制 ——
  真正的天花板是 shared memory(兩列 cost `2·W·4` byte ≤ 96KB → 寬度可到 ~12288)。
- **做法**:把「每 thread 欄數」`CPT` 提升成**編譯期 template 參數**,host 端算
  `cpt = ceil(W / blockDim.x)` 後分派到對應的 kernel 實體(`switch`,目前實作 1–8,
  即寬度 ≤ 8192)。欄位用 grid-stride 配置;prefetch 改成固定大小、完全展開的暫存器陣列
  `epf[CPT]`/`ecur[CPT]`,所以**不會 spill 到 local memory**,保留 v5 的 register prefetch。
  W ≤ 2048 時 `CPT=2`,產生的碼與 v5 逐位元相同(效能、輸出都一致),只是現在更寬的圖
  (3840 → `CPT=4`)也能跑。int8 back-pointer 沿用 v5。
- **驗證**:W ≤ 2048 時 v6 輸出與 v5 **bit-identical**(同碼路徑,已用 `cmp v5.png v6.png` 在
  1920×1080 上確認 IDENTICAL);大圖則是 v6 獨有能力。
- **結果**:見下方多解析度 benchmark。重點:**v6 在每個尺寸都 ≥ v5**(泛化零成本,
  template 全展開的 codegen 甚至穩定快 ~4–6%),且 **3840×2160 只有 v6 跑得動**(v5 被 2048 擋掉),
  對 v2 加速 **1.86x(1428)→ 1.82x(1920)→ 2.07x(3840)**——圖越大加速越多。

---

## 多解析度 benchmark

為了讓報告呈現「效能隨解析度 / 隨 seam 數」的 scaling(而非單一資料點),用
`cuda/bench.sh` 掃過多張不同解析度的圖 × 多個 seam 比例(寬度的 5% / 10% / 20%)×
各版本,best-of-N 取最快,輸出 `results.csv`;`cuda/bench_table.py` 再渲染成
含「對 v2 加速比」的 markdown 表。量的是**整條 pipeline 的 wall-clock**(4 顆 kernel
全包),與前面 ncu 只看 DP 單顆互補。

```bash
make
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 20 \
    bash bench.sh ../input.jpg ../input1.jpg ../Broadway_tower_edit.jpg
python3 bench_table.py results.csv
```

### 對照 CPU baseline(整條故事的最底層)

v0–v6 都是 GPU 內部的比較。報告最直觀的一條線是 **GPU vs CPU**。同工具鏈寫了一支
standalone C++ 版 `seam_carve_cpu.cpp`(grayscale / energy / DP / tie-break 與 `seam_carve.cu`
逐位元一致,共用同一套 stb_image I/O、同樣印 `ms/seam`),編出兩支 binary:
`seam_carve_cpu`(單執行緒,`-O3 -march=native`)和 `seam_carve_cpu_omp`(`-fopenmp`,
跨核平行 grayscale/energy/每列 DP cells;列→列相依仍序列,與 GPU 同理)。
跑 `bash baseline_compare.sh` 時 CPU 兩支會自動被納入(SEAMS 預設 10,CPU 較慢故維持小量）。

```bash
make            # 會一併編 seam_carve_cpu / seam_carve_cpu_omp
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -c 8 -t 30 \
    bash baseline_compare.sh ../Broadway_tower_edit.jpg ../input1.jpg ../input.jpg
# OpenMP 版吃 OMP_NUM_THREADS;srun 配 -c 8 並 export OMP_NUM_THREADS=8
```

**CPU ↔ GPU(V100,固定 10 seams,best-of-3,ms/seam):**

| 圖 | cpu(1thr) | cpu(omp 4) | v0 GPU naive | v6 GPU best | **v6 vs cpu(1thr)** | cpu→omp(4核) |
|---|---|---|---|---|---|---|
| 1428×968  | 14.3837 | 5.2757 | 2.7094 | 0.8878 | **16.2x** | 2.73x |
| 1920×1080 | 21.1454 | 7.1302 | 3.1204 | 1.1226 | **18.8x** | 2.97x |
| 3840×2160 | 75.1486 | 23.7736 | 6.7891 | 3.1404 | **23.9x** | 3.16x |

**解讀:**

1. **GPU(v6)對單執行緒 CPU 是 16–24x**,且**圖越大差距越大**(16.2x → 18.8x → 23.9x):大圖
   每列有更多欄位可餵滿 80 個 SM,GPU 的平行度被用得更滿;CPU 則是固定核數硬啃,面積一大就線性變慢。
2. **連最 naive 的 GPU(v0)都海放 CPU**:v0 對 cpu(1thr) 已是 5.3x(1428)→ 11x(3840)。
   也就是「丟上 GPU」本身就拿走大部分的勝利,**我們的 v2→v6 優化是在這之上再 2–3x**。
3. **OpenMP 4 核只拿到 ~2.7–3.2x、不到理論 4x**:每列 DP 都 fork-join 一次(H 次/seam)有
   同步開銷,且 energy/DP 是記憶體頻寬受限,核數加倍不等於頻寬加倍。即便如此,**v6 仍比
   4 核 OpenMP 快 5.9x(1428)/ 6.4x(1920)/ 7.6x(3840)** —— 一張 V100 >> 4 個 CPU 核。
4. **誠實的 headline**:對單執行緒 CPU baseline,**v6 約 16–24x**;對「已經上 GPU」的 naive 版,
   v6 再 2.2–3.1x(見下節)。兩段合起來才是完整故事。

> `-march=native` 在 login node 編、compute node 跑若 ISA 不同可能 `Illegal instruction`;
> 本次在 compute node(gn1221)上實測無此問題。OpenMP 版以 `--export=ALL,OMP_NUM_THREADS=4`
> 帶入 4 核(該 partition 1 GPU 上限 4 CPU)。

### 對照 naive baseline(v0)

上面 v2→v6 的加速是「已經融成一顆 kernel 之後」的增量。真正的起點是 **v0
(`seam_carve_v0.cu`)**:DP 改成**每列發一顆 kernel**(每 seam H 次 launch、整張 cost
matrix 在 global memory 來回、零重用),對應原始 PyTorch `CUDA_v0` 的逐列 launch 寫法
(server 的 python 是 3.6,跑不動原本的 `benchmark.py`,故用同工具鏈的純 CUDA naive 版當對照,
隔離的正是「fusion」這個優化)。`cuda/baseline_compare.sh` 在固定 10 seams 下比 v0/v2/v5/v6:

```bash
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 30 \
    bash baseline_compare.sh ../Broadway_tower_edit.jpg ../input1.jpg ../input.jpg
```

**分層加速(固定 10 seams,ms/seam):**

| 圖 | v0 naive | v2 fuse | v6 | v0→v2(fusion) | v2→v6(latency hiding) | **v0→v6 總計** |
|---|---|---|---|---|---|---|
| 1428×968  | 2.6843 | 1.5746 | 0.8927 | 1.70x | 1.76x | **3.01x** |
| 1920×1080 | 3.0865 | 1.9218 | 1.1175 | 1.61x | 1.72x | **2.76x** |
| 3840×2160 | 5.8930 | **6.6595** | 3.1456 | **0.88x** ⚠️ | 2.12x | **1.87x** |

**解讀(這裡比「單純報一個大倍數」更有資訊量):**

1. **naive baseline 並沒有慢數十倍,只差 ~3x** —— 因為 v0 每列雖然發一顆 kernel,**那顆
   kernel 仍用整顆 GPU(80 SM)平行算 W 個欄位**;列夠寬時 launch overhead 被執行時間蓋掉。
   (原始 torch `CUDA_v0` 每列還多 ~6 個 op + python 迴圈 overhead,會比我們這顆精實的
   1-launch/列更慢,所以 3x 是**保守下界**,對真正 torch 的差距更大。)
2. **fusion 在大圖反而退步**:1428/1920 上 v0→v2 是 1.6–1.7x 的勝利,但 3840 上 **v2(6.66)
   比 naive(5.89)還慢(0.88x)**。把整個 DP 融進**單一 block = 1 個 SM**,要用 1024 threads
   啃 3840 寬的列,單 SM 吞吐拚不過 naive「每列全 GPU 平行」。**fusion 不是各尺度都免費。**
3. **是 latency hiding 才讓單-block 路線全面獲勝**:v4 prefetch / v5 int8 / v6 泛化貢獻
   v2→v6 的 1.7–2.1x;唯有加上它們,融合路線才在大圖以 v6 1.87x 反超 naive。
4. **誠實的 headline**:對精實 naive GPU baseline,典型圖約 **3x**;單-block 的賭注能贏,
   靠的是每步優化(尤其寬度變大時)。

### 結果(V100,best-of-3,單位 ms/seam = 整條 pipeline)

**Broadway_tower_edit.jpg(1428×968)**

| seams(比例) | v2 | v5 | v6 | v5 vs v2 | v6 vs v2 |
|---|---|---|---|---|---|
| 71 (5%)  | 1.3969 | 0.7857 | **0.7540** | 1.78x | **1.85x** |
| 143 (10%)| 1.3827 | 0.7775 | **0.7431** | 1.78x | **1.86x** |
| 286 (20%)| 1.3485 | 0.7627 | **0.7239** | 1.77x | **1.86x** |

**input1.jpg(1920×1080)**

| seams(比例) | v2 | v5 | v6 | v5 vs v2 | v6 vs v2 |
|---|---|---|---|---|---|
| 96 (5%)  | 1.7299 | 1.0187 | **0.9581** | 1.70x | **1.81x** |
| 192 (10%)| 1.7186 | 1.0048 | **0.9451** | 1.71x | **1.82x** |
| 384 (20%)| 1.6954 | 0.9787 | **0.9234** | 1.73x | **1.84x** |

**input.jpg(3840×2160)** — v5 寬度超過 2048 無法執行,只有 v2 / v6

| seams(比例) | v2 | v6 | v6 vs v2 |
|---|---|---|---|
| 192 (5%)  | 6.2347 | **3.0691** | **2.03x** |
| 384 (10%) | 6.2038 | **3.0192** | **2.05x** |
| 768 (20%) | 6.2217 | **2.9404** | **2.12x** |

**跨解析度 summary(v6 vs v2,各圖三比例平均)**

| 圖 | 解析度 | 像素數 | v2 ms/seam | v6 ms/seam | 加速 |
|---|---|---|---|---|---|
| Broadway | 1428×968  | 1.38 M | 1.376 | 0.740 | **1.86x** |
| input1   | 1920×1080 | 2.07 M | 1.714 | 0.942 | **1.82x** |
| input    | 3840×2160 | 8.29 M | 6.220 | 3.010 | **2.07x** |

### 解讀

1. **v6 ≥ v5 全面成立**:同為 W ≤ 2048 的 CPT=2 路徑,v6 仍穩定快 ~4–6%
   (1920:0.958 vs 1.019;1428:0.754 vs 0.786)。template 全展開讓 nvcc 的暫存器配置/
   排程略優於手寫死的兩欄版——泛化不只沒代價,還小賺。
2. **加速隨解析度上升**(1.86x → 1.82x → **2.07x**):圖越大,DP 的全域記憶體流量(讀 energy、
   寫 back-pointer)占比越高,prefetch 藏延遲 + int8 back-pointer ÷4 寫回省下的絕對量也越大,
   所以最大的 4K 圖反而拿到最高 2x+ 的加速。
3. **每 seam 成本對「刪減比例」幾乎不變**(同一張圖 5%/10%/20% 的 ms/seam 落在 ~1% 內,
   甚至隨寬度變窄微降)→ **總時間對 num_seams 線性**。
4. **每 seam 成本對「像素數」近似線性**:3840 是 1428 的 ~4.2x 面積,v6 ms/seam 0.74 → 3.01
   ≈ 4.1x,符合「DP 工作量 ∝ H×W」。
5. **泛化是必要的**:v5 在 3840 直接被 host 端的 2048 檢查擋掉;唯有 v6 能涵蓋全部解析度,
   報告才有完整的三點 scaling。

---

## 總結

| 版本 | DP Duration | 對 v2 加速 | 關鍵手法 |
|---|---|---|---|
| v2 | 1.460 ms | 1.00x | 單 block 共享記憶體 DP(baseline) |
| ~~v3~~ | ~~1.657 ms~~ | ~~0.88x~~ | ~~grid.sync 跨 SM~~ → barrier-bound,廢案 |
| v4 | 0.872 ms | 1.67x | prefetch energy,消除 long_scoreboard |
| v5 | 0.790 ms | 1.85x | + int8 相對 back-pointer,寫回 ÷4 |
| **v6** | **≈0.79 ms**(=v5) | **1.85x** | + 任意寬度泛化(templated CPT),解除 2048 上限 |

- **DP 加速 ~1.85x**(ncu 單顆,1428 圖);整條 pipeline wall-clock 加速 **1.86x(1428)/
  1.82x(1920)/ 2.07x(3840)**,圖越大越多(見上「多解析度 benchmark」)。
- **v5 是效能終點**:warp stall 已完全攤平(barrier 2.47 / not_selected / wait / math 皆 ~2,
  無單一魔王),逼近「H 列序列依賴」這條路的延遲地板。
- **v6 是能力終點**:在不損效能(實測還微幅更快)的前提下把寬度上限從 2048 推到 ~8192(可再調),
  讓 3840×2160 也能跑——這是涵蓋三解析度 scaling 報告的必要條件。
- **對 naive baseline(v0,每列一顆 kernel)**:v0→v6 總計 **3.01x(1428)/ 2.76x(1920)/
  1.87x(3840)**。注意 fusion 在 3840 單獨看反而退步(v2 比 v0 慢,單 SM 啃寬列),靠 v4–v6 的
  latency hiding 才反超——「先 profile、別假設 fusion 一定贏」的又一例證(見上「對照 naive baseline」)。
- **對 CPU baseline(單執行緒 C++)**:v6 約 **16.2x(1428)/ 18.8x(1920)/ 23.9x(3840)**,
  圖越大 GPU 贏越多;即使對 4 核 OpenMP CPU 仍快 5.9–7.6x。最底層的故事是
  **CPU(1thr)→ GPU naive(v0)拿走大頭(5–11x),v0→v6 再 2.2–3.1x**(見上「對照 CPU baseline」)。

## 關鍵教訓

1. **先 profile 再優化**:v2 的瓶頸是記憶體延遲(`long_scoreboard`),不是直覺以為的「SM 用太少」。
   v3 照直覺攤到多 SM 反而退步——量測救了方向。
2. **看對 metric**:occupancy 一度被誤讀成 3%(其實 ~42%,3% 是別的 metric);低 throughput +
   高 occupancy = 「warp 在但都 stall」,要看 **stall reason** 才知道卡在哪。
3. **同步成本有層級**:`__syncthreads()`(block 內)≪ `grid.sync()`(跨 SM)。序列演算法的每步
   barrier 要盡量便宜,別為了「用更多 SM」付昂貴的全域 barrier。
4. **序列依賴是硬限制**:精確 seam carving 的 H 列依賴改不掉,優化只能壓低「每步成本」
   (prefetch 藏延遲、縮小寫回),不能增加跨列平行度。
5. **報酬遞減要懂得收**:v5 後 stall 已攤平無單一目標,再往下(列內 tiling 等)對這張小圖
   ROI 不划算,適時停手。

## 後續(若要再榨)

- **列內 wavefront tiling**:單 block 內用 register halo 連算 K 列才同步一次,把 968 次
  `__syncthreads` 砍到 ~968/K。但 W=1440 偏小,halo 冗餘會吃掉好處,預期報酬有限。
- **更省的 back-pointer**:2-bit 打包(16 欄/int),寫回再 ÷4,但需 thread 間協調,複雜度高。
