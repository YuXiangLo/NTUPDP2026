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
- **驗證**:W ≤ 2048 時 v6 輸出應與 v5 bit-identical(同碼路徑);大圖則是 v6 獨有能力。
- **結果**:待多解析度 benchmark(見下)填入。

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
    bash bench.sh small.jpg medium.jpg large.jpg
python3 bench_table.py results.csv
```

---

## 總結

| 版本 | DP Duration | 對 v2 加速 | 關鍵手法 |
|---|---|---|---|
| v2 | 1.460 ms | 1.00x | 單 block 共享記憶體 DP(baseline) |
| ~~v3~~ | ~~1.657 ms~~ | ~~0.88x~~ | ~~grid.sync 跨 SM~~ → barrier-bound,廢案 |
| v4 | 0.872 ms | 1.67x | prefetch energy,消除 long_scoreboard |
| **v5** | **0.790 ms** | **1.85x** | + int8 相對 back-pointer,寫回 ÷4 |

- **DP 加速 ~1.85x**,每條 seam 整體 ~1.57 → ~0.90 ms(**~1.75x**)。
- 終點在 v5:warp stall 已完全攤平(barrier 2.47 / not_selected / wait / math 皆 ~2,無單一魔王),
  逼近「968 列序列依賴」這條路的延遲地板。

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
