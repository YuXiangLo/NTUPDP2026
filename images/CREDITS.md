# 來源影像 (Source Images) — 出處與授權

這些是**原生高解析來源圖**，用於由 `bench/make_dataset.py` 降採樣產出標準測資
（small / 1080p / 4K / 8K）。命名規則：`<內容>_<原生寬x高>.jpg`。

| 目前檔名 | 原始檔名 | 內容 | 複雜度類別 | 來源 / 授權（投稿前需核對） |
|---|---|---|---|---|
| `broadway_tower_1428x968.jpg` | Broadway_tower_edit.jpg | 建築塔樓 | 幾何結構（控制圖） | Wikimedia Commons «Broadway tower edit»（拍攝 Nikon D70）；確認 CC 授權 |
| `forest_pano_13583x5417.jpg` | Sample-jpg-image-30mb-16.jpg | 森林全景 | 高密度紋理 | 「30 MB sample JPEG」測試圖；**疑似測試用途，投稿前須確認可商用/可印** |
| `desert_mesa_4611x8192.jpg` | pexels-alex-ning-523843601-34945594.jpg | 沙漠岩柱 | 結構+平滑天空（8K 級） | Pexels，攝影者 Alex Ning（Pexels License，免費可商用、建議標註） |
| `golden_gate_2880x1620.jpg` | wallpaperswide.com-golden-gate-bridge-san-francisco-panorama-wallpaper-2880x1620.jpg | 金門大橋全景 | 線條/邊緣 | wallpaperswide.com；**桌布站來源，授權不明，投稿前須替換或確認** |
| `iceland_waterfall_2880x1620.jpg` | wallpaperswide.com-iceland-hidden-waterfall-wallpaper-2880x1620.jpg | 冰島瀑布 | 自然紋理+結構 | wallpaperswide.com；**同上，授權須確認** |

## ⚠️ 投稿前授權檢核
- **可安全使用**：`desert_mesa`（Pexels License，標註攝影者即可）。
- **須確認/可能替換**：`forest_pano`（sample 測試圖）、`golden_gate` 與 `iceland_waterfall`（桌布站）。
  論文若要放出這些圖，建議改用明確 CC0/Pexels/Unsplash 來源；但**純計時不需公開圖**（時間與內容無關），故僅展示圖需要乾淨授權。

## 來源圖涵蓋範圍
- **橫式高解析**：`forest_pano`（13583×5417, ~73.6 MP）→ 可乾淨降採樣到 8K/4K/1080p 橫式。
- **直式 8K 級**：`desert_mesa`（4611×8192, ~37.8 MP）→ 旋轉裁切後可做橫式 8K，或保留直式。
- **中解析 16:9**：`golden_gate`、`iceland_waterfall`（2880×1620）→ 4K 以下測資。
- **小控制圖**：`broadway_tower`（1428×968）→ 正確性與小尺寸對照。
