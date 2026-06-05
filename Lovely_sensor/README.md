# Lovely-Haimes 法 衝撃波センサー (2D 翼まわり / SU2 後処理)

SU2 で計算した 2D 翼まわりの流れ場 (`.vtu`) に対し、
**Lovely-Haimes 法** (Lovely & Haimes, AIAA 1999-3285) を後処理として適用し、衝撃波断面を検知するスクリプトです。

## このスクリプトが行うこと

SU2 は速度・圧力・密度から法線マッハ数 M_n を計算する機能を持っています。
しかし生の M_n=1 等値線には、膨張波や数値誤差に起因する **偽検知が混入**します。

本スクリプトは Lovely-Haimes 法のフィルタリングを適用し、**真の衝撃波断面のみに絞った
スカラー場 `shock_mach` を出力**します。

```
shock_mach[i] = M_n[i]   (衝撃波検知条件を満たす節点)
shock_mach[i] = 0         (それ以外)
```

ParaView で `shock_mach = 1` の等値線を描くと、フィルタリング済みの衝撃波断面が得られます。
Wu et al. (2013) Fig. 21, 23 に示される衝撃波断面表示がこれに対応します。

## セットアップ

```bash
pip install numpy meshio pyyaml
```

### ファイル構成

```
lovely_sensor.py          ← フィルタなし版 (基本実装)
lovely_sensor_filtered.py ← Wu et al. (2013) フィルタ追加版
config.yaml               ← lovely_sensor.py の設定
config_filtered.yaml      ← lovely_sensor_filtered.py の設定
flow.vtu                  ← SU2 出力 (処理対象)
```

## 使い方

```bash
python lovely_sensor.py           # フィルタなし版
python lovely_sensor_filtered.py  # フィルタあり版 (推奨)
```

引数は不要です。設定はそれぞれの yaml ファイルで変更します。

SU2 側で速度・圧力・密度を含む出力を用意しておきます
(`VOLUME_OUTPUT= (PRIMITIVE)` 等で圧力・密度が含まれることを確認)。

## センサーの定義

衝撃波を通過すると、法線方向の流速は超音速から亜音速に落ちます。
法線方向は局所の圧力勾配 ∇p に平行とみなせるので、速度を ∇p 方向へ射影した
**法線マッハ数** M_n が 1 になる場所が衝撃波です。

```
        u · ∇p
M_n = -------------- ,    a = sqrt(γ · p / ρ)  (局所音速)
        a · |∇p|
```

### フィルタの役割

| フィルタ | 条件 | 除去対象 |
| ------- | ---- | ------- |
| 圧縮フィルタ | u·∇p > 0 | 膨張波 |
| Filter 1 (Wu eq.18) | \|∇p\| ≥ ε·p/l_n | 数値誤差・一様流域の偽検知 |
| Filter 2 (Wu eq.19) | V·∇\|V\| < 0 | 衝撃波上流側 (速度未減速領域) |

フィルタは `shock_mach` の値を変えず、**どの節点を非ゼロにするかだけを決定**します。

## 出力フィールド

### lovely_sensor.py (`*_lovely.vtu`)

| フィールド名     | 内容                                                    |
| --------------- | ------------------------------------------------------- |
| `shock_mach`    | 衝撃波断面スカラー場 (検知節点: M_n 値、それ以外: 0)    |
| `grad_p_mag`    | 圧力勾配の大きさ \|∇p\|                                 |
| `sound_speed`   | 局所音速 a                                              |

### lovely_sensor_filtered.py (`*_lovely_filtered.vtu`)

| フィールド名      | 内容                                                    |
| ---------------- | ------------------------------------------------------- |
| `shock_mach`     | 衝撃波断面スカラー場 (全フィルタ適用後)                 |
| `grad_p_mag`     | 圧力勾配の大きさ \|∇p\|                                 |
| `sound_speed`    | 局所音速 a                                              |
| `v_dot_grad_vmag`| V·∇\|V\| (Filter 2 の判定値)                           |
| `filter1_pass`   | Filter 1 通過フラグ (0/1)                               |
| `filter2_pass`   | Filter 2 通過フラグ (0/1)                               |

## ParaView での可視化

### 衝撃波断面の表示 (推奨)

1. `*_lovely_filtered.vtu` を開き `Apply`
2. **Filters → Common → Contour**
3. **Contour By**: `shock_mach`
4. **Isosurfaces**: `1.0`
5. `Apply` → 検知された衝撃波断面が表示される

### 衝撃波領域のカラー表示

1. 色付けプルダウンで `shock_mach` を選択
2. `Edit Color Map` → `Rescale to Custom Range` で範囲を `0 〜 1.5` 程度に設定
3. カラーマップを `Cool to Warm` などに変更
   - 0 (黒/青): 衝撃波外の領域
   - 1 付近 (中間色): 衝撃波断面
   - 1 超 (暖色): 強い衝撃波領域

### フィルタなし版との比較

両ファイルを開き、それぞれ Contour (`shock_mach=1`) を表示して重ねると、
フィルタの効果 (偽検知の除去) を視覚的に確認できます。

## 設定ファイル (config.yaml)

```yaml
input_file: null          # null で自動検出
multiple_files: first     # first / all
gamma: 1.4                # 比熱比 (空気)
mn_threshold: 1.0         # 法線マッハ数のしきい値
compression_filter: true  # u·∇p>0 で膨張波を除外するか
eps: 1.0e-12
output_file: null
```

## 設定ファイル (config_filtered.yaml)

```yaml
input_file: null
multiple_files: first
gamma: 1.4
mn_threshold: 1.0
filter1_enabled: true
filter1_eps: 0.001        # 論文推奨初期値。弱衝撃波が消える場合は 0.0001 に下げる
filter2_enabled: true
compression_filter: false
output_file: null
```

## スクリプト構成

| 関数                       | 役割                                                             |
| ------------------------- | ---------------------------------------------------------------- |
| `load_config()`           | yaml を読み込み、既定値とマージ                                   |
| `resolve_input_files()`   | `.vtu` ファイルを自動検出または指定から取得                       |
| `load_su2_vtu(path)`      | `.vtu` を読み、座標・速度・圧力・密度を取り出す                  |
| `nodal_gradients_lsq()`   | 非構造格子上の近傍最小二乗による節点勾配の再構成                  |
| `lovely_sensor()`         | 圧力勾配・音速から法線マッハ数 M_n を計算                        |
| `filtered_sensor()`       | M_n 計算 + Filter 1/2 を適用して `shock_mach` を生成            |
| `process_one()`           | 1 つのファイルの全処理 (読込 → 計算 → 出力)                      |
| `main()`                  | 全ファイルの処理ループ                                            |

## トラブルシューティング

| 症状 | 対処 |
| ---- | ---- |
| 「圧力場が見つかりません」 | SU2 出力に `Pressure` / `Density` が含まれるか確認 |
| `shock_mach` が全て 0 | `mn_threshold` を 0.95 に下げる / フィルタを無効化して確認 |
| 衝撃波以外の場所にも等値線が出る | `filter1_eps` を大きくする (例: 0.01) |
| 衝撃波断面に欠けがある | `filter1_eps` を小さくする (例: 0.0001) / `filter2_enabled: false` を試す |
| 等値線が複数本出る | 数値拡散で衝撃波が鈍っている → 正常。最も外側の線が真の衝撃波位置に近い |

## 参考文献

- D. Lovely, R. Haimes, "Shock Detection from Computational Fluid Dynamics Results," AIAA 1999-3285, 1999.
- Z. Wu et al., "Review of shock wave detection method in CFD post-processing," *Chinese Journal of Aeronautics*, 26(3): 501–513, 2013.
