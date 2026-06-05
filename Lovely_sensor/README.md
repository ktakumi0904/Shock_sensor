# Lovely-Haimes 法 衝撃波センサー (2D 翼まわり / SU2 後処理)

SU2 で計算した 2D 翼まわりの流れ場 (`.vtu`) に対し、
**Lovely-Haimes 法** (Lovely & Haimes, AIAA 1999-3285) を後処理として適用し、衝撃波領域を検知するスクリプトです。
衝撃波セグメンテーションの参照ラベル生成や、Ducros センサーとの比較ベースラインとしての利用を想定しています。

## Ducros センサーとの違い

| 項目         | Ducros センサー              | Lovely-Haimes 法                    |
| ----------- | ---------------------------- | ----------------------------------- |
| 使う物理量   | 速度のみ (発散・渦度)         | 速度 + 圧力 + 密度                  |
| 判定の根拠   | 圧縮 (∇·u<0) かつ低渦度       | 法線マッハ数 M_n = 1                |
| 衝撃波の向き | 考慮しない                   | 圧力勾配方向を法線として考慮         |
| 膨張波の除外 | 圧縮成分のみ拾う              | u·∇p>0 のフィルタで除外             |
| 出力値       | phi (0～1)                   | M_n (法線マッハ数, 0～)             |

Lovely-Haimes 法は衝撃波の "向き" を圧力勾配で捉えるため、物理的により厳密です。その分、圧力と密度のフィールドが必要になります。

## セットアップ

### 必要環境

```bash
pip install numpy scipy meshio pyyaml
```

### ファイル構成

```
lovely_sensor.py        ← スクリプト本体
config_lovely.yaml      ← 設定ファイル
flow.vtu                ← SU2 出力 (処理対象)
```

## 使い方

SU2 側で **速度・圧力・密度** を含む ParaView 形式を出力しておきます
(`VOLUME_OUTPUT= (PRIMITIVE)` などで圧力・密度が含まれることを確認)。

スクリプトと同じディレクトリで

```bash
python lovely_sensor.py
```

を実行するだけです。**引数は不要**です。

## センサーの定義

衝撃波を通過すると、衝撃波に垂直な流速は超音速から亜音速に落ちます。衝撃波の法線方向は局所の圧力勾配 ∇p に平行とみなせるので、速度を ∇p 方向へ射影した **法線マッハ数** が 1 になる場所が衝撃波です。

```
        u · ∇p
M_n = -------------- ,    a = sqrt(γ · p / ρ)  (局所音速)
        a · |∇p|
```

- `M_n ≥ 1` : 衝撃波候補 (法線方向が超音速)
- `u · ∇p > 0` : 圧縮 (流れ方向に圧力上昇) → 膨張波を除外するフィルタ

両方を満たす節点を衝撃波と判定します。

## 設定ファイル (config_lovely.yaml)

```yaml
# 入力ファイル (null = 自動検出)
input_file: null

# 複数 .vtu の処理 (first / all)
multiple_files: first

# ---- センサーパラメータ ----
gamma: 1.4                  # 比熱比 (空気)
mn_threshold: 1.0           # 法線マッハ数のしきい値
compression_filter: true    # u·∇p>0 で膨張波を除外するか
eps: 1.0e-12

# 出力ファイル名 (null = 自動生成)
output_file: null
```

### よく使う変更

- **不連続が鈍って検知が甘い** : `mn_threshold: 0.95` に下げる
- **膨張波もまとめて見たい** : `compression_filter: false`
- **空気以外の気体** : `gamma` を変更

## 出力

入力メッシュに以下の節点データを追加した `<名前>_lovely.vtu` を生成します。

| フィールド名     | 内容                                    |
| ----------- | --------------------------------------- |
| `normal_mach`    | 法線マッハ数 M_n                        |
| `grad_p_mag`     | 圧力勾配の大きさ \|∇p\|                 |
| `sound_speed`    | 局所音速 a                              |
| `shock_mask`     | 衝撃波判定 (0/1)                        |

実行時にはコンソールに以下が表示されます。

```
--- flow.vtu ---
  節点数            : 3600
  M_n の範囲         : [0.000, 1.690]
  衝撃波判定節点数   : 1140  (M_n>=1.0, u·∇p>0)
  出力               : flow_lovely.vtu
完了 (ParaView で normal_mach / shock_mask を可視化)
```

## スクリプト構成

| 関数                     | 役割                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `load_config()`         | `config_lovely.yaml` を読み込み、既定値とマージ                      |
| `resolve_input_files()` | `.vtu` ファイルを自動検出または指定から取得                           |
| `load_su2_vtu(path)`    | `.vtu` を読み、座標・速度・圧力・密度を取り出す                      |
| `nodal_gradients_lsq()`| 非構造格子上で近傍節点との最小二乗フィットにより節点勾配を再構成    |
| `lovely_sensor()`       | 圧力勾配・音速から法線マッハ数 M_n を計算                            |
| `process_one()`         | 1 つのファイルの全処理 (読込 → 計算 → 出力)                          |
| `main()`                | ドライバー：全ファイルの処理ループ                                    |

## トラブルシューティング

### 「圧力場が見つかりません」「密度場が見つかりません」

Ducros 版と違い、Lovely-Haimes 法は圧力 `p` と密度 `ρ` が必須です。SU2 の出力に含まれているか確認してください。エラーメッセージに利用可能なフィールド一覧が出るので、フィールド名 (`Pressure` / `Density` など) を `load_su2_vtu` の探索候補に追加できます。

### 衝撃波が過剰に検知される / されない

- `mn_threshold` を調整 (`1.0` を基準に上下)
- `compression_filter: true` で膨張波が除外されているか確認
- `normal_mach` を ParaView で色付けし、M_n=1 の等値線が衝撃波位置に来ているか目視確認

## 注意点

- **必要フィールド**: 速度・圧力・密度の 3 つが必須。Ducros 版より要求が多い。
- **勾配の精度**: 圧力勾配 ∇p の計算精度が検知精度を左右します。本実装は近傍最小二乗で再構成しています。SU2 が圧力勾配を直接出力できる場合はそちらが有利です。
- **音速の計算**: `a = sqrt(γ p / ρ)` は完全気体を仮定しています。無次元化された SU2 データでは p, ρ のスケールに注意してください。
- **しきい値依存**: 法線マッハ数法は本質的にしきい値の設定に敏感です (先行研究でも指摘あり)。膨張波や弱い圧縮波の混入を防ぐにはフィルタリングが重要です。

## 参考文献

- D. Lovely, R. Haimes, "Shock Detection from Computational Fluid Dynamics Results," AIAA 1999-3285, 14th Computational Fluid Dynamics Conference, Norfolk, VA, 1999.
- M. Kanamori, K. Suzuki, "Shock wave detection in two-dimensional flow based on the theory of characteristics from CFD data," *Journal of Computational Physics*, vol. 230, pp. 3085–3092, 2011. (法線マッハ数法の課題と代替手法)
