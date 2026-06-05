# Ducros センサー (2D 翼まわり / SU2 後処理)

SU2 で計算した 2D 翼まわりの流れ場 (`.vtu`) に対し、最も単純な
**Ducros センサー** (Ducros et al., 1999) を後処理として適用し、衝撃波領域を検知するスクリプトです。
衝撃波セグメンテーションの参照ラベル生成や、AI モデルとの比較ベースラインとしての利用を想定しています。

# 1. セットアップ

# 1.1 必要環境

```bash
pip install numpy scipy meshio pyyaml
```

- `numpy` : 数値計算
- `meshio` : `.vtu` (VTK Unstructured Grid) の読み書き
- `scipy` : (オプション)
- `pyyaml` : `config.yaml` の読み込み

# 1.2 ファイル構成

同じディレクトリに以下の 2 つをセットで置いてください。

```
ducros_sensor.py    ← スクリプト本体
config.yaml         ← 設定ファイル
flow.vtu            ← SU2 出力 (処理対象)
```

# 2. 使い方

SU2 側で ParaView 形式を出力しておきます (例: `OUTPUT_FILES= (RESTART, PARAVIEW)`)。

その後、スクリプトと同じディレクトリで単に

```bash
python ducros_sensor.py
```

を実行するだけです。**引数は不要**です。

# 2.1 実行の流れ

1. `config.yaml` を読み込む
2. 同じディレクトリの `.vtu` を自動検出 (複数あれば設定に応じて処理)
3. 各 `.vtu` に対して Ducros センサーを計算
4. 結果を `<元のファイル名>_ducros.vtu` として出力
5. 出力ファイルは次回の入力候補から自動で除外

# 3. 設定ファイル (config.yaml)

パラメータの全てをこちらで管理できます。

```yaml
# 入力ファイル (null = 自動検出)
input_file: null

# 複数の .vtu があった場合の処理
#   first : 最初の 1 つだけ
#   all   : 全て順に処理
multiple_files: first

# ---- センサーパラメータ ----
# 圧縮のみを分子に拾うか
compression_only: true

# ゼロ割回避用の微小量
eps: 1.0e-12

# 衝撃波判定の閾値 (0～1)
threshold: 0.95

# 出力ファイル名 (null = 自動生成)
output_file: null
```

# 3.1 よく使う変更

- **複数の `.vtu` を一度に処理** : `multiple_files: all`
- **閾値を下げて衝撃波検知の感度を上げる** : `threshold: 0.90`
- **膨張波も拾いたい** : `compression_only: false`

# 4. 出力

入力メッシュに以下の節点データを追加した `<名前>_ducros.vtu` を生成します。ParaView で色付け表示できます。

| フィールド名     | 内容                                    |
| ----------- | --------------------------------------- |
| `ducros_phi`     | Ducros センサー値 (0～1)                |
| `divergence`     | 速度の発散 ∇·u                          |
| `vorticity_z`    | 渦度 ω_z                                |
| `shock_mask`     | `phi >= threshold` を満たす節点 (0/1)   |

実行時にはコンソールに以下が表示されます。

```
--- flow.vtu ---
  節点数            : 2500
  phi の範囲         : [0.000, 1.000]
  衝撃波判定節点数   : 758  (threshold=0.95)
  出力               : flow_ducros.vtu
完了 (ParaView で ducros_phi / shock_mask を可視化)
```

# 5. センサーの定義

衝撃波 (圧縮 ∇·u < 0) を 1 に近づける標準形を採用しています。

```
        max(-∇·u, 0)
phi = ----------------------
       |∇·u| + |ω_z| + eps
```

- `phi ≈ 1` : 強い圧縮 → **衝撃波**
- `phi ≈ 0` : 渦・滑らかな領域

ここで 2D の発散と渦度は次の通りです。

```
∇·u  = ∂u/∂x + ∂v/∂y
ω_z  = ∂v/∂x − ∂u/∂y
```

`eps` はゼロ割回避用の微小量です。分子で圧縮成分のみを拾うため、渦度優位の領域では分母が大きくなり `phi` が小さく抑えられます。膨張側も拾いたい場合は `compression_only: false` で `|∇·u|` を分子に使えます。

# 6. スクリプト構成

| 関数                     | 役割                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `load_config()`         | `config.yaml` を読み込み、既定値とマージ                              |
| `resolve_input_files()` | `.vtu` ファイルを自動検出または指定から取得                           |
| `load_su2_vtu(path)`    | `.vtu` を読み、座標と速度場 (u, v) を取り出す                        |
| `nodal_gradients_lsq()`| 非構造格子上で近傍節点との最小二乗フィットにより節点勾配を再構成    |
| `ducros_sensor()`       | 発散・渦度から `phi` を計算                                          |
| `process_one()`         | 1 つのファイルの全処理 (読込 → 計算 → 出力)                          |
| `main()`                | ドライバー：全ファイルの処理ループ                                    |

速度フィールド名は SU2 のバージョン・設定で揺れるため、`Velocity` / `Velocity_x` / `Momentum` などの候補を自動探索します。見つからない場合は利用可能なフィールド一覧を表示してエラー終了します。

# 7. トラブルシューティング

# 7.1 「.vtu ファイルが見つかりません」

- スクリプトと同じディレクトリに `.vtu` を置いているか確認
- ファイル名が本当に `.vtu` か確認 (大文字小文字も同じ)
- 別のディレクトリにある場合は `config.yaml` の `input_file` に相対パスを指定

# 7.2 「速度場が見つかりません」

SU2 のバージョンによってフィールド名が違う場合があります。エラーメッセージに「利用可能なフィールド: ...」と出ていたら、`load_su2_vtu` の探索候補に追加するか、直接フィールド名を指定する必要があります。

# 7.3 衝撃波が検知されない

- `threshold` を下げてみる (例: `0.90` → `0.85`)
- `ducros_phi` の実際の分布を ParaView で色付けして確認
- センサーの計算結果 `divergence` や `vorticity_z` が正しく計算されているか確認

## 注意点

- **勾配の精度**: 本スクリプトは非構造格子上で近傍節点の最小二乗フィットにより勾配を再構成しています。SU2 は設定次第で速度勾配を直接出力できるため、本番ではそちらを使う方が精度・速度の面で有利です。
- **閾値**: `0.95` はあくまで目安です。Ducros センサーは本来 0/1 マスクではなく人工粘性のスイッチとして連続値で使うものなので、実データで `ducros_phi` の分布を確認してから閾値を決めることを推奨します。
- **対象**: 2D を前提としています (3D 化する場合は渦度をベクトルとして扱い `|ω|` を計算してください)。
- **出力の自動除外**: `*_ducros.vtu` の出力ファイルは自動的に再入力候補から除外されます。同じディレクトリで繰り返し実行しても大丈夫です。

## 参考文献

- F. Ducros et al., "Large-Eddy Simulation of the Shock/Turbulence Interaction," *Journal of Computational Physics*, vol. 152, pp. 517–549, 1999.
- T. A. Hendrickson, A. Kartha, G. V. Candler, "An improved Ducros sensor for the simulation of compressible flows with shocks," AIAA 2018-3710, 2018.
