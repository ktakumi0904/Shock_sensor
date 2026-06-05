"""
Kanamori-Suzuki 衝撃波センサー (Kanamori & Suzuki, JCP 2011) の練習用実装
対象: SU2 で計算した 2D 翼まわりの流れ場 (非構造格子 .vtu)

--------------------------------------------------------------------
原理 (Ducros との違い):
  Ducros は「圧縮 (div u < 0) かつ渦度が小さい」領域を経験的閾値で拾う。
  Kanamori-Suzuki (以下 KS) は特性曲線理論に基づき、衝撃波を
  「特性線 (マッハ線) の収束」として捉える。原理的に経験的閾値が不要。

  2D 定常流では、流れ方向に対してマッハ角
        mu = arcsin(1 / M)        (M = マッハ数, M > 1 でのみ実数)
  だけ傾いた 2 本のマッハ線が存在する。マッハ線が実数で定義できるのは
  M > 1 の超音速領域のみ。つまり本センサーは超音速領域でのみ有効。

  検出の核心 (法線マッハ数のゼロクロス):
  圧力勾配 grad(p) を衝撃波面の法線方向とみなすと、その法線方向の
  流入マッハ数は
        Mn = M * cos(alpha),   cos(alpha) = (V·grad p)/(|V||grad p|)
  衝撃波を横切ると Mn は上流で >1、下流で <1 となり、衝撃波内部で
  必ず Mn = 1 を横切る (Rankine-Hugoniot の帰結)。この Mn=1 の
  ゼロクロスを検出する。判定境界は Mn=1 で固定 = 経験的閾値が不要。
  膨張波は流れが低圧側へ向かう (cos(alpha)<0) ので自然に除外される。

  ★ 重要な制約 ★
  M > 1 でしか評価できない。翼まわり遷音速流では、超音速ポケット内部
  だけが検出対象になる (亜音速領域は ks_phi = 0, valid = 0)。

  ※ 本実装は KS 法の核心 (特性線収束 = 法線マッハ数の音速横断) を
     再現した実用版。原論文は接触不連続/膨張波の厳密な区別など、
     より精緻な特性線追跡の手続きを含む。
--------------------------------------------------------------------
使い方:
    同じディレクトリに config.yaml を置いて
        python kanamori_suzuki_sensor.py
    だけで実行できる。設定は config.yaml 側で変更する。
--------------------------------------------------------------------
"""

import os
import glob
import numpy as np
import meshio
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "config.yaml"


# ----------------------------------------------------------------------
# 0. 設定ファイルの読み込み / 入出力パスの解決
# ----------------------------------------------------------------------
def load_config():
    """同じディレクトリの config.yaml を読み込む。無ければ既定値を返す。"""
    defaults = {
        "input_file": None,
        "multiple_files": "first",
        "gamma": 1.4,          # 比熱比
        "eps": 1e-12,
        "output_file": None,
    }
    cfg_path = os.path.join(SCRIPT_DIR, CONFIG_NAME)
    if not os.path.exists(cfg_path):
        print(f"[警告] {CONFIG_NAME} が見つかりません。既定値で実行します。")
        return defaults
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    for k, v in user_cfg.items():
        defaults[k] = v
    return defaults


def resolve_input_files(cfg):
    """処理対象の .vtu パスのリストを決める。"""
    if cfg.get("input_file"):
        path = cfg["input_file"]
        if not os.path.isabs(path):
            path = os.path.join(SCRIPT_DIR, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定された input_file が見つかりません: {path}")
        return [path]

    found = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.vtu")))
    found = [f for f in found if not f.endswith("_ks.vtu")]   # 自分の出力は除外
    if not found:
        raise FileNotFoundError(f"{SCRIPT_DIR} に .vtu ファイルが見つかりません。")

    if cfg.get("multiple_files", "first") == "all":
        return found
    if len(found) > 1:
        print(f"[情報] .vtu が {len(found)} 個見つかりました。最初の 1 つを処理します: "
              f"{os.path.basename(found[0])}")
        print(f"       全部処理したい場合は config.yaml の multiple_files を all に。")
    return [found[0]]


def make_output_path(cfg, in_path):
    """出力パスを決める。設定が null なら入力名から自動生成 (例 flow.vtu -> flow_ks.vtu)。"""
    if cfg.get("output_file"):
        out = cfg["output_file"]
        return out if os.path.isabs(out) else os.path.join(SCRIPT_DIR, out)
    base, _ = os.path.splitext(in_path)
    return base + "_ks.vtu"


# ----------------------------------------------------------------------
# 1. データ読み込み (速度・圧力・密度)
# ----------------------------------------------------------------------
def load_su2_vtu(path):
    """SU2 出力の .vtu を読み、座標・速度・圧力・密度を取り出す。"""
    mesh = meshio.read(path)
    points = mesh.points[:, :2]
    pdata = mesh.point_data

    def find(*names):
        for n in names:
            if n in pdata:
                return n
        return None

    # --- 速度 ---
    vkey = find("Velocity", "velocity", "Momentum")
    if vkey:
        vel = np.asarray(pdata[vkey])
        u, v = vel[:, 0], vel[:, 1]
    else:
        uk = find("Velocity_x", "X-Velocity", "Momentum_x")
        vk = find("Velocity_y", "Y-Velocity", "Momentum_y")
        if uk is None or vk is None:
            raise KeyError(f"速度場が見つかりません。利用可能フィールド: {list(pdata.keys())}")
        u, v = np.asarray(pdata[uk]), np.asarray(pdata[vk])

    # --- 圧力 ---
    pk = find("Pressure", "pressure", "p")
    if pk is None:
        raise KeyError(f"圧力場が見つかりません。利用可能フィールド: {list(pdata.keys())}")
    p = np.asarray(pdata[pk])

    # --- 密度 ---
    dk = find("Density", "density", "rho")
    if dk is None:
        raise KeyError(f"密度場が見つかりません。利用可能フィールド: {list(pdata.keys())}")
    rho = np.asarray(pdata[dk])

    return mesh, points, u, v, p, rho


# ----------------------------------------------------------------------
# 2. 非構造格子上の節点勾配 (近傍最小二乗) ― Ducros 版と同じ
# ----------------------------------------------------------------------
def build_neighbors(points, cells):
    """各節点の近傍節点インデックス集合のリストを作る。"""
    N = len(points)
    neigh = [set() for _ in range(N)]
    for block in cells:
        for elem in block.data:
            for a in elem:
                for b in elem:
                    if a != b:
                        neigh[a].add(b)
    return neigh


def nodal_gradient(points, neigh, field):
    """近傍最小二乗で節点勾配 (df/dx, df/dy) を求める。"""
    N = len(points)
    grad = np.zeros((N, 2))
    for i in range(N):
        nb = list(neigh[i])
        if len(nb) < 2:
            continue
        dx = points[nb] - points[i]
        df = field[nb] - field[i]
        try:
            g, *_ = np.linalg.lstsq(dx, df, rcond=None)
            grad[i] = g
        except np.linalg.LinAlgError:
            pass
    return grad[:, 0], grad[:, 1]


# ----------------------------------------------------------------------
# 3. Kanamori-Suzuki センサー
# ----------------------------------------------------------------------
def kanamori_suzuki_sensor(points, cells, u, v, p, rho, gamma=1.4, eps=1e-12):
    """
    特性線理論に基づく衝撃波検知 (KS 法の核心を再現した実装)。

    考え方:
      圧力勾配方向 grad(p) を衝撃波面の法線方向とみなす。
      この法線方向の流入マッハ数
            Mn = M * cos(alpha),   cos(alpha) = (V·grad p)/(|V||grad p|)
      は、衝撃波を横切ると上流で Mn>1、下流で Mn<1 となり、
      衝撃波内部で必ず Mn = 1 を横切る (Rankine-Hugoniot の帰結)。
      この「Mn-1 のゼロクロス」を検出する。閾値は 1 で固定 (経験パラメータ不要)。

      マッハ角 mu = arcsin(1/M) は M>1 でのみ実数なので、
      評価は超音速領域に限られる (亜音速領域は valid=0)。

    返り値:
      ks_phi : 衝撃波らしさ (0〜1)。Mn=1 ゼロクロスの強さ
      mask   : 衝撃波判定 (0/1)
      mach   : マッハ数
      valid  : 超音速 (M>1) かつ評価可能な節点 (0/1)
    """
    neigh = build_neighbors(points, cells)

    # --- マッハ数・音速 ---
    speed = np.sqrt(u**2 + v**2)
    a = np.sqrt(np.maximum(gamma * p / np.maximum(rho, eps), 0.0))   # 音速
    mach = speed / np.maximum(a, eps)
    supersonic = mach > 1.0

    # --- 圧力勾配 (衝撃波面の法線方向の推定) ---
    dpdx, dpdy = nodal_gradient(points, neigh, p)
    gradp = np.sqrt(dpdx**2 + dpdy**2)

    # --- 法線方向の流入マッハ数 Mn = M cos(alpha) ---
    # cos(alpha) = (V·grad p)/(|V||grad p|) : 流れが圧力増加方向へ向かう度合い
    cos_alpha = (u * dpdx + v * dpdy) / (np.maximum(speed, eps) * np.maximum(gradp, eps))

    # 圧力勾配がほぼ無い領域 (一様流) では法線方向が定義できず cos_alpha がノイズ化する。
    # こうした点は衝撃波ではないので、cos_alpha=1 (=超音速のまま) として保護する。
    # ※ これは衝撃波の経験的閾値ではなく、方向ベクトルが不定な点を除く数値的保護。
    sup_gradp = gradp[supersonic]
    gradp_ref = sup_gradp.max() if sup_gradp.size else 0.0
    weak = gradp < 0.01 * gradp_ref
    cos_alpha = np.where(weak, 1.0, cos_alpha)

    Mn = mach * cos_alpha
    f = Mn - 1.0           # これが衝撃波内部でゼロクロスする

    # --- ゼロクロス検出 ---
    # 圧縮 (流れが高圧側へ: cos_alpha>0) かつ、近傍との間で f の符号が反転する点を衝撃波とする。
    # Mn=1 のゼロクロスは「超音速→亜音速」の境界で起きるため、
    # 近傍は超音速に限定せず全点を見る。自分は超音速 (f>0)、近傍に Mn<1 (f<0) が
    # あれば、その間で法線マッハ数が 1 を横切る = 衝撃波を挟んでいる。
    N = len(points)
    ks_phi = np.zeros(N)
    for i in range(N):
        if not supersonic[i] or cos_alpha[i] <= 0:   # 評価は超音速かつ圧縮側のみ
            continue
        nb = list(neigh[i])
        if not nb:
            continue
        fi = f[i]                       # > 0 (超音速)
        fj = f[nb]
        crossing = fj < 0.0             # 近傍に法線方向が亜音速 (Mn<1) の点がある
        if np.any(crossing):
            # ジャンプの大きさを指標に (圧力勾配の強さで重み付け)
            ks_phi[i] = np.max(np.abs(fi - fj[crossing])) * gradp[i]

    # 正規化 (超音速領域内の最大値で 0〜1 に)
    sup_vals = ks_phi[supersonic]
    if sup_vals.size and sup_vals.max() > 0:
        ks_phi = ks_phi / sup_vals.max()

    valid = supersonic.astype(np.int32)
    mask = (ks_phi > 0).astype(np.int32)
    return ks_phi, mask, mach, valid


# ----------------------------------------------------------------------
# 4. 1 ファイルの処理
# ----------------------------------------------------------------------
def process_one(in_path, cfg):
    mesh, points, u, v, p, rho = load_su2_vtu(in_path)
    ks_phi, mask, mach, valid = kanamori_suzuki_sensor(
        points, mesh.cells, u, v, p, rho,
        gamma=float(cfg["gamma"]),
        eps=float(cfg["eps"]),
    )

    mesh.point_data["ks_phi"] = ks_phi
    mesh.point_data["shock_mask"] = mask
    mesh.point_data["mach"] = mach
    mesh.point_data["supersonic"] = valid

    out_path = make_output_path(cfg, in_path)
    meshio.write(out_path, mesh)

    n_sup = int(valid.sum())
    print(f"--- {os.path.basename(in_path)} ---")
    print(f"  節点数            : {len(points)}")
    print(f"  超音速節点数       : {n_sup}  (M>1 のみ評価対象)")
    print(f"  マッハ数の範囲     : [{mach.min():.3f}, {mach.max():.3f}]")
    print(f"  衝撃波判定節点数   : {int(mask.sum())}")
    print(f"  出力               : {os.path.basename(out_path)}")
    return ks_phi, mask


# ----------------------------------------------------------------------
# 5. メイン
# ----------------------------------------------------------------------
def main():
    cfg = load_config()
    targets = resolve_input_files(cfg)
    for in_path in targets:
        process_one(in_path, cfg)
    print("完了 (ParaView で ks_phi / shock_mask / mach を可視化)")


if __name__ == "__main__":
    main()
