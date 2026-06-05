"""
Lovely-Haimes 法による衝撃波検知 + Wu et al. (2013) フィルタ版
論文: Wu Ziniu et al., "Review of shock wave detection method in CFD
      post-processing," Chinese Journal of Aeronautics, 26(3): 501-513, 2013
      Section 4.3 実装

基本アルゴリズム (Lovely & Haimes, AIAA 1999-3285):
    法線マッハ数 M_n = (V·∇p) / (a|∇p|) の等値面 M_n = 1 が衝撃波。

追加フィルタ (Wu et al. 2013, Section 4.3):
    Filter 1 (Eq. 18):  |∇p| >= eps_f * p / l_n
        数値誤差・一様流域の偽検知を除去。
        l_n: 各節点で ∇p 方向への隣接節点の射影距離の平均 (局所メッシュサイズ)
        eps_f: フィルタ閾値 (論文推奨初期値 0.001; config_filtered.yaml の filter1_eps で調整)

    Filter 2 (Eq. 19):  V·∇|V| < 0
        衝撃波を通過すると流速が減少するという物理的事実を利用し、
        衝撃波の上流側 (速度未減速) を除去して衝撃波面のみを残す。

出力フィールド (7 つ):
    normal_mach      : 法線マッハ数 M_n (連続値; 等値線 M_n=1 が衝撃波)
    grad_p_mag       : 圧力勾配の大きさ |∇p|
    sound_speed      : 局所音速 a = sqrt(gamma*p/rho)
    v_dot_grad_vmag  : V·∇|V| (Filter 2 の判定値; 符号が重要)
    filter1_pass     : Filter 1 通過フラグ (0/1)
    filter2_pass     : Filter 2 通過フラグ (0/1)
    shock_mask       : 最終的な衝撃波判定 (0/1)

使い方:
    同じディレクトリに config_filtered.yaml を置いて
        python lovely_sensor_filtered.py
    で実行。
"""

import os
import glob
import numpy as np
import meshio
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "config_filtered.yaml"


# ------------------------------------------------------------------
# 0. 設定ファイル読み込み
# ------------------------------------------------------------------
def load_config():
    defaults = {
        "input_file": None,
        "multiple_files": "first",
        "gamma": 1.4,
        "mn_threshold": 1.0,
        "eps": 1e-12,
        "output_file": None,
        "filter1_enabled": True,
        "filter1_eps": 0.001,
        "filter2_enabled": True,
        "compression_filter": False,
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
    if cfg.get("input_file"):
        path = cfg["input_file"]
        if not os.path.isabs(path):
            path = os.path.join(SCRIPT_DIR, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定された input_file が見つかりません: {path}")
        return [path]

    found = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.vtu")))
    found = [f for f in found if not any(
        f.endswith(s) for s in ("_lovely.vtu", "_ducros.vtu", "_filtered.vtu", "_lovely_filtered.vtu")
    )]
    if not found:
        raise FileNotFoundError(f"{SCRIPT_DIR} に .vtu ファイルが見つかりません。")

    if cfg.get("multiple_files", "first") == "all":
        return found
    if len(found) > 1:
        print(f"[情報] .vtu が {len(found)} 個見つかりました。最初の 1 つを処理します: "
              f"{os.path.basename(found[0])}")
    return [found[0]]


def make_output_path(cfg, in_path):
    if cfg.get("output_file"):
        out = cfg["output_file"]
        return out if os.path.isabs(out) else os.path.join(SCRIPT_DIR, out)
    base, _ = os.path.splitext(in_path)
    return base + "_lovely_filtered.vtu"


# ------------------------------------------------------------------
# 1. データ読み込み
# ------------------------------------------------------------------
def load_su2_vtu(path):
    """SU2 出力の .vtu を読み、座標・速度・圧力・密度を取り出す。"""
    from meshio.vtu._vtu import VtuReader

    reader = VtuReader(path)
    n_pts = len(reader.points)
    clean_pd = {k: v for k, v in reader.point_data.items() if len(v) == n_pts}
    skipped = [k for k in reader.point_data if k not in clean_pd]
    if skipped:
        print(f"[警告] 節点数({n_pts})と長さが一致しないフィールドをスキップ: {skipped}")
    mesh = meshio.Mesh(
        reader.points,
        reader.cells,
        point_data=clean_pd,
        cell_data=reader.cell_data,
        field_data=reader.field_data,
    )
    points = mesh.points[:, :2]
    pdata = mesh.point_data

    def find(*names):
        for n in names:
            if n in pdata:
                return n
        return None

    vkey = find("Velocity", "velocity", "Momentum")
    if vkey:
        vel = np.asarray(pdata[vkey])
        u, v = vel[:, 0], vel[:, 1]
    else:
        ux = find("Velocity_x", "X-Velocity", "Momentum_x")
        vy = find("Velocity_y", "Y-Velocity", "Momentum_y")
        if ux is None or vy is None:
            raise KeyError(f"速度場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
        u, v = np.asarray(pdata[ux]), np.asarray(pdata[vy])

    pkey = find("Pressure", "pressure", "p")
    if pkey is None:
        raise KeyError(f"圧力場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
    p = np.asarray(pdata[pkey]).squeeze()

    rkey = find("Density", "density", "rho", "Rho")
    if rkey is None:
        raise KeyError(f"密度場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
    rho = np.asarray(pdata[rkey]).squeeze()

    return mesh, points, u, v, p, rho


# ------------------------------------------------------------------
# 2. 非構造格子節点勾配 (LSQ)
# ------------------------------------------------------------------
def build_neighbors(cells, N):
    """セル情報から節点の隣接集合リストを構築する。"""
    neigh = [set() for _ in range(N)]
    for block in cells:
        conn = block.data
        for elem in conn:
            for a in elem:
                for b in elem:
                    if a != b:
                        neigh[a].add(b)
    return neigh


def nodal_gradients_lsq(points, neigh, field):
    """
    各節点で近傍との差分を最小二乗フィットして勾配 (df/dx, df/dy) を求める。
    field は 1D 配列を前提とする。
    """
    field = np.asarray(field).ravel()
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
            grad[i] = np.asarray(g).ravel()[:2]
        except np.linalg.LinAlgError:
            pass
    return grad[:, 0], grad[:, 1]


# ------------------------------------------------------------------
# 3. ∇p 方向の局所メッシュサイズ l_n (Wu et al. 2013, Eq. 18)
# ------------------------------------------------------------------
def local_mesh_size_in_grad_dir(points, neigh, dpdx, dpdy, eps=1e-12):
    """
    各節点での ∇p 方向の局所メッシュサイズ l_n を計算する。

    ∇p の単位法線方向 n = ∇p/|∇p| に隣接節点のオフセットを射影し、
    その絶対値の平均を l_n とする。
    |∇p| が微小な節点では平均エッジ長にフォールバックする。
    """
    N = len(points)
    ln = np.zeros(N)
    grad_mag = np.sqrt(dpdx**2 + dpdy**2)

    for i in range(N):
        nb = list(neigh[i])
        if not nb:
            ln[i] = 1.0
            continue
        dx = points[nb] - points[i]
        gm = grad_mag[i]
        if gm > eps:
            nx, ny = dpdx[i] / gm, dpdy[i] / gm
            proj = np.abs(dx[:, 0] * nx + dx[:, 1] * ny)
            valid = proj[proj > eps]
            if len(valid) > 0:
                ln[i] = float(np.mean(valid))
            else:
                ln[i] = float(np.mean(np.linalg.norm(dx, axis=1)))
        else:
            ln[i] = float(np.mean(np.linalg.norm(dx, axis=1)))

    return ln


# ------------------------------------------------------------------
# 4. Lovely-Haimes センサー + Wu et al. (2013) フィルタ
# ------------------------------------------------------------------
def filtered_sensor(points, cells, u, v, p, rho, cfg):
    """
    M_n を計算し、Filter 1 (Eq. 18) と Filter 2 (Eq. 19) を適用して
    衝撃波マスクを返す。

    Returns
    -------
    Mn              : 法線マッハ数
    grad_p_mag      : |∇p|
    a               : 局所音速
    v_dot_grad_vmag : V·∇|V|  (Filter 2 判定値)
    filter1         : Filter 1 通過フラグ (int32, 0/1)
    filter2         : Filter 2 通過フラグ (int32, 0/1)
    shock_mask      : 最終衝撃波判定 (int32, 0/1)
    """
    gamma = float(cfg["gamma"])
    eps   = float(cfg["eps"])
    mn_threshold = float(cfg["mn_threshold"])

    neigh = build_neighbors(cells, len(points))

    # ---- 圧力勾配 ∇p と法線マッハ数 M_n ----
    dpdx, dpdy = nodal_gradients_lsq(points, neigh, p)
    grad_p_mag  = np.sqrt(dpdx**2 + dpdy**2)
    u_dot_gradp = u * dpdx + v * dpdy
    a  = np.sqrt(np.maximum(gamma * p / np.maximum(rho, eps), 0.0))
    Mn = u_dot_gradp / (a * grad_p_mag + eps)

    # ---- 速度大きさ |V| の勾配と V·∇|V| ----
    vmag = np.sqrt(u**2 + v**2)
    dvdx, dvdy      = nodal_gradients_lsq(points, neigh, vmag)
    v_dot_grad_vmag = u * dvdx + v * dvdy

    # ---- 基本マスク: M_n >= threshold ----
    mask = (Mn >= mn_threshold)

    # 元の圧縮フィルタ u·∇p > 0 (オプション; 既定は無効)
    if bool(cfg.get("compression_filter", False)):
        mask = mask & (u_dot_gradp > 0)

    # ---- Filter 1: |∇p| >= eps_f * p / l_n  (Wu et al. 2013, Eq. 18) ----
    filter1 = np.ones(len(points), dtype=bool)
    if bool(cfg.get("filter1_enabled", True)):
        eps_f = float(cfg.get("filter1_eps", 0.001))
        ln    = local_mesh_size_in_grad_dir(points, neigh, dpdx, dpdy, eps)
        thr1  = eps_f * np.maximum(p, eps) / np.maximum(ln, eps)
        filter1 = (grad_p_mag >= thr1)
        mask    = mask & filter1

    # ---- Filter 2: V·∇|V| < 0  (Wu et al. 2013, Eq. 19) ----
    filter2 = np.ones(len(points), dtype=bool)
    if bool(cfg.get("filter2_enabled", True)):
        filter2 = (v_dot_grad_vmag < 0)
        mask    = mask & filter2

    return (
        Mn, grad_p_mag, a,
        v_dot_grad_vmag,
        filter1.astype(np.int32),
        filter2.astype(np.int32),
        mask.astype(np.int32),
    )


# ------------------------------------------------------------------
# 5. 1 ファイルの処理
# ------------------------------------------------------------------
def process_one(in_path, cfg):
    mesh, points, u, v, p, rho = load_su2_vtu(in_path)

    Mn, grad_p_mag, a, v_dot_grad_vmag, f1, f2, shock_mask = filtered_sensor(
        points, mesh.cells, u, v, p, rho, cfg
    )

    mesh.point_data["normal_mach"]      = Mn
    mesh.point_data["grad_p_mag"]       = grad_p_mag
    mesh.point_data["sound_speed"]      = a
    mesh.point_data["v_dot_grad_vmag"]  = v_dot_grad_vmag
    mesh.point_data["filter1_pass"]     = f1
    mesh.point_data["filter2_pass"]     = f2
    mesh.point_data["shock_mask"]       = shock_mask

    out_path = make_output_path(cfg, in_path)
    meshio.write(out_path, mesh)

    mn_thr = float(cfg["mn_threshold"])
    f1_en  = bool(cfg.get("filter1_enabled", True))
    f2_en  = bool(cfg.get("filter2_enabled", True))
    print(f"--- {os.path.basename(in_path)} ---")
    print(f"  節点数             : {len(points)}")
    print(f"  M_n の範囲          : [{Mn.min():.3f}, {Mn.max():.3f}]")
    if f1_en:
        print(f"  Filter 1 通過      : {f1.sum()} 節点  (|grad_p| >= eps_f*p/l_n)")
    if f2_en:
        print(f"  Filter 2 通過      : {f2.sum()} 節点  (V*grad|V| < 0)")
    print(f"  衝撃波判定節点数    : {shock_mask.sum()}  (M_n>={mn_thr})")
    print(f"  出力                : {os.path.basename(out_path)}")
    return Mn, shock_mask


# ------------------------------------------------------------------
# 6. メイン
# ------------------------------------------------------------------
def main():
    cfg = load_config()
    targets = resolve_input_files(cfg)
    for in_path in targets:
        process_one(in_path, cfg)
    print("完了 (ParaView で normal_mach / shock_mask / filter1_pass / filter2_pass を可視化)")


if __name__ == "__main__":
    main()