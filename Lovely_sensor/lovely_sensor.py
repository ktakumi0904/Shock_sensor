"""
Lovely-Haimes 法による衝撃波検知 (Lovely & Haimes, AIAA 1999-3285) の練習用実装
対象: SU2 で計算した 2D 翼まわりの流れ場 (非構造格子 .vtu)

考え方:
    衝撃波を通過すると、衝撃波に垂直な方向の流速は超音速→亜音速に落ちる。
    衝撃波の法線方向は局所の圧力勾配 ∇p に平行とみなせるので、
    速度ベクトルを ∇p 方向へ射影した「法線マッハ数」M_n が 1 になる等値面が衝撃波。

法線マッハ数:
            u · ∇p
    M_n = ------------ ,   a = sqrt(gamma * p / rho)   (局所音速)
           a · |∇p|

出力スカラー場 shock_mach:
    フィルタ条件 (M_n >= threshold, u·∇p > 0) を満たす節点では M_n 値をそのまま保持し、
    それ以外の節点では 0 とする。

        shock_mach[i] = M_n[i]  (検知条件を満たす節点)
        shock_mach[i] = 0       (それ以外)

    ParaView で shock_mach = 1 の等値線を描くと、Lovely 法が検知した衝撃波断面が得られる。
    SU2 が出力する生の M_n 場から Mn=1 等値線を引くのと異なり、膨張波などの偽検知が
    フィルタリングされた衝撃波断面のみが表示される。

Ducros センサーが速度の発散/渦度だけで判定するのに対し、
こちらは圧力・密度・速度を使い、衝撃波の "向き" (法線) を圧力勾配で捉える点が異なる。

--------------------------------------------------------------------
使い方:
    同じディレクトリに config.yaml を置いて
        python lovely_sensor.py
    だけで実行できる。設定は config.yaml 側で変更する。
--------------------------------------------------------------------
"""

import os
import glob
import numpy as np
import meshio
import yaml


# スクリプト自身が置かれているディレクトリ (実行場所に依存しない)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "config.yaml"


# ----------------------------------------------------------------------
# 0. 設定ファイルの読み込み
# ----------------------------------------------------------------------
def load_config():
    """同じディレクトリの config_lovely.yaml を読み込む。無ければ既定値を返す。"""
    defaults = {
        "input_file": None,
        "multiple_files": "first",
        "gamma": 1.4,            # 比熱比
        "mn_threshold": 1.0,     # 法線マッハ数のしきい値
        "compression_filter": True,  # u·∇p>0 で膨張波を除外するか
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
    # 自分が生成した出力 (*_lovely.vtu, *_ducros.vtu) は対象から除外
    found = [f for f in found if not (f.endswith("_lovely.vtu") or f.endswith("_ducros.vtu"))]
    if not found:
        raise FileNotFoundError(f"{SCRIPT_DIR} に .vtu ファイルが見つかりません。")

    if cfg.get("multiple_files", "first") == "all":
        return found
    if len(found) > 1:
        print(f"[情報] .vtu が {len(found)} 個見つかりました。最初の 1 つを処理します: "
              f"{os.path.basename(found[0])}")
        print(f"       全部処理したい場合は config_lovely.yaml の multiple_files を all に。")
    return [found[0]]


def make_output_path(cfg, in_path):
    """出力パスを決める。設定が null なら入力名から自動生成。"""
    if cfg.get("output_file"):
        out = cfg["output_file"]
        return out if os.path.isabs(out) else os.path.join(SCRIPT_DIR, out)
    base, _ = os.path.splitext(in_path)
    return base + "_lovely.vtu"


# ----------------------------------------------------------------------
# 1. データ読み込み (速度・圧力・密度)
# ----------------------------------------------------------------------
def load_su2_vtu(path):
    """SU2 出力の .vtu を読み、座標・速度・圧力・密度を取り出す。"""
    from meshio.vtu._vtu import VtuReader

    reader = VtuReader(path)
    n_pts = len(reader.points)
    # 節点数と長さが一致しないフィールド (SU2 が出力する不完全データ等) を除外する
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
    points = mesh.points[:, :2]            # 2D なので x, y のみ
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
        ux = find("Velocity_x", "X-Velocity", "Momentum_x")
        vy = find("Velocity_y", "Y-Velocity", "Momentum_y")
        if ux is None or vy is None:
            raise KeyError(f"速度場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
        u, v = np.asarray(pdata[ux]), np.asarray(pdata[vy])

    # --- 圧力 ---
    pkey = find("Pressure", "pressure", "p")
    if pkey is None:
        raise KeyError(f"圧力場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
    p = np.asarray(pdata[pkey]).squeeze()

    # --- 密度 ---
    rkey = find("Density", "density", "rho", "Rho")
    if rkey is None:
        raise KeyError(f"密度場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
    rho = np.asarray(pdata[rkey]).squeeze()

    return mesh, points, u, v, p, rho


# ----------------------------------------------------------------------
# 2. 非構造格子上の節点勾配 (近傍最小二乗) ※ Ducros 版と同一実装
# ----------------------------------------------------------------------
def nodal_gradients_lsq(points, cells, field):
    """各節点で近傍との差分を最小二乗フィットして勾配 (df/dx, df/dy) を求める。"""
    N = len(points)
    neigh = [set() for _ in range(N)]
    for block in cells:
        conn = block.data
        for elem in conn:
            for a in elem:
                for b in elem:
                    if a != b:
                        neigh[a].add(b)

    grad = np.zeros((N, 2))
    for i in range(N):
        nb = list(neigh[i])
        if len(nb) < 2:
            continue
        dx = points[nb] - points[i]
        df = field[nb] - field[i]
        try:
            g, *_ = np.linalg.lstsq(dx, df.ravel(), rcond=None)
            grad[i] = np.asarray(g).ravel()[:2]
        except np.linalg.LinAlgError:
            pass
    return grad[:, 0], grad[:, 1]


# ----------------------------------------------------------------------
# 3. Lovely-Haimes センサー (法線マッハ数)
# ----------------------------------------------------------------------
def lovely_sensor(points, cells, u, v, p, rho, gamma=1.4, eps=1e-12):
    """
    法線マッハ数 M_n = (u·∇p) / (a |∇p|) を計算して返す。
    a = sqrt(gamma p / rho)
    """
    dpdx, dpdy = nodal_gradients_lsq(points, cells, p)   # 圧力勾配 ∇p

    grad_p_mag = np.sqrt(dpdx**2 + dpdy**2)              # |∇p|
    u_dot_gradp = u * dpdx + v * dpdy                    # u·∇p (圧力勾配方向の速度成分×|∇p|)

    a = np.sqrt(np.maximum(gamma * p / np.maximum(rho, eps), 0.0))  # 局所音速

    # M_n = (u·∇p)/(a|∇p|)
    Mn = u_dot_gradp / (a * grad_p_mag + eps)
    return Mn, u_dot_gradp, grad_p_mag, a


# ----------------------------------------------------------------------
# 4. 1 ファイルの処理
# ----------------------------------------------------------------------
def process_one(in_path, cfg):
    mesh, points, u, v, p, rho = load_su2_vtu(in_path)

    Mn, u_dot_gradp, grad_p_mag, a = lovely_sensor(
        points, mesh.cells, u, v, p, rho,
        gamma=float(cfg["gamma"]),
        eps=float(cfg["eps"]),
    )

    mn_threshold = float(cfg["mn_threshold"])
    # 法線マッハ数のしきい値判定
    mask = (Mn >= mn_threshold)
    # 膨張波除去フィルタ: u·∇p>0 (流れ方向に圧力上昇=圧縮) のみ残す
    if bool(cfg["compression_filter"]):
        mask = mask & (u_dot_gradp > 0)

    # 衝撃波断面スカラー場: 検知条件を満たす節点は M_n 値を保持、それ以外は 0
    # ParaView で shock_mach=1 の等値線を描くと Lovely 法の衝撃波断面が得られる
    shock_mach = np.where(mask, Mn, 0.0)

    mesh.point_data["shock_mach"]  = shock_mach
    mesh.point_data["grad_p_mag"]  = grad_p_mag
    mesh.point_data["sound_speed"] = a

    out_path = make_output_path(cfg, in_path)
    meshio.write(out_path, mesh)

    print(f"--- {os.path.basename(in_path)} ---")
    print(f"  節点数              : {len(points)}")
    print(f"  M_n の範囲           : [{Mn.min():.3f}, {Mn.max():.3f}]")
    print(f"  衝撃波検知節点数     : {mask.sum()}  (M_n>={mn_threshold}"
          f"{', u·∇p>0' if cfg['compression_filter'] else ''})")
    print(f"  出力                 : {os.path.basename(out_path)}")
    print(f"  → ParaView: shock_mach=1 の等値線で衝撃波断面を表示")
    return shock_mach


# ----------------------------------------------------------------------
# 5. メイン
# ----------------------------------------------------------------------
def main():
    cfg = load_config()
    targets = resolve_input_files(cfg)
    for in_path in targets:
        process_one(in_path, cfg)
    print("完了 (ParaView で shock_mach=1 の等値線を描いて衝撃波断面を可視化)")


if __name__ == "__main__":
    main()
