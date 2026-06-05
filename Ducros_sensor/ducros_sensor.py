"""
最も単純な Ducros センサー (原典 Ducros et al. 1999) の練習用実装
対象: SU2 で計算した 2D 翼まわりの流れ場 (非構造格子 .vtu)

Ducros センサー:
    phi = (-div u) / ( |div u| + |omega| + eps )       ... (A) 圧縮優位スイッチ版
ここでは衝撃波(圧縮 div u < 0)を 1 に近づけたいので、よく使われる形
    phi = max(-div u, 0) / ( |div u| + |omega| + eps )
を採用する。phi ~ 1 : 衝撃波(強い圧縮), phi ~ 0 : 渦・滑らかな領域。

div u   = du/dx + dv/dy
omega_z = dv/dx - du/dy   (2D の渦度)

勾配は SU2 が出力していればそれを使い、無ければ
非構造格子上で「最小二乗(Green-Gauss 近似)」により節点勾配を再構成する。

--------------------------------------------------------------------
使い方:
    同じディレクトリに config.yaml を置いて
        python ducros_sensor.py
    だけで実行できる。設定は config.yaml 側で変更する。
--------------------------------------------------------------------
"""

import os
import glob
import numpy as np
import meshio
import yaml


# SU2 の VTU 出力で一部フィールドの配列長が節点数と合わない場合がある
# (Eddy_Viscosity 等が境界節点を除いた内部節点数で書き出されるケース)。
# meshio の Mesh.__init__ がバリデーションで例外を投げる前に除外する。
_orig_mesh_init = meshio.Mesh.__init__

def _lenient_mesh_init(self, points, cells, point_data=None, **kwargs):
    if point_data is not None:
        n = len(points)
        filtered = {k: v for k, v in point_data.items()
                    if np.asarray(v).shape[0] == n}
        skipped = sorted(set(point_data) - set(filtered))
        if skipped:
            print(f"[警告] 節点数不一致のフィールドを除外しました: {skipped}")
        point_data = filtered
    _orig_mesh_init(self, points, cells, point_data=point_data, **kwargs)

meshio.Mesh.__init__ = _lenient_mesh_init


# スクリプト自身が置かれているディレクトリ (実行場所に依存しない)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "config.yaml"


# ----------------------------------------------------------------------
# 0. 設定ファイルの読み込み
# ----------------------------------------------------------------------
def load_config():
    """同じディレクトリの config.yaml を読み込む。無ければ既定値を返す。"""
    defaults = {
        "input_file": None,
        "multiple_files": "first",
        "eps_rel": 1e-3,
        "magnitude_rel": 0.1,
        "ref_percentile": 99.0,
        "threshold": 0.95,
        "mach_threshold": 0.8,
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
    # 設定でファイル名が明示されていればそれを使う
    if cfg.get("input_file"):
        path = cfg["input_file"]
        if not os.path.isabs(path):
            path = os.path.join(SCRIPT_DIR, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定された input_file が見つかりません: {path}")
        return [path]

    # 未指定なら同じディレクトリの .vtu を自動検出
    found = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.vtu")))
    # 自分が生成した出力 (*_ducros.vtu) は対象から除外
    found = [f for f in found if not f.endswith("_ducros.vtu")]
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
    """出力パスを決める。設定が null なら入力名から自動生成。"""
    if cfg.get("output_file"):
        out = cfg["output_file"]
        return out if os.path.isabs(out) else os.path.join(SCRIPT_DIR, out)
    base, _ = os.path.splitext(in_path)
    return base + "_ducros.vtu"


# ----------------------------------------------------------------------
# 1. データ読み込み
# ----------------------------------------------------------------------
def load_su2_vtu(path):
    """SU2 出力の .vtu を読み、座標・速度場・マッハ数を取り出す。"""
    mesh = meshio.read(path)
    points = mesh.points[:, :2]            # 2D なので x, y のみ
    pdata = mesh.point_data

    # SU2 の速度フィールド名のゆらぎに対応
    def find(*names):
        for n in names:
            if n in pdata:
                return n
        return None

    # ベクトルでまとまっている場合 (Velocity = [u,v,w])
    if find("Velocity", "velocity"):
        key = find("Velocity", "velocity")
        vel = np.asarray(pdata[key])
        u, v = vel[:, 0], vel[:, 1]
    else:
        ukey = find("Velocity_x", "X-Velocity", "Momentum_x")
        vkey = find("Velocity_y", "Y-Velocity", "Momentum_y")
        if ukey is None or vkey is None:
            raise KeyError(f"速度場が見つかりません。利用可能なフィールド: {list(pdata.keys())}")
        u, v = np.asarray(pdata[ukey]), np.asarray(pdata[vkey])

    # マッハ数 (あれば): よどみ点誤検知の排除に使う
    mach_key = find("Mach", "mach", "Mach_Number", "MACH")
    mach = np.asarray(pdata[mach_key]).ravel() if mach_key else None
    if mach is None:
        print("[情報] Mach フィールドが見つかりません。マッハ数ゲートはスキップします。")

    return mesh, points, u, v, mach


# ----------------------------------------------------------------------
# 2. 非構造格子上の節点勾配 (Green-Gauss / 近傍最小二乗)
# ----------------------------------------------------------------------
def nodal_gradients_lsq(points, cells, field):
    """
    各節点で、接続する近傍節点との差分を最小二乗フィットして勾配 (df/dx, df/dy) を求める。
    points : (N,2)
    cells  : 三角形/四角形の節点接続リスト (リスト of (M,k))
    field  : (N,) スカラー
    """
    N = len(points)
    # --- 節点ごとの近傍リストを作る ---
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
        dx = points[nb] - points[i]          # (k,2)
        df = field[nb] - field[i]            # (k,)
        # 最小二乗: dx @ g ≈ df  ->  g = (A^T A)^-1 A^T df
        A = dx
        try:
            g, *_ = np.linalg.lstsq(A, df, rcond=None)
            grad[i] = g
        except np.linalg.LinAlgError:
            pass
    return grad[:, 0], grad[:, 1]


# ----------------------------------------------------------------------
# 3. Ducros センサー
# ----------------------------------------------------------------------
def ducros_sensor(points, cells, u, v,
                  eps_rel=1e-3,
                  magnitude_rel=0.1,
                  ref_percentile=99.0):
    """
    原典 Ducros (1999) の 2 乗形式セレクタ:
        phi = (div u)^2 / ( (div u)^2 + omega^2 + eps )
    phi ~ 1 : 圧縮が渦より優勢 (衝撃波の候補), phi ~ 0 : 渦優位

    重要: phi は「圧縮 vs 渦」のセレクタにすぎず、滑らかな一様流
    (div も omega もほぼ 0) では数値ノイズで誤って 1 に張り付く。
    これを防ぐため
      (1) eps を「場の圧縮スケールに対する床」として与える
      (2) 振幅ゲート: 実際に強く圧縮している節点だけを衝撃波とみなす
    の 2 段構えにする。

    戻り値: phi(セレクタ), div, omega, comp(圧縮強度), eps, mag_floor
    """
    dudx, dudy = nodal_gradients_lsq(points, cells, u)
    dvdx, dvdy = nodal_gradients_lsq(points, cells, v)

    div = dudx + dvdy            # ∇·u
    omega = dvdx - dudy          # ω_z

    div2 = div * div
    omega2 = omega * omega

    # 圧縮強度 (膨張側は 0)
    comp = np.maximum(-div, 0.0)
    # ロバストな基準スケール (外れ値に強い高パーセンタイル)
    ref_comp = np.percentile(comp, ref_percentile)
    if ref_comp <= 0:
        ref_comp = comp.max() if comp.max() > 0 else 1.0

    # eps: 基準圧縮の (eps_rel 倍)^2 を床にする。
    # → 滑らかな領域 (div,omega≈0) では分母が eps 支配となり phi≈0。
    eps = (eps_rel * ref_comp) ** 2 + 1e-30

    # 2 乗形式 Ducros セレクタ
    phi = div2 / (div2 + omega2 + eps)

    # 振幅ゲートの床 (基準圧縮の magnitude_rel 倍)
    mag_floor = magnitude_rel * ref_comp

    return phi, div, omega, comp, eps, mag_floor


# ----------------------------------------------------------------------
# 4. 1 ファイルの処理
# ----------------------------------------------------------------------
def process_one(in_path, cfg):
    mesh, points, u, v, mach = load_su2_vtu(in_path)
    phi, div, omega, comp, eps, mag_floor = ducros_sensor(
        points, mesh.cells, u, v,
        eps_rel=float(cfg["eps_rel"]),
        magnitude_rel=float(cfg["magnitude_rel"]),
        ref_percentile=float(cfg["ref_percentile"]),
    )

    threshold = float(cfg["threshold"])
    mach_threshold = float(cfg.get("mach_threshold", 0.8))

    # ---- 3 段階ゲート ----
    # (1) Ducros セレクタ: 圧縮が渦より卓越
    is_selector = phi >= threshold
    # (2) 振幅ゲート: 圧縮強度が十分大きい (よどみ点は通過する可能性あり)
    is_strong = comp >= mag_floor
    # (3) マッハ数ゲート: 衝撃波は超音速領域にしか存在できない
    #     よどみ点は M≈0 なので確実に除外できる (SU2 の二値検知用途では必須)
    if mach is not None:
        is_supersonic = mach >= mach_threshold
    else:
        is_supersonic = np.ones(len(phi), dtype=bool)   # フィールド無し: ゲートなし

    shock_mask = (is_selector & is_strong & is_supersonic).astype(np.int32)

    # 結果を point_data に追加して書き出す
    mesh.point_data["ducros_phi"] = phi
    mesh.point_data["divergence"] = div
    mesh.point_data["vorticity_z"] = omega
    mesh.point_data["compression"] = comp     # 圧縮強度 max(-div,0)
    mesh.point_data["shock_mask"] = shock_mask
    if mach is not None:
        mesh.point_data["mach_gate"] = is_supersonic.astype(np.int32)

    out_path = make_output_path(cfg, in_path)
    meshio.write(out_path, mesh)

    n_sel = int(is_selector.sum())
    n_strong = int((is_selector & is_strong).sum())
    print(f"--- {os.path.basename(in_path)} ---")
    print(f"  節点数                  : {len(points)}")
    print(f"  phi の範囲               : [{phi.min():.3f}, {phi.max():.3f}]")
    print(f"  eps(床)                 : {eps:.3e}")
    print(f"  振幅ゲート床 mag_floor    : {mag_floor:.3e}")
    print(f"  (1) セレクタ通過          : {n_sel}  (phi>={threshold})")
    print(f"  (2) 振幅ゲート後          : {n_strong}")
    if mach is not None:
        print(f"  (3) マッハ数ゲート後      : {int(shock_mask.sum())}  (M>={mach_threshold})")
    else:
        print(f"  衝撃波判定              : {int(shock_mask.sum())}")
    print(f"  出力                     : {os.path.basename(out_path)}")
    return phi, shock_mask


# ----------------------------------------------------------------------
# 5. メイン
# ----------------------------------------------------------------------
def main():
    cfg = load_config()
    targets = resolve_input_files(cfg)
    for in_path in targets:
        process_one(in_path, cfg)
    print("完了 (ParaView で ducros_phi / shock_mask を可視化)")


if __name__ == "__main__":
    main()
