"""Ocena kształtu chmury: funkcje czyste i test integracyjny z CloudCompare na syntetyce.

Syntetyka: TLS = powierzchnia pudełka 4 × 3 × 2,5 m (siatka 5 mm na ścianach), rekonstrukcja =
to samo pudełko próbkowane niezależnie, po znanym podobieństwie (skala 1,5 m / jedn.), z szumem
gaussowskim σ = 1 cm na każdej osi i 2 % punktów odstających (jednostajnie w objętości).
Macierz zgrubna z 8 naroży wskazanych z błędem 2 cm (Umeyama). Kryteria ustalone przed
uruchomieniem: skala odtworzona do 0,5 %, RMS C2C w granicach ±15 % σ (składowa szumu
prostopadła do ściany ma RMS σ; siatka 5 mm dokłada ok. √(h²/6) = 2 mm w kwadraturze).
Test integracyjny wymaga CloudCompare (flatpak) — bez niego jest pomijany.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ec = _load("eval_cloud")
cfp = _load("coarse_from_pairs")
from plyio import read_ply, write_ply, xyz, xyz_vertices  # noqa: E402

CFG = json.loads((ROOT / "eval" / "eval_config.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Funkcje czyste
# --------------------------------------------------------------------------- #


def test_config_values_fixed_before_results():
    """Wartości z promptu sesji 2026-10-30 — zmiana konfiguracji ma wywrócić ten test."""
    assert CFG["tls_subsample_m"] == 0.01
    assert (CFG["sor_knn"], CFG["sor_sigma"]) == (6, 1.0)
    assert CFG["icp"] == {"adjust_scale": True, "min_error_diff": 1e-6, "random_sampling_limit": 50000,
                          "final_overlap_pct": 90}
    assert CFG["thresholds_m"] == [0.05, 0.10, 0.20]
    assert CFG["crop_margin_m"] == 0.20 and CFG["crop_percentiles"] == [1, 99]


def test_crop_box_percentiles_and_margin():
    x = np.arange(101, dtype=float)  # percentyl 1 = 1, 99 = 99
    P = np.column_stack([x, 2 * x, -x])
    lo, hi = ec.crop_box(P, (1, 99), 0.2)
    np.testing.assert_allclose(lo, [0.8, 1.8, -99.2])
    np.testing.assert_allclose(hi, [99.2, 198.2, -0.8])
    assert ec.in_box(np.array([[0.8, 1.8, -50], [0.79, 100, -50]]), lo, hi).tolist() == [True, False]


def test_distance_stats_hand_values():
    """d = [0.03, 0.04, 0.12, 0.30]: średnia 0.1225, mediana 0.08, RMS √(0.1069/4)."""
    s = ec.distance_stats([0.03, 0.04, 0.12, 0.30, np.nan], [0.05, 0.10, 0.20])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.1225)
    assert s["median"] == pytest.approx(0.08)
    assert s["rms"] == pytest.approx(np.sqrt((0.0009 + 0.0016 + 0.0144 + 0.09) / 4))
    assert s["fraction_le"] == {"0.05": 0.5, "0.10": 0.5, "0.20": 0.75}


def test_parse_icp_log_cc_2_13():
    log = ("[16:05:47] [ICP] Registration done in 1.195 sec\n[16:05:47] Entity 'rec' has been registered\n"
           "[16:05:47] RMS: 0.0226167\n[16:05:47] Number of points used for final step: 45525\n")
    assert ec.parse_icp_log(log) == {"rms": 0.0226167, "n_final": 45525}
    with pytest.raises(ValueError):
        ec.parse_icp_log("[x] nic\n")


def test_similarity_parts_and_total_scale():
    R = cfp.umeyama(np.eye(3) * [1, 2, 3], (np.eye(3) * [1, 2, 3])[:, [1, 0, 2]] * [1, 1, -1])[1]
    M1 = cfp.to_matrix(1.5, R, [1, 2, 3])
    M2 = cfp.to_matrix(0.98, np.eye(3), [0.1, 0, 0])
    s, R2, t = ec.similarity_parts(M2 @ M1)
    assert s == pytest.approx(1.5 * 0.98)
    np.testing.assert_allclose(R2 @ R2.T, np.eye(3), atol=1e-12)


def test_s_mod_median_of_factor_final(tmp_path):
    p = tmp_path / "scale_per_keyframe.csv"
    p.write_text("kf,factor_final,status\n0,1.5,ok\n1,,x\n2,1.7,ok\n3,1.6,ok\n", encoding="utf-8")
    assert ec.s_mod_from_csv(p) == (pytest.approx(1.6), 3)


def test_cc_arguments_order_and_switches():
    a = ec.cc_main_args(CFG, Path("r.ply"), Path("t.ply"), Path("s.ply"), Path("c.ply"))
    s = " ".join(a)
    assert s.index("-O r.ply") < s.index("-SOR 6 1") < s.index("-O t.ply") < s.index("-ICP") < s.index("-C2C_DIST")
    assert "-MIN_ERROR_DIFF 1e-06 -RANDOM_SAMPLING_LIMIT 50000 -OVERLAP 90 -ADJUST_SCALE" in s
    assert "-MODEL" not in s  # C2C bez modelu lokalnego
    r = " ".join(ec.cc_rigid_args(CFG, Path("r.ply"), Path("t.ply"), Path("c.ply")))
    assert "-ADJUST_SCALE" not in r and "-OVERLAP 90" in r


def test_update_summary_replaces_same_cloud(tmp_path):
    p = tmp_path / "s.csv"
    ec.update_summary(p, {"name": "a", "cloud": "x.ply", "c2c_rms": 1.0})
    ec.update_summary(p, {"name": "b", "cloud": "y.ply", "c2c_rms": 2.0})
    ec.update_summary(p, {"name": "a", "cloud": "x.ply", "c2c_rms": 3.0})
    rows = p.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and rows[-1].startswith("a,x.ply") and ",3.0," not in rows[1]


# --------------------------------------------------------------------------- #
# Integracja z CloudCompare
# --------------------------------------------------------------------------- #

L = np.array([4.0, 3.0, 2.5])
S_TRUE = 1.5
SIGMA = 0.01


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


R_TRUE = _rot([0.2, 1.0, -0.3], 40.0)
T_TRUE = np.array([1.8, 1.4, 1.2])  # kamera kf 0 w pudełku


def _box_grid(h):
    pts = []
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        gu = np.arange(h / 2, L[u], h)
        gv = np.arange(h / 2, L[v], h)
        U, V = np.meshgrid(gu, gv, indexing="ij")
        for w in (0.0, L[ax]):
            P = np.zeros((U.size, 3))
            P[:, u], P[:, v], P[:, ax] = U.ravel(), V.ravel(), w
            pts.append(P)
    return np.vstack(pts)


def _box_random(n, rng):
    areas = np.array([L[1] * L[2], L[0] * L[2], L[0] * L[1]]).repeat(2)
    face = rng.choice(6, size=n, p=areas / areas.sum())
    P = rng.uniform(0, 1, size=(n, 3)) * L
    ax = face // 2
    P[np.arange(n), ax] = np.where(face % 2 == 0, 0.0, L[ax])
    return P


def _cc_available():
    if shutil.which("flatpak") is None:
        return False
    r = subprocess.run(["flatpak", "info", "org.cloudcompare.CloudCompare"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _cc_available(), reason="brak CloudCompare (flatpak)")
def test_synthetic_box_scale_and_rms(tmp_path):
    rng = np.random.default_rng(7)
    tls = _box_grid(0.005)
    write_ply(tmp_path / "tls.ply", xyz_vertices(tls))

    # rekonstrukcja: y = s R x + t  =>  x = R^T (y - t) / s
    true = _box_random(200_000, rng) + rng.normal(0, SIGMA, size=(200_000, 3))
    out_n = int(0.02 * len(true))
    outliers = rng.uniform(-0.1 * L, 1.1 * L, size=(out_n, 3))
    world = np.vstack([true, outliers])
    rec = (world - T_TRUE) @ R_TRUE / S_TRUE
    run_dir = tmp_path / "run"
    write_ply(run_dir / "cloud.ply", xyz_vertices(rec))
    # factor_final wokół prawdziwej skali, mediana = S_TRUE
    (run_dir / "scale_per_keyframe.csv").write_text(
        "kf,factor_final\n0,%r\n1,%r\n2,%r\n" % (S_TRUE * 0.99, S_TRUE, S_TRUE * 1.01), encoding="utf-8")

    # macierz zgrubna z 8 naroży, błąd wskazania 2 cm na TLS
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * L
    rec_c = (corners - T_TRUE) @ R_TRUE / S_TRUE
    np.savetxt(tmp_path / "p_rec.txt", rec_c, delimiter=",")
    np.savetxt(tmp_path / "p_tls.txt", corners + rng.normal(0, 0.02, size=corners.shape), delimiter=",")
    coarse = tmp_path / "coarse.txt"
    assert cfp.main(["syn", "--rec", str(tmp_path / "p_rec.txt"), "--tls", str(tmp_path / "p_tls.txt"),
                     "--out", str(coarse)]) == 0

    out = tmp_path / "out"
    assert ec.main([str(run_dir / "cloud.ply"), "--name", "syn", "--out", str(out), "--eval-cc", str(tmp_path / "ecc"),
                    "--tls", str(tmp_path / "tls.ply"), "--coarse", str(coarse),
                    "--crop-source", str(run_dir / "cloud.ply")]) == 0
    r = json.loads((out / "c2c_stats.json").read_text(encoding="utf-8"))
    print(f"\ns_coarse {r['s_coarse']:.5f}, s_icp {r['s_icp']:.5f} (prawda {S_TRUE}), "
          f"RMS C2C {r['c2c']['rms']:.5f}, mediana {r['c2c']['median']:.5f}, RMS ICP (CC) {r['icp_rms_cc']:.5f}, "
          f"po SOR {r['n_after_sor']}/{r['n_before_sor']}, s_mod/s_icp-1 {r['s_mod_vs_icp_pct']:+.3f} %, "
          f"RMS sztywno {r['rigid_c2c']['rms']:.5f}")
    assert abs(r["s_icp"] / S_TRUE - 1) < 0.005
    assert r["c2c"]["rms"] == pytest.approx(SIGMA, rel=0.15)
    assert r["n_after_sor"] < r["n_before_sor"]
    assert abs(r["s_mod_vs_icp_pct"]) < 0.5
    assert r["rigid_det"] == pytest.approx(1.0, abs=1e-6)
    assert r["rigid_c2c"]["rms"] == pytest.approx(SIGMA, rel=0.15)
    assert (out / "c2c.ply").exists() and (out / "cc_commands.txt").read_text().count("cc.sh") == 2
    assert (tmp_path / "ecc" / "crop" / "syn.json").exists()
    summary = (tmp_path / "ecc" / "c2c_summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(summary) == 2
    c2c = read_ply(out / "c2c.ply")
    assert len(c2c) == r["n_after_sor"] and ec.c2c_field(c2c)
