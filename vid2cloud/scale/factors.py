"""Współczynnik skali osobno dla każdej klatki kluczowej — funkcje czyste (numpy).

Obserwacja: para (klatka kluczowa k, odcinek j) ze statusem ok w `scale_per_keyframe.csv`,
f_kj = factor_final = L_j / (s_k · d_canon_final). Pozycja klatki: pts_k = czas klatki źródła [s].
„Mediana klatki” to wszędzie mediana jej f_kj (kolumna f_obs w `scale_keyframes.csv`).

Warianty (wektor f_k, k = 0 … K−1 w kolejności powstania klatek):

* GLOBALNY — jedna liczba: `estimate_scale()` z pomiarami s_k · d_final odcinka po wszystkich
  klatkach, `select_factor("median")` (jak w docs/decisions.md, wpis z 2026-09-15);
* A, interpolacja — klatka z obserwacjami: f_k = mediana jej f_kj; pozostałe: log f liniowo
  w pts_k między najbliższymi klatkami z obserwacjami, przed pierwszą i za ostatnią — stały;
* B, MNK (wyrównanie pośredniczące) — niewiadome u_k = log f_k; równania u_k = log f_kj
  (waga 1/σ_obs²) i u_(k+1) − u_k = 0 (waga 1/(τ²·(pts_(k+1) − pts_k))); u = (AᵀPA)⁻¹AᵀPl,
  błąd średni u_k = √diag((AᵀPA)⁻¹) przy σ0 a priori = 1.

Położenia klatek (`keyframe_positions`) jak ciąg poligonowy: p_0 = f_0·t_0,
p_k = p_(k−1) + ½(f_(k−1) + f_k)·(t_k − t_(k−1)).

σ_obs i τ (`sigma_obs_estimate`, `tau_estimate`) liczy raz `eval/scale_params.py` i zapisuje
do `vid2cloud/scale/params.json` (`load_params`) — to są wartości domyślne każdego przebiegu.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from vid2cloud.scale.estimate import ScaleEstimate, Targets, estimate_scale, select_factor

PARAMS_PATH = Path(__file__).with_name("params.json")


class NoPairs(ValueError):
    """Żadna para (klatka kluczowa, odcinek) nie ma statusu ok — nie ma z czego liczyć skali."""


@dataclass(frozen=True)
class Observation:
    kf: int
    segment: str
    f: float  # factor_final [m / jedn. chmury]


# --------------------------------------------------------------------------- #
# Obserwacje
# --------------------------------------------------------------------------- #


def observations(rows: Iterable[dict]) -> list[Observation]:
    """Pary (k, odcinek) ze statusem ok i z factor_final — bez odrzucania czegokolwiek."""
    out = [Observation(int(r["kf"]), str(r["segment"]), float(r["factor_final"]))
           for r in rows if r.get("status") == "ok" and r.get("factor_final") is not None]
    for o in out:
        if not (math.isfinite(o.f) and o.f > 0):
            raise ValueError(f"kf {o.kf} {o.segment}: factor_final = {o.f} — oczekiwano liczby > 0")
    return out


def frame_medians(obs: Iterable[Observation]) -> dict[int, float]:
    """kf -> mediana f_kj tej klatki (tylko klatki z obserwacjami)."""
    groups: dict[int, list[float]] = {}
    for o in obs:
        groups.setdefault(o.kf, []).append(o.f)
    return {k: float(np.median(v)) for k, v in sorted(groups.items())}


def frame_counts(obs: Iterable[Observation]) -> dict[int, int]:
    out: dict[int, int] = {}
    for o in obs:
        out[o.kf] = out.get(o.kf, 0) + 1
    return out


def _check_pts(pts: Sequence[float]) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 1 or not len(pts):
        raise ValueError(f"pts ma być niepustym wektorem, ma kształt {pts.shape}")
    if len(pts) > 1 and not np.all(np.diff(pts) > 0):
        raise ValueError("pts_k nie rośnie ściśle z k — kolejność powstania klatek i czas się rozjeżdżają")
    return pts


# --------------------------------------------------------------------------- #
# Wariant globalny
# --------------------------------------------------------------------------- #


def global_factor(rows: Iterable[dict], targets: Targets) -> tuple[float, str, ScaleEstimate]:
    """Jedno f: mediana współczynników odcinków, każdy z mediany s_k·d_final po klatkach."""
    meas: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") != "ok" or r.get("d_canon_final") is None:
            continue
        meas.setdefault(r["segment"], []).append(float(r["s_final"]) * float(r["d_canon_final"]))
    if not meas:
        raise NoPairs("brak par (klatka kluczowa, odcinek) ze statusem ok")
    order = [name for name in targets.segments if name in meas]  # kolejność z targets.json
    est = estimate_scale([{"segment": n, "measurements": meas[n]} for n in order], targets)
    f, how = select_factor(est, "median")
    return f, how, est


# --------------------------------------------------------------------------- #
# Metoda A — interpolacja log f w czasie
# --------------------------------------------------------------------------- #


def interp_factors(pts: Sequence[float], obs: Sequence[Observation]) -> np.ndarray:
    """f_k wszystkich klatek: mediana klatki albo log f interpolowany liniowo w pts (stały na końcach)."""
    pts = _check_pts(pts)
    med = frame_medians(obs)
    if not med:
        raise NoPairs("brak obserwacji")
    for k in med:
        if not 0 <= k < len(pts):
            raise ValueError(f"obserwacja w kf {k}, a klatek jest {len(pts)}")
    ks = sorted(med, key=lambda k: pts[k])
    u = np.interp(pts, pts[ks], np.log([med[k] for k in ks]))  # np.interp: stały poza zakresem
    f = np.exp(u)
    for k, v in med.items():
        f[k] = v  # w węźle dokładnie mediana, bez obiegu przez log/exp
    return f


# --------------------------------------------------------------------------- #
# Metoda B — wyrównanie pośredniczące
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LsqResult:
    u: np.ndarray  # (K,) log f_k
    sigma_u: np.ndarray  # (K,) błąd średni u_k, σ0 a priori = 1
    sigma0_post: Optional[float]  # √(vᵀPv / r); None przy r = 0
    dof: int  # r = liczba równań − K
    n_obs: int
    n_eq: int
    vtpv: float

    @property
    def f(self) -> np.ndarray:
        return np.exp(self.u)


def lsq_design(pts: Sequence[float], obs: Sequence[Observation], sigma_obs: float, tau: float):
    """Macierz A, wagi p (przekątna P) i wyrazy wolne l układu metody B."""
    pts = _check_pts(pts)
    if not (sigma_obs > 0 and tau > 0):
        raise ValueError(f"σ_obs i τ muszą być > 0, są {sigma_obs}, {tau}")
    K = len(pts)
    rows, p, l = [], [], []
    for o in obs:
        if not 0 <= o.kf < K:
            raise ValueError(f"obserwacja w kf {o.kf}, a klatek jest {K}")
        a = np.zeros(K)
        a[o.kf] = 1.0
        rows.append(a)
        p.append(1.0 / sigma_obs ** 2)
        l.append(math.log(o.f))
    for k in range(K - 1):
        a = np.zeros(K)
        a[k], a[k + 1] = -1.0, 1.0
        rows.append(a)
        p.append(1.0 / (tau ** 2 * (pts[k + 1] - pts[k])))
        l.append(0.0)
    return np.array(rows).reshape(-1, K), np.array(p), np.array(l)


def lsq_factors(pts: Sequence[float], obs: Sequence[Observation], sigma_obs: float, tau: float) -> LsqResult:
    """u = (AᵀPA)⁻¹AᵀPl; błędy średnie z diag((AᵀPA)⁻¹); σ0 a posteriori z vᵀPv / r."""
    if not obs:
        raise NoPairs("brak obserwacji")
    A, p, l = lsq_design(pts, obs, sigma_obs, tau)
    N = A.T @ (p[:, None] * A)
    Q = np.linalg.inv(N)
    u = Q @ (A.T @ (p * l))
    v = A @ u - l
    vtpv = float(v @ (p * v))
    dof = A.shape[0] - A.shape[1]
    return LsqResult(u=u, sigma_u=np.sqrt(np.diag(Q)), sigma0_post=math.sqrt(vtpv / dof) if dof > 0 else None,
                     dof=int(dof), n_obs=len(obs), n_eq=int(A.shape[0]), vtpv=vtpv)


# --------------------------------------------------------------------------- #
# Położenia klatek — ciąg poligonowy
# --------------------------------------------------------------------------- #


def keyframe_positions(t: np.ndarray, f: Sequence[float]) -> np.ndarray:
    """p_0 = f_0·t_0; p_k = p_(k−1) + ½(f_(k−1) + f_k)·(t_k − t_(k−1))."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    if len(f) != len(t):
        raise ValueError(f"{len(t)} położeń, {len(f)} współczynników")
    p = np.empty_like(t)
    if len(t):
        p[0] = f[0] * t[0]
    for k in range(1, len(t)):
        p[k] = p[k - 1] + 0.5 * (f[k - 1] + f[k]) * (t[k] - t[k - 1])
    return p


# --------------------------------------------------------------------------- #
# σ_obs i τ — z danych, raz (eval/scale_params.py)
# --------------------------------------------------------------------------- #


def sigma_obs_estimate(groups: Iterable[Sequence[float]]) -> dict:
    """σ_obs² = Σ (log f_kj − log mediana_k)² / (n − m) po klatkach z ≥ 2 odcinkami.

    `groups` — f_kj kolejnych klatek (z dowolnych nagrań). n = liczba reszt, m = liczba klatek;
    dzielnik n − m, bo każda klatka zużywa jeden stopień swobody na swoją medianę.
    """
    res, m = [], 0
    for g in groups:
        g = [float(x) for x in g]
        if len(g) < 2:
            continue
        m += 1
        med = math.log(float(np.median(g)))
        res.extend(math.log(x) - med for x in g)
    n = len(res)
    dof = n - m
    value = math.sqrt(sum(r * r for r in res) / dof) if dof > 0 else None
    return {"sigma_obs": value, "n_residuals": n, "n_frames": m, "dof": dof,
            "sum_sq": float(sum(r * r for r in res)), "residuals": res}


def tau_estimate(series: Iterable[Sequence[tuple[float, float]]]) -> dict:
    """τ² = Σ (Δu)² / Σ Δt po kolejnych klatkach z obserwacjami w obrębie nagrania.

    `series` — dla każdego nagrania lista (pts_k, mediana f_kj) klatek z obserwacjami;
    Δu = różnica log median kolejnych (po pts) klatek, Δt = różnica ich pts.
    """
    du2, dts = [], []
    for s in series:
        s = sorted((float(t), float(f)) for t, f in s)
        for (t0, f0), (t1, f1) in zip(s, s[1:]):
            if not t1 > t0:
                raise ValueError(f"dwie klatki z tym samym pts {t0}")
            du2.append((math.log(f1) - math.log(f0)) ** 2)
            dts.append(t1 - t0)
    tau2 = sum(du2) / sum(dts) if dts else None
    return {"tau2": tau2, "tau": math.sqrt(tau2) if tau2 is not None else None, "n_increments": len(dts),
            "sum_du2": float(sum(du2)), "sum_dt": float(sum(dts))}


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Zamrożone σ_obs i τ (params.json). Brak pliku albo wartości to błąd, nie wartość domyślna."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("sigma_obs", "tau"):
        v = data.get(key)
        if not isinstance(v, (int, float)) or not v > 0:
            raise ValueError(f"{path}: {key} = {v!r} — oczekiwano liczby > 0")
    return data
