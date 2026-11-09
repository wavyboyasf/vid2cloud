"""Macierz zgrubna dla chmury w metrach (cloud_m_*.ply) z macierzy zgrubnej nagrania.

Pary punktów i macierz zgrubna M_c są klikane na `cloud.ply` (jednostki chmury). Chmura
w metrach to w przybliżeniu f_global · cloud.ply (dokładnie dla wariantu globalnego), więc

    M_m = M_c · diag(1/f_global, 1/f_global, 1/f_global, 1)

doprowadza ją do TLS tak samo jak M_c chmurę wyjściową; resztę (warianty A i B) doprecyzuje ICP
ze skalą. f_global z `scale.json` przebiegu (`vid2cloud scale`). Obok macierzy raport
<out>_report.json (macierz bazowa, sha256 obu plików wejściowych, f_global, skale macierzy).

    python eval/coarse_metric.py eval_cc/coarse/daleko_wolno.txt runs/…/daleko_wolno/scale.json \
        eval_cc/coarse/daleko_wolno_m_global.txt
Env vid2cloud-tools.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coarse_from_pairs import matrix_scale, read_matrix, write_matrix  # noqa: E402


def metric_coarse(M_c: np.ndarray, f_global: float) -> np.ndarray:
    """M_c · diag(1/f, 1/f, 1/f, 1): macierz dla chmury przeskalowanej przez f."""
    if not f_global > 0:
        raise ValueError(f"f_global = {f_global} — oczekiwano liczby > 0")
    return np.asarray(M_c, dtype=np.float64) @ np.diag([1.0 / f_global] * 3 + [1.0])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__)
        return 2
    base, scale_json, out = (Path(a) for a in argv)
    rec = json.loads(scale_json.read_text(encoding="utf-8"))
    if rec.get("status") != "ok":
        print(f"{scale_json}: status {rec.get('status')!r} — brak f_global")
        return 1
    f = float(rec["global"]["f"])
    M_c = read_matrix(base)
    M = metric_coarse(M_c, f)
    write_matrix(out, M)
    rep = {"base_matrix": str(base), "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
           "scale_json": str(scale_json), "scale_json_sha256": hashlib.sha256(scale_json.read_bytes()).hexdigest(),
           "f_global": f, "s_base": matrix_scale(M_c), "s_metric": matrix_scale(M), "matrix": M.tolist()}
    out.with_name(out.stem + "_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    print(f"{out}: f_global {f:.6f}, skala {rep['s_base']:.6f} -> {rep['s_metric']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
