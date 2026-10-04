"""Adapter do upstreamowego MASt3R-SLAM z third_party/mast3r-slam/ (commit e6f4e3d).

To JEDYNE miejsce w projekcie, które zna API MASt3R-SLAM. Upstreamu nie modyfikujemy.

Co robi adapter
---------------
`Mast3rSlamEngine` odtwarza pętlę główną `main.py:233-310` krok po kroku, ale klatki
dostaje z zewnątrz (`push_frame`) zamiast z `load_dataset()`. Wszystkie elementy silnika
są importowane z upstreamu bez zmian: `create_frame`, `FrameTracker`, `SharedKeyframes`,
`SharedStates`, `mast3r_inference_mono`, `load_mast3r`, a proces backendu uruchamia
oryginalne `main.run_backend` (import z `main.py` działa: proces `spawn` dostaje sys.path
rodzica i importuje moduł `main` po nazwie — sprawdzone 2026-10-03).

Różnice względem `main.py` — wyłącznie wokół pętli, nie w niej:
* Źródło klatek: RGB uint8 z `input/sources.py` zamieniane na float32 0..1 tak jak
  `dataloader.py:50` (`img.astype(float32) / 255.0`), dalej `create_frame` bez zmian.
* `config["single_thread"]` nadpisywane parametrem (base.yaml ma False). True = po każdej
  nowej klatce kluczowej czekamy na koniec optymalizacji backendu (`main.py:301`), więc
  zrzuty są deterministyczne względem zdarzeń. False = zachowanie upstreamu.
* Proces backendu startuje z cwd = katalog upstreamu, bo `load_retriever` czyta checkpoint
  ścieżką względną (`mast3r_utils.py:26`). Model główny ładujemy ścieżką bezwzględną.
* Haki (nie zmieniają przebiegu, tylko czytają stan pod `keyframes.lock`):
  - kf_created  — przy każdej nowej klatce kluczowej `keyframes/kf_{k:03d}_created.npz`
                  i wiersze `poses_history.csv`. Nowe wykrywamy po `len(keyframes)`, bo
                  klatka kluczowa może powstać też w backendzie przy relokalizacji (`main.py:42`).
  - backend_done — po zakończonej optymalizacji wiersze `poses_history.csv` dla wszystkich
                  klatek kluczowych. Przy single_thread: po każdym czekaniu z `main.py:301`.
                  Bez single_thread: sprawdzane po każdej klatce (licznik zadań), więc kilka
                  optymalizacji między dwiema klatkami zlewa się w jedno zdarzenie.
* `finalize()` najpierw kończy i dołącza backend, potem zapisuje. Upstream zapisuje przed
  `backend.join()` (`main.py:312-333`), co bez single_thread ściga się z ostatnią
  optymalizacją; przy single_thread kolejka jest wtedy pusta i kolejność nie ma znaczenia.

Stan klatki kluczowej z relokalizacji: backend dokłada `states.get_frame()`, a ta nie
przenosi N (`frame.py:169-183`), więc w `SharedKeyframes` taka klatka ma N = 0 do pierwszej
aktualizacji przez tracker. Zrzut zapisuje to, co jest w silniku, bez poprawiania. Jej pozę
widzimy dopiero po optymalizacji w `relocalization()` — stąd zdarzenie "reloc", nie "kf_created".
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "mast3r-slam"
DEFAULT_CONFIG = UPSTREAM_DIR / "config" / "base.yaml"
CHECKPOINT = UPSTREAM_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
#: Próg ufności chmury: domyślne `WindowMsg.C_conf_threshold` (visualization.py:38),
#: czyli to, z czym `main.py:319` zapisuje chmurę przy --no-viz.
CLOUD_C_CONF_THRESHOLD = 1.5
#: Dłuższy bok wejścia modelu: `MonocularDataset.img_size` (dataloader.py:25).
IMG_SIZE = 512

# `main.py` nie jest częścią pakietu mast3r_slam. Dopisujemy jego katalog NA KOŃCU sys.path
# (żeby niczego nie przesłonić); proces spawn dziedziczy sys.path i zaimportuje `main` sam.
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.append(str(UPSTREAM_DIR))

import cv2  # noqa: E402
import lietorch  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

import main as upstream_main  # noqa: E402
from mast3r_slam import evaluate as upstream_eval  # noqa: E402
from mast3r_slam.config import config, load_config  # noqa: E402
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame  # noqa: E402
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, resize_img  # noqa: E402
from mast3r_slam.tracker import FrameTracker  # noqa: E402

from vid2cloud.engine.base import FrameResult, InputTransform  # noqa: E402
from vid2cloud.export.keyframes import keyframe_npz_path, write_keyframe_npz  # noqa: E402
from vid2cloud.export.trajectory import PosesHistoryWriter, write_trajectory_sim3  # noqa: E402

if Path(upstream_main.__file__).resolve() != (UPSTREAM_DIR / "main.py").resolve():
    raise ImportError(f"moduł `main` pochodzi z {upstream_main.__file__}, a nie z {UPSTREAM_DIR}")


def upstream_input_transform(full_h: int, full_w: int, size: int = IMG_SIZE) -> InputTransform:
    """Przekształcenie pełna klatka -> wejście modelu wzięte z `resize_img()` upstreamu.

    Tak samo jak `Intrinsics.__init__` (dataloader.py:286): `resize_img` na zerowym obrazie
    z `return_transformation=True` zwraca (W1/W, H1/H, połowa cropu w poziomie, w pionie).
    Upstream liczy połowę cropu jako (W - W_out) / 2; PIL tnie od `cx - halfw`, co dla
    nieparzystego W różni się o 0,5 px — dla 3840x2160 i 2160x3840 oba dają 0.
    """
    res, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((int(full_h), int(full_w), 3)), size, return_transformation=True
    )
    out_h, out_w = (int(v) for v in res["true_shape"][0])
    return InputTransform(
        scale_x=1.0 / scale_w,
        scale_y=1.0 / scale_h,
        left=float(half_crop_w),
        top=float(half_crop_h),
        resized_shape=(int(round(full_h / scale_h)), int(round(full_w / scale_w))),
        out_shape=(out_h, out_w),
    )


def upstream_commit() -> dict:
    """Commit submodule upstreamu i to, czy drzewo ma lokalne zmiany (łatki sm_120)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(UPSTREAM_DIR), *args], capture_output=True, text=True)

    head = git("rev-parse", "HEAD")
    dirty = git("diff", "--quiet", "HEAD")
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": dirty.returncode == 1 if dirty.returncode in (0, 1) else None,
    }


@contextlib.contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _sim3(T_WC) -> np.ndarray:
    """lietorch.Sim3 albo tensor (..., 8) -> numpy (8,) float64."""
    data = T_WC.data if isinstance(T_WC, lietorch.Sim3) else T_WC
    return data.detach().reshape(-1).cpu().numpy().astype(np.float64)


class Mast3rSlamEngine:
    """MASt3R-SLAM klatka po klatce. Wyniki i zrzuty zapisuje do `out_dir`."""

    name = "mast3r-slam"

    def __init__(
        self,
        out_dir: Path,
        config_path: Path = DEFAULT_CONFIG,
        single_thread: bool = True,
        device: str = "cuda:0",
        checkpoint: Path = CHECKPOINT,
    ):
        self.out_dir = Path(out_dir)
        self.kf_dir = self.out_dir / "keyframes"
        self.config_path = Path(config_path).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device

        load_config(str(self.config_path))
        config["single_thread"] = bool(single_thread)
        self.single_thread = bool(single_thread)
        if config["use_calib"]:
            raise NotImplementedError("adapter działa bez kalibracji, jak przebiegi upstreamu (bez --calib)")
        # TODO frame.py:117 — przy img_downsample > 1 pointmapa ma inną siatkę niż wejście
        # modelu, a input_transform() i lift tego nie uwzględniają.
        assert config["dataset"]["img_downsample"] == 1, "img_downsample != 1 nie jest obsłużony"
        # main.py:171 przerzedza dataset; tu każda podana klatka idzie do silnika.
        assert config["dataset"]["subsample"] == 1, "subsample != 1: przerzedzaj klatki przed push_frame"

        self._started = False
        self._finalized = False
        self._n_pushed = 0
        self._n_queued = 0  # zadań global_optimization zleconych przez nas
        self._n_done_logged = 0  # zadań, po których zapisano backend_done
        self._kf_known = 0  # klatek kluczowych już zrzuconych (kf_created / reloc)
        self._kf_source: list[str] = []
        self._kf_frame_index: list[int] = []
        self._pts: dict[int, float] = {}
        self._last_index: Optional[int] = None
        self.timings: dict[str, float] = {}
        # reloc_kf — klatki kluczowe z udanej relokalizacji; reloc_attempts — klatki przetworzone
        # w trybie RELOC (każda zleca relokalizację, main.py:286); tracking_lost — ile razy tracker
        # zwrócił try_reloc.
        self.counts = {"reloc_kf": 0, "reloc_attempts": 0, "tracking_lost": 0}

    # ------------------------------------------------------------------ #
    # Protokół Engine
    # ------------------------------------------------------------------ #

    def input_transform(self, full_h: int, full_w: int) -> InputTransform:
        return upstream_input_transform(full_h, full_w)

    def start(self, full_h: int, full_w: int) -> None:
        """Pamięć współdzielona, model i proces backendu (main.py:146-226)."""
        if self._started:
            return
        t0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = True  # main.py:147
        torch.set_grad_enabled(False)  # main.py:148
        ctx = mp.get_context("spawn")  # main.py:146 robi to samo globalnie

        self.transform = self.input_transform(full_h, full_w)
        h, w = self.transform.out_shape  # == dataset.get_img_shape()[0], main.py:172
        self.h, self.w = h, w

        self._manager = ctx.Manager()
        self.keyframes = SharedKeyframes(self._manager, h, w)
        self.states = SharedStates(self._manager, h, w)

        self.model = load_mast3r(path=str(self.checkpoint), device=self.device)
        self.model.share_memory()
        K = None  # use_calib == False, main.py:205

        self.tracker = FrameTracker(self.model, self.keyframes, self.device)
        self.backend = ctx.Process(
            target=upstream_main.run_backend, args=(config, self.model, self.states, self.keyframes, K)
        )
        with _cwd(UPSTREAM_DIR):
            self.backend.start()

        self.kf_dir.mkdir(parents=True, exist_ok=True)
        self.history = PosesHistoryWriter(self.out_dir / "poses_history.csv")
        self._started = True
        self.timings["setup_s"] = time.perf_counter() - t0

    def push_frame(self, rgb: np.ndarray, index: int, pts_s: float) -> FrameResult:
        """Jedna iteracja pętli `main.py:233-310` dla klatki RGB uint8 (H, W, 3)."""
        if not self._started:
            self.start(*rgb.shape[:2])
        if self._finalized:
            raise RuntimeError("przebieg zakończony — push_frame po finalize()")
        index = int(index)
        self._pts[index] = float(pts_s)
        self._last_index = index
        kf_before = self._kf_known
        wait_s = 0.0

        mode = self.states.get_mode()
        img = rgb.astype(np.float32) / 255.0  # dataloader.py:50
        T_WC = (
            lietorch.Sim3.Identity(1, device=self.device)
            if self._n_pushed == 0
            else self.states.get_frame().T_WC
        )
        frame = create_frame(index, img, T_WC, img_size=IMG_SIZE, device=self.device)

        add_new_kf = False
        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            with self.keyframes.lock:
                self.keyframes.append(frame)
                self._register_new_keyframes(index, own="init")
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self._n_queued += 1
            self.states.set_mode(Mode.TRACKING)
            self.states.set_frame(frame)
        elif mode == Mode.TRACKING:
            add_new_kf, _match_info, try_reloc = self.tracker.track(frame)
            if try_reloc:
                self.counts["tracking_lost"] += 1
                self.states.set_mode(Mode.RELOC)
            self.states.set_frame(frame)
        elif mode == Mode.RELOC:
            # W main.py add_new_kf zostaje tu z poprzedniej iteracji, a ta zawsze kończy się
            # na track() == (False, [], True) — więc jest False, tak jak tutaj.
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self.states.set_frame(frame)
            self.states.queue_reloc()
            self.counts["reloc_attempts"] += 1
            t_wait = time.perf_counter()
            while config["single_thread"]:  # main.py:288
                with self.states.lock:
                    if self.states.reloc_sem.value == 0:
                        break
                self._check_backend()
                time.sleep(0.01)
            wait_s += time.perf_counter() - t_wait
            if self.single_thread:
                self._register_new_keyframes(index, own=None)
        else:
            raise RuntimeError(f"nieznany tryb silnika: {mode}")

        if add_new_kf:
            with self.keyframes.lock:
                self.keyframes.append(frame)
                self._register_new_keyframes(index, own="tracker")
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self._n_queued += 1
            t_wait = time.perf_counter()
            while config["single_thread"]:  # main.py:301
                with self.states.lock:
                    if len(self.states.global_optimizer_tasks) == 0:
                        break
                self._check_backend()
                time.sleep(0.01)
            wait_s += time.perf_counter() - t_wait
            if self.single_thread:
                self._log_backend_done(index)

        if not self.single_thread:
            self._register_new_keyframes(index, own=None)
            self._poll_backend_done(index)

        self._n_pushed += 1
        new = tuple((k, self._kf_frame_index[k]) for k in range(kf_before, self._kf_known))
        own = [k for k, fi in new if fi == index]
        return FrameResult(
            index=index,
            pts_s=float(pts_s),
            is_keyframe=bool(own),
            kf_index=own[0] if own else None,
            mode=mode.name,
            T_WC=_sim3(frame.T_WC),
            input_shape=(self.h, self.w),
            pointmap=frame.X_canon.detach().reshape(-1, 3).cpu().numpy(),
            conf=frame.get_average_conf().detach().reshape(-1).cpu().numpy(),
            n_kf=self._kf_known,
            new_keyframes=new,
            t_backend_wait_ms=1000.0 * wait_s,
        )

    def finalize(self) -> dict[str, Path]:
        """Koniec przebiegu: TERMINATED, join backendu, zrzuty końcowe, chmura, trajektoria."""
        if not self._started:
            raise RuntimeError("finalize() przed pierwszą klatką")
        if self._finalized:
            raise RuntimeError("finalize() wywołane drugi raz")
        t0 = time.perf_counter()
        self.states.set_mode(Mode.TERMINATED)  # main.py:250
        self.backend.join()
        self._finalized = True
        last = self._last_index if self._last_index is not None else -1
        # Relokalizacja zakończona już po ostatniej klatce (bez single_thread).
        self._register_new_keyframes(last, own=None)
        with self.states.lock:
            self.counts["backend_tasks_left"] = len(self.states.global_optimizer_tasks)

        n_kf = len(self.keyframes)
        poses = self._all_poses()
        self.history.write_event("final", last, poses)
        self.history.close()

        pts = []
        for k in range(n_kf):
            snap = self._snapshot(k)
            write_keyframe_npz(keyframe_npz_path(self.kf_dir, k, "final"), **snap)
            pts.append(snap["pts_s"])
            uimg = (self.keyframes.uimg[k].cpu().numpy() * 255).astype(np.uint8)  # evaluate.py:83
            cv2.imwrite(str(self.kf_dir / f"kf_{k:03d}.png"), cv2.cvtColor(uimg, cv2.COLOR_RGB2BGR))

        upstream_eval.save_reconstruction(self.out_dir, "cloud.ply", self.keyframes, CLOUD_C_CONF_THRESHOLD)
        write_trajectory_sim3(self.out_dir / "trajectory_sim3.txt", pts, poses)
        self.timings["finalize_s"] = time.perf_counter() - t0
        self._manager.shutdown()
        return {
            "cloud": self.out_dir / "cloud.ply",
            "trajectory_sim3": self.out_dir / "trajectory_sim3.txt",
            "poses_history": self.out_dir / "poses_history.csv",
            "keyframes": self.kf_dir,
        }

    # ------------------------------------------------------------------ #
    # Haki
    # ------------------------------------------------------------------ #

    def _all_poses(self) -> np.ndarray:
        with self.keyframes.lock:
            n = len(self.keyframes)
            return self.keyframes.T_WC[:n, 0, :].detach().cpu().numpy().astype(np.float64)

    def _snapshot(self, k: int) -> dict:
        """Stan klatki kluczowej k z pamięci współdzielonej, w formacie export/keyframes."""
        with self.keyframes.lock:
            kf = self.keyframes[k]
            T_WC = _sim3(kf.T_WC)
            X = kf.X_canon.detach().cpu().numpy()
            C = kf.C.detach().cpu().numpy()
            shape = [int(v) for v in kf.img_true_shape.reshape(-1).tolist()]
            frame_index, N, N_updates = int(kf.frame_id), int(kf.N), int(kf.N_updates)
        h, w = shape
        return {
            "kf": k,
            "frame_index": frame_index,
            "pts_s": self._pts.get(frame_index, float("nan")),
            "source": self._kf_source[k],
            "T_WC": T_WC,
            "X_canon": X.reshape(h, w, 3),
            "C": C.reshape(h, w),
            "N": N,
            "N_updates": N_updates,
            "img_true_shape": (h, w),
        }

    def _register_new_keyframes(self, frame_index: int, own: Optional[str]) -> None:
        """Zrzuca klatki kluczowe, które pojawiły się od ostatniego razu (po len(keyframes)).

        `own` — źródło klatki właśnie dołożonej przez tę pętlę ("init" / "tracker"); jest
        nią ostatnia. Pozostałe nowe (i wszystkie przy `own=None`) dołożył backend przy
        relokalizacji. Odczyt pod `keyframes.lock`, który `relocalization()` trzyma przez
        całe dołożenie-i-ewentualne-zdjęcie klatki, więc nieudanej próby tu nie zobaczymy.
        """
        with self.keyframes.lock:
            n = len(self.keyframes)
            if n <= self._kf_known:
                return
            for k in range(self._kf_known, n):
                source = own if (own is not None and k == n - 1) else "reloc"
                self._kf_source.append(source)
                if source == "reloc":
                    self.counts["reloc_kf"] += 1
                snap = self._snapshot(k)
                self._kf_frame_index.append(snap["frame_index"])
                write_keyframe_npz(keyframe_npz_path(self.kf_dir, k, "created"), **snap)
                event = "reloc" if source == "reloc" else "kf_created"
                self.history.write_event(event, snap["frame_index"] if source == "reloc" else frame_index,
                                         self._all_poses()[: k + 1])
            self._kf_known = n

    def _check_backend(self) -> None:
        """W pętlach czekania `main.py` wisiałby w nieskończoność, gdyby backend padł."""
        if not self.backend.is_alive():
            raise RuntimeError(f"proces backendu zakończył się (kod {self.backend.exitcode}) w trakcie czekania")

    def _log_backend_done(self, frame_index: int) -> None:
        with self.states.lock:
            done = self._n_queued - len(self.states.global_optimizer_tasks)
        self.history.write_event("backend_done", frame_index, self._all_poses())
        self._n_done_logged = done

    def _poll_backend_done(self, frame_index: int) -> None:
        """Bez single_thread: zdarzenie, jeśli od ostatniego razu backend skończył zadanie."""
        with self.states.lock:
            done = self._n_queued - len(self.states.global_optimizer_tasks)
        if done > self._n_done_logged:
            self._log_backend_done(frame_index)
