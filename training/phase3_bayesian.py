"""
Phase 3: Single-Objective Bayesian Optimisation via Optuna TPE.

Searches only the five **normalised** loss weights (BCE, Dice, Boundary, Focal,
HD95-DT surrogate). Optimizer settings remain fixed to the supplied baseline
configuration so the ablation compares loss weights without LR confounding.

Objective = 0.5 × Dice  +  0.3 × (1 − hd95/100)  +  0.2 × BoundaryF1

Design notes
-----------
* Persistent SQLite storage at <output_dir>/phase3/studies/<study-name>/ enables
  full resume support — if the process crashes or is interrupted, re-run the
  same command and Optuna will continue from where it left off.

* Raw weights are suggested independently by TPE then L1-normalised to sum=1,
  so CompositeMaskLoss always receives a simplex point. The raw values are
  stored in the study, the normalised values go into the config.

* Phase3TrialTrainer wraps (does NOT subclass) the existing Trainer and
  monkey-patches its bound _val_epoch method on the instance so that
  per-epoch Optuna reporting works without any change
  to Trainer itself.

* Trials run sequentially (n_jobs=1) and require CUDA — this phase is intended
  for the RTX 4050 GPU, not a silent CPU fallback.

* All Phase 1 / 2 / MGDA outputs are untouched.  Phase 3 writes exclusively
  to <output_dir>/phase3/.

* n_trials is the target number of successful COMPLETE trials. Failed or
  interrupted attempts remain recorded but are retried when resuming.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def _phase3_study_lock(lock_path: Path):
    """Hold an OS-released exclusive lock for a Phase 3 study directory.

    Optuna's SQLite study is not safe for two independent training processes
    when startup recovery marks all RUNNING trials as failed. The OS lock
    prevents a second process from changing a live trial; the lock is released
    automatically if its owner exits or crashes.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"A Phase 3 study process already holds {lock_path}. "
                    "Wait for it to finish before resuming this study."
                ) from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"A Phase 3 study process already holds {lock_path}. "
                    "Wait for it to finish before resuming this study."
                ) from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        locked = True
        yield
    finally:
        if locked:
            unlock()
        stream.close()

# ---------------------------------------------------------------------------
# Objective-formula constants
# ---------------------------------------------------------------------------
_DICE_COEFF: float = 0.50
_HD95_COEFF: float = 0.30
_BDF1_COEFF: float = 0.20

# ---------------------------------------------------------------------------
# Search-space bounds
# (raw values; normalised to sum=1 before they reach CompositeMaskLoss)
# ---------------------------------------------------------------------------
_LOSS_WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha":   (0.0, 1.0),   # BCE weight
    "beta":    (0.0, 1.0),   # Soft-Dice weight
    "gamma":   (0.0, 1.0),   # Boundary weight
    "delta":   (0.0, 0.5),   # Focal weight  — capped to avoid domination
    "epsilon": (0.0, 0.3),   # HD95-DT surrogate — capped (slow)
}

# Additional hyperparameters ("more options")
# Default TPE startup trials before the surrogate model kicks in
_TPE_N_STARTUP: int = 10

# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------
_CSV_FIELDNAMES: list[str] = [
    "trial_number", "state", "elapsed_s",
    # raw (un-normalised) weights
    "alpha_raw", "beta_raw", "gamma_raw", "delta_raw", "epsilon_raw",
    # normalised weights (what the model actually used)
    "alpha_norm", "beta_norm", "gamma_norm", "delta_norm", "epsilon_norm",
    # extra hyperparameters
    "bce_pos_weight", "lr",
    # objective and components
    "score", "dice", "hd95", "boundary_f1",
    # bookkeeping
    "best_epoch", "pruned",
]


# ===========================================================================
# Public helper functions (also imported by tests)
# ===========================================================================

def normalise_weights(raw: dict[str, float]) -> dict[str, float]:
    """
    L1-project five raw loss weights onto the probability simplex (sum = 1).

    If all weights are effectively zero (sum < 1e-9) a uniform distribution
    is returned as a safe fallback; the caller should guard against this by
    clamping at least one weight to a small positive value.
    """
    keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    total = sum(raw[k] for k in keys)
    if total < 1e-9:
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: raw[k] / total for k in keys}


def compute_objective(metrics: dict) -> float:
    """
    Composite scalar objective for a single trial.

    Formula
    -------
        score = 0.5 × Dice  +  0.3 × (1 − hd95_norm)  +  0.2 × BoundaryF1

    where hd95_norm = min(hd95, 100) / 100 so lower HD95 ⇒ higher score.

    Missing or non-finite metrics default to their worst-case values:
      * dice → 0.0
      * hd95 → 999.0  (clipped to 100.0 ⇒ contributes 0 to the score)
      * boundary_f1 → 0.0

    Parameters
    ----------
    metrics : dict
        Expects keys ``val_dice``, ``val_hd95``, ``val_boundary_f1``
        (the naming convention used by Trainer._val_epoch).
    """
    def _safe(v, worst):
        if v is None or not np.isfinite(float(v)):
            return worst
        return float(v)

    dice = _safe(metrics.get("val_dice"),        0.0)
    hd95 = _safe(metrics.get("val_hd95"),      999.0)
    bdf1 = _safe(metrics.get("val_boundary_f1"), 0.0)

    hd95_norm = min(hd95, 100.0) / 100.0
    return _DICE_COEFF * dice + _HD95_COEFF * (1.0 - hd95_norm) + _BDF1_COEFF * bdf1


# ===========================================================================
# Trial-aware trainer wrapper
# ===========================================================================

class Phase3TrialTrainer:
    """
    Wraps the existing ``Trainer`` and injects per-epoch Optuna pruning support
    **without modifying Trainer itself**.

    Strategy
    --------
    ``Trainer.train()`` calls ``self._val_epoch(epoch)`` once per epoch.
    ``Phase3TrialTrainer.train()`` replaces that bound method on the Trainer
    *instance* (not the class) with a thin wrapper that:

    1. Calls the original ``_val_epoch`` and collects its result.
    2. Computes the composite objective score.
    3. Reports it to the Optuna trial via ``trial.report(score, epoch)``.
    4. Raises ``optuna.exceptions.TrialPruned`` if ``trial.should_prune()`` is
       True.  This propagates through ``train()`` back to the objective closure
       where it is re-raised so Optuna can mark the trial as PRUNED.

    The original method is restored in a ``finally`` block so a leaked trainer
    reference can never carry stale state.
    """

    def __init__(
        self,
        config: dict,
        run_dir: str,
        optuna_trial: Optional[Any] = None,
    ) -> None:
        from .trainer import Trainer          # avoid circular at import time

        self._trainer = Trainer(config, run_dir)
        self._optuna_trial = optuna_trial
        self.run_dir = self._trainer.run_dir  # expose for external callers

    # ------------------------------------------------------------------
    def train(self) -> dict:
        """
        Delegate to Trainer.train() with per-epoch Optuna pruning injected.

        Returns the same summary dict that Trainer.train() returns.
        Raises optuna.exceptions.TrialPruned if the trial is pruned.
        """
        try:
            import optuna
        except ImportError:
            optuna = None

        trial   = self._optuna_trial
        trainer = self._trainer

        # Save the bound method so we can restore it
        orig_val_epoch = trainer._val_epoch

        def _pruning_val_epoch(epoch: int) -> dict:
            result = orig_val_epoch(epoch)
            if trial is not None and optuna is not None:
                score = compute_objective(result)
                if np.isfinite(score):
                    trial.report(score, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return result

        # Install the hook on the instance (not the class)
        trainer._val_epoch = _pruning_val_epoch
        try:
            return trainer.train()
        finally:
            # Always restore to avoid leaking state
            trainer._val_epoch = orig_val_epoch


# ===========================================================================
# CSV persistence helpers
# ===========================================================================

def _init_csv(csv_path: Path) -> None:
    """Write a header row if the CSV does not yet exist."""
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES,
                       extrasaction="ignore").writeheader()


def _append_trial_csv(
    csv_path: Path,
    *,
    trial_number: int,
    state: str,
    elapsed_s: float,
    raw_weights: dict[str, float],
    norm_weights: dict[str, float],
    extra_hp: dict[str, float],
    score: float,
    dice: float,
    hd95: float,
    bdf1: float,
    best_epoch: int,
    pruned: bool,
) -> None:
    """Append one trial row to the running CSV file."""
    row: dict[str, Any] = {
        "trial_number": trial_number,
        "state":        state,
        "elapsed_s":    f"{elapsed_s:.1f}",
        **{f"{k}_raw":  v for k, v in raw_weights.items()},
        **{f"{k}_norm": v for k, v in norm_weights.items()},
        **extra_hp,
        "score":        f"{score:.6f}",
        "dice":         f"{dice:.6f}",
        "hd95":         f"{hd95:.4f}",
        "boundary_f1":  f"{bdf1:.6f}",
        "best_epoch":   best_epoch,
        "pruned":       str(pruned),
    }
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES,
                       extrasaction="ignore").writerow(row)


def _rebuild_csv_from_study(study: Any, csv_path: Path) -> None:
    """
    Regenerate the full CSV from the Optuna study.

    Called on resume when the existing CSV is missing or has fewer rows than
    the number of completed trials in the SQLite store.  This ensures the CSV
    is always consistent with the authoritative SQLite record.
    """
    _init_csv(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES,
                                extrasaction="ignore")
        writer.writeheader()
        for t in sorted(study.trials, key=lambda x: x.number):
            ua = t.user_attrs
            rw = ua.get("raw_weights", t.params)
            nw = ua.get("norm_weights", t.params)
            extra = {
                "bce_pos_weight": ua.get("bce_pos_weight",
                                         t.params.get("bce_pos_weight", "")),
                "lr":             ua.get("lr", t.params.get("lr", "")),
            }
            is_pruned = t.state.name == "PRUNED"
            row = {
                "trial_number": t.number,
                "state":        t.state.name,
                "elapsed_s":    (
                    f"{t.duration.total_seconds():.1f}"
                    if t.duration else ""
                ),
                **{f"{k}_raw":  rw.get(k, "") for k in
                   ["alpha", "beta", "gamma", "delta", "epsilon"]},
                **{f"{k}_norm": nw.get(k, "") for k in
                   ["alpha", "beta", "gamma", "delta", "epsilon"]},
                **extra,
                "score":       f"{t.value:.6f}" if t.value is not None else "",
                "dice":        f"{ua.get('dice', ''):.6f}"
                               if ua.get("dice") is not None else "",
                "hd95":        f"{ua.get('hd95', ''):.4f}"
                               if ua.get("hd95") is not None else "",
                "boundary_f1": f"{ua.get('boundary_f1', ''):.6f}"
                               if ua.get("boundary_f1") is not None else "",
                "best_epoch":  ua.get("best_epoch", ""),
                "pruned":      str(is_pruned),
            }
            writer.writerow(row)


# ===========================================================================
# Best-config JSON helpers
# ===========================================================================

def _update_best_json(study: Any, best_json: Path) -> None:
    """
    Write / overwrite the best-so-far configuration JSON after every trial.
    Silently skips if no trials are complete yet.
    """
    try:
        best = study.best_trial
    except ValueError:
        return   # no completed trials yet

    ua   = best.user_attrs
    norm = ua.get("norm_weights", best.params)
    raw  = ua.get("raw_weights",  best.params)

    result: dict[str, Any] = {
        "study_name":        study.study_name,
        "best_trial":        best.number,
        "best_score":        best.value,
        "objective_formula": (
            "0.5*dice + 0.3*(1 - min(hd95,100)/100) + 0.2*boundary_f1"
        ),
        # Normalised weights — use these in production
        "best_norm_weights": norm,
        # Raw un-normalised values — for reference / reproducibility
        "best_raw_weights":  raw,
        # Secondary hyperparameters
        "best_bce_pos_weight": ua.get("bce_pos_weight"),
        "best_lr":             ua.get("lr"),
        # Metrics
        "best_dice":         ua.get("dice"),
        "best_hd95":         ua.get("hd95"),
        "best_boundary_f1":  ua.get("boundary_f1"),
        "best_epoch":        ua.get("best_epoch"),
        # Study summary
        "total_trials":     len(study.trials),
        "completed_trials": sum(1 for t in study.trials
                                if t.state.name == "COMPLETE"),
        "pruned_trials":    sum(1 for t in study.trials
                                if t.state.name == "PRUNED"),
    }

    # Scrub any NaN / Inf before JSON serialisation
    def _clean(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    clean = {k: _clean(v) for k, v in result.items()}
    with best_json.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)


def _build_best_result(study: Any, best_json: Path) -> dict:
    """Return the final best-result dict (from file when available)."""
    if best_json.exists():
        try:
            with best_json.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    try:
        best = study.best_trial
        return {
            "best_trial":  best.number,
            "best_score":  best.value,
            "best_params": best.params,
        }
    except ValueError:
        return {"error": "no completed trials found in study"}


# ===========================================================================
# Main entry point
# ===========================================================================

def _run_phase3_bayesian_unlocked(
    base_cfg: dict,
    output_dir: str,
    n_trials: int = 30,
    epochs_per_trial: int = 10,
    seed: int = 42,
    study_name: str = "lung_nodule_loss_weights_v1",
) -> dict:
    """
    Run (or resume) the Phase 3 Bayesian optimisation study.

    Parameters
    ----------
    base_cfg : dict
        Base training configuration.  ``epochs`` and ``mask_loss`` are
        overridden per trial.  Other keys (data_root, batch_size, etc.) are
        inherited unchanged.
    output_dir : str
        Root output directory. Phase 3 artefacts go to
        ``<output_dir>/phase3/studies/<study-name>/``.
        Phase 1 / 2 / MGDA outputs are **not touched**.
        n_trials : int
        Target number of successfully completed trials. Failed or interrupted
        attempts do not consume this target; rerunning resumes safely.
    epochs_per_trial : int
        Training epochs per trial (default 10; full training uses 30).
    seed : int
        TPE sampler seed for reproducibility across restarts.
    study_name : str
        Optuna study name.  Changing it starts a fresh study from scratch.

    Returns
    -------
    dict
        Best-trial result, also written to the selected study directory.

    Notes
    -----
    * The SQLite database in the selected study directory is the
      authoritative record.  The CSV file is rebuilt from the DB on resume if
      it is missing or incomplete.
    * Trials are labelled ``p3_trial_NNNN`` and stored in their own sub-directory
      under ``<output_dir>/phase3/studies/<study-name>/trial_NNNN/``.
    * Weights are suggested raw then L1-normalised to sum=1 so the model always
      receives a simplex point.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if epochs_per_trial < 1:
        raise ValueError("epochs_per_trial must be at least 1")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Install optuna>=3.0:  pip install optuna") from exc

    # ---- directory / path setup -----------------------------------------
    phase3_root = Path(output_dir) / "phase3"
    study_slug = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in study_name
    ).strip("._")
    if not study_slug or study_slug == "..":
        raise ValueError("study_name must contain at least one safe filename character")
    phase3_dir = phase3_root / "studies" / study_slug
    phase3_dir.mkdir(parents=True, exist_ok=True)

    db_path   = phase3_dir / "optuna_study.db"
    csv_path  = phase3_dir / "trials.csv"
    best_json = phase3_dir / "phase3_best.json"

    # POSIX separators keep Windows drive paths valid in SQLAlchemy URLs.
    storage_url = f"sqlite:///{db_path.resolve().as_posix()}"
    # Optuna applies str.format() to the URL template; escape literal braces
    # that can occur in Windows profile/temp paths (e.g. sandbox GUID folders).
    storage_url = storage_url.replace("{", "{{").replace("}", "}}")

    logger.info(
        "Phase 3 Bayesian optimisation | study=%s | n_trials_target=%d | "
        "epochs_per_trial=%d | storage=%s",
        study_name, n_trials, epochs_per_trial, storage_url,
    )

    # ---- create / resume study ------------------------------------------
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=_TPE_N_STARTUP,
        ),
        # Every scheduled trial must run all configured epochs; pruning would
        # make a nominal 10-epoch Phase 3 comparison inconsistent.
        pruner=optuna.pruners.NopPruner(),
        storage=storage_url,
        load_if_exists=True,   # <— resume support
    )

    # A process killed mid-trial leaves a RUNNING row in SQLite. Mark the
    # abandoned attempt failed so a restarted study can continue.
    for previous_trial in study.get_trials(deepcopy=False):
        if previous_trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(previous_trial.number, state=optuna.trial.TrialState.FAIL)

    completed_before = sum(
        1 for t in study.trials if t.state.name == "COMPLETE"
    )
    logger.info(
        "Study loaded.  Completed=%d | Pruned=%d | Total=%d",
        completed_before,
        sum(1 for t in study.trials if t.state.name == "PRUNED"),
        len(study.trials),
    )

    # Rebuild CSV if it was lost or is incomplete relative to the DB
    csv_rows = 0
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8") as fh:
                csv_rows = sum(1 for _ in fh) - 1  # exclude header
        except OSError:
            csv_rows = 0

    if csv_rows < len(study.trials):
        logger.info(
            "CSV has %d rows but study has %d trials — rebuilding CSV from DB.",
            csv_rows, len(study.trials),
        )
        _rebuild_csv_from_study(study, csv_path)
    else:
        _init_csv(csv_path)

    # ---- objective closure ----------------------------------------------
    def objective(trial: optuna.Trial) -> float:
        t0 = time.monotonic()

        # --- suggest raw (un-normalised) loss weights --------------------
        raw_weights: dict[str, float] = {
            key: trial.suggest_float(key, lo, hi)
            for key, (lo, hi) in _LOSS_WEIGHT_BOUNDS.items()
        }

        # The all-zero corner has a defined uniform-simplex fallback; no
        # per-objective minimum is imposed on the sampled weights.
        norm_weights = normalise_weights(raw_weights)

        # Keep optimizer and BCE class weighting at the baseline settings.
        # Phase 3's search space is strictly the five requested loss weights.
        base_loss = base_cfg.get("mask_loss", {})
        bce_pos_weight = float(base_loss.get("bce_pos_weight", 1.0))
        lr = float(base_cfg.get("lr", 0.005))

        # --- build trial config ------------------------------------------
        cfg = deepcopy(base_cfg)
        cfg["config_id"] = f"p3_trial_{trial.number:04d}"
        cfg["epochs"]    = int(epochs_per_trial)
        cfg["require_cuda"] = True
        cfg["lr"]        = lr
        cfg["mask_loss"] = {
            **norm_weights,
            "bce_pos_weight": bce_pos_weight,
        }
        # Annotate config for downstream inspection
        cfg["phase3_trial_number"]  = trial.number
        cfg["phase3_raw_weights"]   = raw_weights
        cfg["phase3_norm_weights"]  = norm_weights
        cfg["phase3_study_name"]    = study_name

        # Store in Optuna for the dashboard / analysis
        trial.set_user_attr("norm_weights",   norm_weights)
        trial.set_user_attr("raw_weights",    raw_weights)
        trial.set_user_attr("bce_pos_weight", bce_pos_weight)
        trial.set_user_attr("lr",             lr)
        trial.set_user_attr("phase3_process_id", os.getpid())

        run_dir = phase3_dir / f"trial_{trial.number:04d}"

        # --- bookkeeping defaults ----------------------------------------
        pruned     = False
        dice       = 0.0
        hd95       = 999.0
        bdf1       = 0.0
        best_epoch = 0
        score      = 0.0
        state      = "FAIL"

        try:
            trainer = Phase3TrialTrainer(cfg, str(run_dir),
                                         optuna_trial=trial)
            summary = trainer.train()

            # Prefer best-epoch metrics over final-epoch metrics
            best_m  = summary.get("best_metrics", {})
            final_m = summary.get("final_metrics", {})
            m       = best_m if best_m else final_m

            def _safe(v, worst):
                return worst if (v is None or not np.isfinite(float(v))) else float(v)

            dice  = _safe(m.get("val_dice"),        0.0)
            hd95  = _safe(m.get("val_hd95"),      999.0)
            bdf1  = _safe(m.get("val_boundary_f1"), 0.0)
            best_epoch = int(summary.get("best_epoch", 0))
            score = compute_objective(
                {"val_dice": dice, "val_hd95": hd95, "val_boundary_f1": bdf1}
            )
            state = "COMPLETE"

            # Persist metrics so the dashboard and best.json have them
            trial.set_user_attr("dice",        dice)
            trial.set_user_attr("hd95",        hd95)
            trial.set_user_attr("boundary_f1", bdf1)
            trial.set_user_attr("best_epoch",  best_epoch)
            trial.set_user_attr("score",       score)

        except optuna.exceptions.TrialPruned:
            pruned = True
            state  = "PRUNED"
            raise   # let Optuna record it as PRUNED

        except Exception as exc:
            logger.warning(
                "Trial %d failed: %s", trial.number, exc, exc_info=True
            )
            state = "FAIL"
            raise

        finally:
            elapsed = time.monotonic() - t0
            _append_trial_csv(
                csv_path,
                trial_number=trial.number,
                state=state,
                elapsed_s=elapsed,
                raw_weights=raw_weights,
                norm_weights=norm_weights,
                extra_hp={"bce_pos_weight": bce_pos_weight, "lr": lr},
                score=score,
                dice=dice,
                hd95=hd95,
                bdf1=bdf1,
                best_epoch=best_epoch,
                pruned=pruned,
            )
            # Keep best.json up-to-date after every completed trial
            _update_best_json(study, best_json)

        return score

    # ---- compute how many more trials to run ----------------------------
    completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    if completed >= n_trials:
        logger.info(
            "Target of %d completed trials already reached (%d complete). "
            "Nothing to do — returning best result.",
            n_trials, completed,
        )
        _update_best_json(study, best_json)
        return _build_best_result(study, best_json)

    # Failed/pruned attempts do not satisfy the requested number of completed
    # experiments. Retry pruned attempts in this invocation; ordinary errors
    # propagate so they are visible and can be resumed explicitly later.
    while completed < n_trials:
        remaining = n_trials - completed
        logger.info(
            "Running up to %d more trial(s) to reach %d completed (have %d).",
            remaining, n_trials, completed,
        )
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,               # sequential — one RTX 4050 6 GB GPU
            show_progress_bar=True,
            gc_after_trial=True,    # free GPU memory between trials
        )
        completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")

    # Finalize aggregate files after Optuna has committed each trial state.
    _update_best_json(study, best_json)
    _rebuild_csv_from_study(study, csv_path)
    return _build_best_result(study, best_json)


def run_phase3_bayesian(
    base_cfg: dict,
    output_dir: str,
    n_trials: int = 30,
    epochs_per_trial: int = 10,
    seed: int = 42,
    study_name: str = "lung_nodule_loss_weights_v1",
) -> dict:
    """Run/resume Phase 3 while excluding concurrent writers to its SQLite DB."""
    phase3_root = Path(output_dir) / "phase3"
    with _phase3_study_lock(phase3_root / ".study.lock"):
        return _run_phase3_bayesian_unlocked(
            base_cfg=base_cfg,
            output_dir=output_dir,
            n_trials=n_trials,
            epochs_per_trial=epochs_per_trial,
            seed=seed,
            study_name=study_name,
        )
