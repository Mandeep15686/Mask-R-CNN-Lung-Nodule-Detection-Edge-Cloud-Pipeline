"""
Tests for boundary loss warm-up schedulers.
Run: pytest tests/test_warmup.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import importlib.util

def load(rel):
    spec = importlib.util.spec_from_file_location(rel, rel + ".py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

warmup_mod = load("training/warmup")
BoundaryWarmupScheduler    = warmup_mod.BoundaryWarmupScheduler
AdaptiveLossWeightScheduler = warmup_mod.AdaptiveLossWeightScheduler


# ---------------------------------------------------------------------------
# BoundaryWarmupScheduler
# ---------------------------------------------------------------------------
class TestBoundaryWarmupScheduler:

    @pytest.mark.parametrize("schedule", ["step", "linear", "cosine"])
    def test_starts_at_zero(self, schedule):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule=schedule)
        assert ws.get_gamma(1) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("schedule", ["step", "linear", "cosine"])
    def test_reaches_target_at_warmup_end(self, schedule):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule=schedule)
        assert ws.get_gamma(10) == pytest.approx(0.5, abs=1e-5)

    @pytest.mark.parametrize("schedule", ["step", "linear", "cosine"])
    def test_stays_at_target_after_warmup(self, schedule):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule=schedule)
        for ep in [11, 15, 20, 30]:
            assert ws.get_gamma(ep) == pytest.approx(0.5, abs=1e-5), f"Failed at epoch {ep}"

    def test_linear_monotone(self):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="linear")
        gammas = [ws.get_gamma(ep) for ep in range(1, 11)]
        assert all(gammas[i] <= gammas[i+1] for i in range(len(gammas)-1))

    def test_cosine_monotone(self):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="cosine")
        gammas = [ws.get_gamma(ep) for ep in range(1, 11)]
        assert all(gammas[i] <= gammas[i+1] + 1e-8 for i in range(len(gammas)-1))

    def test_step_binary(self):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="step")
        for ep in range(1, 10):
            assert ws.get_gamma(ep) == pytest.approx(0.0, abs=1e-5), f"Non-zero before warmup end at ep={ep}"
        assert ws.get_gamma(10) == pytest.approx(0.5, abs=1e-5)

    def test_all_values_in_range(self):
        for sched in ["step", "linear", "cosine"]:
            ws = BoundaryWarmupScheduler(50, warmup_epochs=20, target_gamma=0.7, schedule=sched)
            for ep in range(1, 51):
                g = ws.get_gamma(ep)
                assert 0.0 <= g <= 0.7 + 1e-5, f"{sched} epoch {ep}: gamma={g} out of range"

    def test_schedule_summary_length(self):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="linear")
        summary = ws.schedule_summary()
        assert len(summary) == 30
        assert summary[0]["epoch"]  == 1
        assert summary[-1]["epoch"] == 30

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="invalid")
            ws.get_gamma(5)

    def test_warmup_longer_than_total_raises(self):
        with pytest.raises(AssertionError):
            BoundaryWarmupScheduler(10, warmup_epochs=20, target_gamma=0.5)

    def test_epoch_zero_returns_start(self):
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule="linear")
        assert ws.get_gamma(0) == pytest.approx(0.0, abs=1e-5)

    def test_custom_start_gamma(self):
        ws = BoundaryWarmupScheduler(
            30, warmup_epochs=10, target_gamma=0.5,
            start_gamma=0.1, schedule="linear"
        )
        assert ws.get_gamma(1) == pytest.approx(0.1, abs=1e-5)
        assert ws.get_gamma(10) == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# AdaptiveLossWeightScheduler
# ---------------------------------------------------------------------------
class TestAdaptiveLossWeightScheduler:

    def test_phase_a_has_zero_gamma(self):
        sched = AdaptiveLossWeightScheduler(
            total_epochs=30, warmup_epochs=10,
            alpha_start=0.5, alpha_end=0.3,
            beta=0.5, gamma_target=0.5,
        )
        w = sched.get_weights(1)
        assert w["gamma"] == pytest.approx(0.0, abs=1e-5)

    def test_phase_b_has_target_gamma(self):
        sched = AdaptiveLossWeightScheduler(
            total_epochs=30, warmup_epochs=10,
            alpha_start=0.5, alpha_end=0.3,
            beta=0.5, gamma_target=0.5,
        )
        w = sched.get_weights(10)
        assert w["gamma"] == pytest.approx(0.5, abs=1e-5)

    def test_alpha_decreases_during_warmup(self):
        sched = AdaptiveLossWeightScheduler(
            total_epochs=30, warmup_epochs=10,
            alpha_start=0.5, alpha_end=0.3,
            beta=0.5, gamma_target=0.5,
        )
        alphas = [sched.get_weights(ep)["alpha"] for ep in range(1, 11)]
        assert all(alphas[i] >= alphas[i+1] - 1e-8 for i in range(len(alphas)-1))

    def test_alpha_stable_after_warmup(self):
        sched = AdaptiveLossWeightScheduler(
            total_epochs=30, warmup_epochs=10,
            alpha_start=0.5, alpha_end=0.3,
            beta=0.5, gamma_target=0.5,
        )
        for ep in [11, 20, 30]:
            assert sched.get_weights(ep)["alpha"] == pytest.approx(0.3, abs=1e-5)

    def test_beta_constant(self):
        sched = AdaptiveLossWeightScheduler(30, 10, 0.5, 0.3, beta=0.7, gamma_target=0.5)
        for ep in range(1, 31):
            assert sched.get_weights(ep)["beta"] == pytest.approx(0.7, abs=1e-5)

    def test_all_weights_have_required_keys(self):
        sched = AdaptiveLossWeightScheduler(30, 10, 0.5, 0.3, 0.5, 0.5)
        w = sched.get_weights(15)
        assert all(k in w for k in ("alpha", "beta", "gamma", "delta", "epsilon"))

    def test_weights_non_negative(self):
        sched = AdaptiveLossWeightScheduler(30, 10, 0.5, 0.3, 0.5, 0.5)
        for ep in range(1, 31):
            w = sched.get_weights(ep)
            assert all(v >= 0.0 for v in w.values()), f"Negative weight at epoch {ep}: {w}"
