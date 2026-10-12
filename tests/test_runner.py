"""Phase 6 check: the example runner executes end to end. Run from the repo root: python -m pytest"""

import contextlib
import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

from energy_quant_engine.main_analytics import main


def test_main_prints_results_and_saves_plots():
    out_dir = os.path.join(tempfile.mkdtemp(), "output")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(output_dir=out_dir, n_sims=2_000)
    text = buffer.getvalue()

    assert "BRENT_CRUDE" in text
    assert "Black-Scholes" in text and "Monte Carlo" in text and "Stress test" in text
    for name in ("mc_convergence_call.png", "mc_convergence_put.png", "stress_test.png", "price_surface.png"):
        assert os.path.getsize(os.path.join(out_dir, name)) > 10_000
