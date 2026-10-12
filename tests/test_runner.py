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
    assert "Black-Scholes" in text and "Monte Carlo" in text
    for option_type in ("call", "put"):
        assert os.path.getsize(os.path.join(out_dir, f"mc_convergence_{option_type}.png")) > 10_000
