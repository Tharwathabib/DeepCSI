import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("samples", [1000, 5000])
def test_splits_are_proportional_not_hardcoded(tmp_path, samples):
    """
    70/10/20 must hold at any --samples, not only at 10000.

    The original code sliced [:7000], [7000:8000], [8000:], which silently
    degraded to 14/2/84 at --samples 50000 -- starving training while making
    the test set look impressively large.
    """
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(root_dir / "data" / "generate_data.py"),
         "--samples", str(samples), "--output-dir", str(out)],
        check=True, capture_output=True, cwd=str(root_dir),
    )

    n = {k: len(np.load(out / f"{k}.npy")) for k in ("train", "val", "test")}
    assert sum(n.values()) == samples
    assert n["train"] == pytest.approx(0.7 * samples, abs=1)
    assert n["val"] == pytest.approx(0.1 * samples, abs=1)
    assert n["test"] == pytest.approx(0.2 * samples, abs=2)

    meta = json.loads((out / "norm_params.json").read_text())
    assert meta["num_train"] == n["train"]
    assert meta["num_val"] == n["val"]
    assert meta["num_test"] == n["test"]
