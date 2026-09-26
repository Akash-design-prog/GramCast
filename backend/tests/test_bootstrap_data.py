"""Tests for backend/bootstrap_data.py - the deployment data-fetch logic. Uses monkeypatched paths
(never the real project data/checkpoints dirs) and a real local HTTP-like flow via a mocked requests.get
that streams from a real temp zip file, so the download+extract path is genuinely exercised, not just
mocked into nothing.

Run: python -m pytest backend/tests/test_bootstrap_data.py -v
"""
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap_data as bd


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    training_pairs = root / "data" / "processed" / "training_pairs"
    chirps = root / "data" / "raw" / "chirps"
    checkpoints = root / "ml" / "checkpoints"
    for d in [training_pairs, chirps, checkpoints]:
        d.mkdir(parents=True)

    monkeypatch.setattr(bd, "ROOT", root)
    monkeypatch.setattr(bd, "TRAINING_PAIRS_DIR", training_pairs)
    monkeypatch.setattr(bd, "CHIRPS_DIR", chirps)
    monkeypatch.setattr(bd, "CHECKPOINT_DIR", checkpoints)
    return root, training_pairs, chirps, checkpoints


def _make_real_bundle_zip(path: Path):
    with zipfile.ZipFile(path, "w") as zf:
        for name in bd.REQUIRED_TRAINING_PAIR_FILES:
            zf.writestr(f"training_pairs/{name}", b"fake data for " + name.encode())
        zf.writestr(f"chirps/{bd.REQUIRED_CHIRPS_FILE}", b"fake chirps nc bytes")
        zf.writestr("checkpoints/gramcast_fake_best.pt", b"fake checkpoint bytes")


class _FakeStreamResponse:
    """Mimics requests.get(..., stream=True)'s context-manager response, reading real bytes from a
    local file - so the chunked-download loop in ensure_data_available is genuinely exercised."""

    def __init__(self, path: Path):
        self._path = path

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        with open(self._path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk


def test_is_data_present_false_when_nothing_exists(fake_root):
    assert bd._is_data_present() is False


def test_is_data_present_true_when_everything_exists(fake_root):
    root, training_pairs, chirps, checkpoints = fake_root
    for name in bd.REQUIRED_TRAINING_PAIR_FILES:
        (training_pairs / name).write_bytes(b"x")
    (chirps / bd.REQUIRED_CHIRPS_FILE).write_bytes(b"x")
    (checkpoints / "run_best.pt").write_bytes(b"x")
    assert bd._is_data_present() is True


def test_is_data_present_false_when_only_some_training_pair_files_exist(fake_root):
    root, training_pairs, chirps, checkpoints = fake_root
    (training_pairs / bd.REQUIRED_TRAINING_PAIR_FILES[0]).write_bytes(b"x")
    (chirps / bd.REQUIRED_CHIRPS_FILE).write_bytes(b"x")
    (checkpoints / "run_best.pt").write_bytes(b"x")
    assert bd._is_data_present() is False


def test_is_data_present_false_when_checkpoint_missing(fake_root):
    root, training_pairs, chirps, checkpoints = fake_root
    for name in bd.REQUIRED_TRAINING_PAIR_FILES:
        (training_pairs / name).write_bytes(b"x")
    (chirps / bd.REQUIRED_CHIRPS_FILE).write_bytes(b"x")
    assert bd._is_data_present() is False


def test_ensure_data_available_noop_when_already_present(fake_root, monkeypatch):
    root, training_pairs, chirps, checkpoints = fake_root
    for name in bd.REQUIRED_TRAINING_PAIR_FILES:
        (training_pairs / name).write_bytes(b"x")
    (chirps / bd.REQUIRED_CHIRPS_FILE).write_bytes(b"x")
    (checkpoints / "run_best.pt").write_bytes(b"x")

    with patch("bootstrap_data.requests.get") as mock_get:
        bd.ensure_data_available()
    mock_get.assert_not_called()


def test_ensure_data_available_raises_clearly_when_no_url_and_data_missing(fake_root, monkeypatch):
    monkeypatch.delenv("GRAMCAST_DATA_BUNDLE_URL", raising=False)
    with pytest.raises(RuntimeError, match="GRAMCAST_DATA_BUNDLE_URL is not set"):
        bd.ensure_data_available()


def test_ensure_data_available_downloads_and_extracts_real_zip(fake_root, tmp_path, monkeypatch):
    """The core regression test: build a REAL zip with the documented structure, serve it through a
    mocked requests.get that streams real bytes from disk, and confirm every required file ends up in
    the right place afterward - not just that download was 'called'."""
    monkeypatch.setenv("GRAMCAST_DATA_BUNDLE_URL", "https://example.com/fake-bundle.zip")
    bundle_path = tmp_path / "bundle.zip"
    _make_real_bundle_zip(bundle_path)

    with patch("bootstrap_data.requests.get", return_value=_FakeStreamResponse(bundle_path)):
        bd.ensure_data_available()

    root, training_pairs, chirps, checkpoints = fake_root
    for name in bd.REQUIRED_TRAINING_PAIR_FILES:
        assert (training_pairs / name).exists(), f"{name} not extracted"
        assert (training_pairs / name).read_bytes() == b"fake data for " + name.encode()
    assert (chirps / bd.REQUIRED_CHIRPS_FILE).exists()
    assert (checkpoints / "gramcast_fake_best.pt").exists()
    # the downloaded zip itself must be cleaned up, not left lying around
    assert not (root / "_gramcast_deploy_bundle_download.zip").exists()


def test_ensure_data_available_raises_if_bundle_missing_a_required_file(fake_root, tmp_path, monkeypatch):
    """Test the test: a bundle that's missing one required file must still raise, not silently
    'succeed' with incomplete data."""
    monkeypatch.setenv("GRAMCAST_DATA_BUNDLE_URL", "https://example.com/fake-bundle.zip")
    bundle_path = tmp_path / "incomplete_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        # deliberately omit era5_humidity_wind.npz and the checkpoint
        for name in bd.REQUIRED_TRAINING_PAIR_FILES:
            if name == "era5_humidity_wind.npz":
                continue
            zf.writestr(f"training_pairs/{name}", b"data")
        zf.writestr(f"chirps/{bd.REQUIRED_CHIRPS_FILE}", b"data")

    with patch("bootstrap_data.requests.get", return_value=_FakeStreamResponse(bundle_path)):
        with pytest.raises(RuntimeError, match="still missing"):
            bd.ensure_data_available()
