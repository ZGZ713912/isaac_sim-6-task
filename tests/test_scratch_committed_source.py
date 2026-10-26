"""A training launch must authenticate actual committed bytes before Isaac."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scratch_entry_source_guard", ROOT / "scripts/train_scratch_standing.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = root / "actor.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "actor.py"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Source test", "-c", "user.email=test@localhost",
                    "commit", "-qm", "source"], cwd=root, check=True)
    output = tmp_path / "output"
    output.mkdir()
    return root, source, output


def test_archive_is_exact_committed_source(repository):
    root, source, output = repository
    hashes = {"actor.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    receipt = entry.archive_committed_sources(hashes, output, root=root)
    assert receipt["all_source_bytes_match_commit"] and receipt["source_count"] == 1
    assert receipt["archive_sha256"] == hashlib.sha256(Path(receipt["archive"]).read_bytes()).hexdigest()
    with tarfile.open(receipt["archive"]) as archive:
        assert archive.extractfile("actor.py").read() == source.read_bytes()


@pytest.mark.parametrize("fault", ["modified", "untracked", "stale_snapshot"])
def test_noncommitted_or_changed_source_fails_without_writing_archive(repository, fault):
    root, source, output = repository
    old = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("value = 2\n")
    name = "actor.py"
    if fault == "untracked":
        name = "new_actor.py"
        (root / name).write_bytes(source.read_bytes())
    hashes = {name: old if fault == "stale_snapshot" else hashlib.sha256((root / name).read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="requires committed source"):
        entry.archive_committed_sources(hashes, output, root=root)
    assert not (output / "committed_source.tar").exists()
