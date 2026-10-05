import shutil
import subprocess

import numpy as np
import pytest

from viralsense.data.splitzip import LocalSplitZip, read_central_directory


@pytest.mark.skipif(shutil.which("zip") is None, reason="zip CLI not available")
def test_split_zip_members_read_back_exactly(tmp_path):
    src = tmp_path / "info"
    src.mkdir()
    rng = np.random.default_rng(0)
    files = {}
    for i in range(60):  # mix of compressible text and incompressible bytes, so members straddle parts
        data = (f'{{"id": {i}, "text": "' + "caption " * int(rng.integers(50, 4000)) + '"}').encode() if i % 2 \
            else rng.bytes(int(rng.integers(1000, 90_000)))
        (src / f"acct-{i}.info").write_bytes(data)
        files[f"info/acct-{i}.info"] = data
    subprocess.run(["zip", "-q", "-r", "-s", "128k", "posts_info.zip", "info"], cwd=tmp_path, check=True)
    parts = sorted(tmp_path.glob("posts_info.z*"))
    assert len(parts) > 3

    index, eocd = read_central_directory(tmp_path / "posts_info.zip")
    z = LocalSplitZip(tmp_path, "posts_info", index)
    assert z.n_parts == len(parts)
    straddle = 0
    for name, data in files.items():
        assert z.read(name) == data
        r = index.set_index("name").loc[name]
        straddle += int(r.offset + r.csize > (tmp_path / z.part_path(int(r.disk)).name).stat().st_size)
    assert straddle > 0  # at least one member crosses a part boundary
    with pytest.raises(KeyError):
        z.read("info/missing.info")
    z.close()
