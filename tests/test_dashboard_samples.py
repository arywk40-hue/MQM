from __future__ import annotations

from dashboard import app


def test_local_sample_images_exclude_internal_zone_fixtures(tmp_path, monkeypatch):
    (tmp_path / "queue.jpg").write_bytes(b"image")
    (tmp_path / "another.PNG").write_bytes(b"image")
    (tmp_path / "mess_hall_dense.jpg").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("not an image")
    monkeypatch.setattr(app, "LOCAL_SAMPLE_DIR", tmp_path)
    assert [path.name for path in app.local_sample_images()] == ["another.PNG", "queue.jpg"]
