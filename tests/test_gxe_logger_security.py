from __future__ import annotations

import os

import pytest

from summit.logger import Logger


def test_logger_creates_private_file_and_rejects_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("untouched\n", encoding="utf-8")
    link = tmp_path / "run.gxe.log"
    link.symlink_to(target)

    logger = Logger(suppress=True)
    with pytest.raises(OSError):
        logger.attach_file(str(link), mode="w")
    assert target.read_text(encoding="utf-8") == "untouched\n"

    link.unlink()
    logger.attach_file(str(link), mode="a")
    logger._log("private")
    logger.close()
    assert (link.stat().st_mode & 0o777) == 0o600
    assert link.read_text(encoding="utf-8") == "private\n"


def test_logger_rejects_multiply_linked_target_without_modifying_it(tmp_path):
    original = tmp_path / "original.log"
    original.write_text("preserve\n", encoding="utf-8")
    linked = tmp_path / "run.gxe.log"
    os.link(original, linked)

    logger = Logger(suppress=True)
    with pytest.raises(PermissionError, match="singly linked"):
        logger.attach_file(str(linked), mode="w")
    assert original.read_text(encoding="utf-8") == "preserve\n"
