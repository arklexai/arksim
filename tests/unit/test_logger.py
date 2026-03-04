# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.utils.logger.logging."""

import logging
import os

from arksim.utils.logger.logging import add_file_handler, get_logger


class TestAddFileHandler:
    def test_creates_file_handler(self, temp_dir: str) -> None:
        log_file = os.path.join(temp_dir, "test.log")
        lg = logging.getLogger("test_add_file_handler")
        add_file_handler(lg, log_file)

        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == os.path.abspath(log_file)

        # cleanup
        for h in file_handlers:
            h.close()
            lg.removeHandler(h)

    def test_creates_directory(self, temp_dir: str) -> None:
        log_file = os.path.join(temp_dir, "subdir", "nested", "test.log")
        lg = logging.getLogger("test_add_file_handler_dir")
        add_file_handler(lg, log_file)

        assert os.path.isdir(os.path.join(temp_dir, "subdir", "nested"))

        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                lg.removeHandler(h)

    def test_file_in_cwd_no_dir(self) -> None:
        """When log_file has no directory component, skip makedirs."""
        lg = logging.getLogger("test_add_file_handler_cwd")
        # log_dir will be "" so makedirs is skipped
        add_file_handler(lg, "bare.log")

        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

        for h in file_handlers:
            h.close()
            lg.removeHandler(h)

        # cleanup the file if created
        if os.path.exists("bare.log"):
            os.remove("bare.log")


class TestGetLoggerWithFile:
    def test_log_file_adds_file_handler(self, temp_dir: str) -> None:
        log_file = os.path.join(temp_dir, "get_logger.log")
        lg = get_logger("test_get_logger_file", log_file=log_file)

        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
            lg.removeHandler(h)

    def test_duplicate_file_handler_skipped(self, temp_dir: str) -> None:
        log_file = os.path.join(temp_dir, "dedup.log")
        name = "test_get_logger_dedup"
        lg = get_logger(name, log_file=log_file)
        lg2 = get_logger(name, log_file=log_file)

        assert lg is lg2
        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
            lg.removeHandler(h)
