import subprocess
import sys
import time
import unittest
from unittest.mock import Mock

from cloud.runtime import checked_process


class ProcessTests(unittest.TestCase):
    def test_success_commits(self):
        commit = Mock()
        checked_process([sys.executable, "-c", "pass"],
                        deadline=time.monotonic() + 10, commit=commit)
        commit.assert_called()

    def test_error_is_not_reported_as_success(self):
        commit = Mock()
        with self.assertRaises(subprocess.CalledProcessError):
            checked_process([sys.executable, "-c", "raise SystemExit(7)"],
                            deadline=time.monotonic() + 10, commit=commit)
        commit.assert_called()

    def test_timeout_stops_process_and_commits(self):
        commit = Mock()
        with self.assertRaises(TimeoutError):
            checked_process([sys.executable, "-c", "import time; time.sleep(60)"],
                            deadline=time.monotonic() + .1, commit=commit, interval=.05)
        self.assertGreaterEqual(commit.call_count, 2)
