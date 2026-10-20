"""Controlli locali dell'adattatore: nessuna chiamata al servizio Modal."""
import contextlib
import importlib.util
import io
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("modal"), "SDK Modal opzionale non installato")
class ModalPlanTests(unittest.TestCase):
    def test_plan_never_launches_remote_job(self):
        import modal_app
        with patch.object(modal_app, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            modal_app.main(action="plan")
        execute.remote.assert_not_called()

    def test_real_run_requires_validation_or_explicit_exploratory(self):
        import modal_app
        with patch.object(modal_app, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "validation"):
                modal_app.main(action="run")
        execute.remote.assert_not_called()

    def test_output_must_be_persistent(self):
        import modal_app
        with patch.object(modal_app, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "output"):
                modal_app.main(action="run", output="/tmp/lost-results", exploratory=True)
        execute.remote.assert_not_called()
