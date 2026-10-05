"""Small deployment checks without Docker, models, imagery or network access."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy import healthcheck, serve


class ContainerDeploymentTests(unittest.TestCase):
    def test_default_port(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(serve.configured_port(), 8000)

    def test_custom_port(self) -> None:
        with patch.dict("os.environ", {"PORT": "8123"}):
            self.assertEqual(serve.configured_port(), 8123)

    def test_invalid_port_cannot_be_interpolated_into_commands(self) -> None:
        for value in ("0", "65536", "invalid", "8000;echo unsafe"):
            with self.subTest(value=value), patch.dict("os.environ", {"PORT": value}):
                with self.assertRaises(ValueError):
                    serve.configured_port()

    def test_startup_uses_one_worker_and_container_bind_address(self) -> None:
        with patch.dict("os.environ", {"PORT": "8123"}), patch.object(serve.uvicorn, "run") as run:
            serve.main()
        run.assert_called_once_with("src.api_service:app", host="0.0.0.0", port=8123, workers=1)

    def test_healthcheck_probes_liveness_not_model_readiness(self) -> None:
        with patch.dict("os.environ", {"PORT": "8123"}), patch.object(
            healthcheck, "urlopen", return_value=nullcontext(SimpleNamespace(status=200))
        ) as request:
            self.assertEqual(healthcheck.main(), 0)
        request.assert_called_once_with("http://127.0.0.1:8123/health/live", timeout=3)

    def test_failed_healthcheck_returns_nonzero(self) -> None:
        with patch.object(healthcheck, "urlopen", side_effect=OSError("unavailable")):
            self.assertEqual(healthcheck.main(), 1)

    def test_invalid_healthcheck_port_makes_no_request(self) -> None:
        with patch.dict("os.environ", {"PORT": "bad"}), patch.object(healthcheck, "urlopen") as request:
            self.assertEqual(healthcheck.main(), 1)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
