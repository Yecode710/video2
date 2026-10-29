import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from video2.backend import ModelConfig
from video2.cli import main
from video2.experiment import run_experiment
from video2.video import SampledVideo


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"test fixture; decoding is injected")
        self.dataset = self.root / "dataset.jsonl"
        self.examples = [
            {"id": "one", "video": "clip.mp4", "references": ["A red ball moves."]},
            {"id": "two", "video": "clip.mp4", "references": ["PRIVATE REFERENCE SENTINEL"]},
        ]
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.examples), encoding="utf-8")

    def test_records_partial_failures_without_leaking_references(self):
        prompts = []

        class Backend:
            metadata = {"test_double": True}

            def __init__(self, config):
                self.calls = 0

            def summarize(self, video, prompt):
                prompts.append(prompt)
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("intentional fixture failure")
                return "A red ball moves."

        def sampler(path, **kwargs):
            self.assertEqual(path, self.video)
            self.assertEqual(kwargs["num_frames"], 32)
            return SampledVideo([object(), object()], [0, 1], 2, 1, 1)

        output = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = run_experiment(self.dataset, output, ModelConfig(),
                                     backend_factory=Backend, sampler=sampler)
        self.assertEqual(metrics["counts"]["failed"], 1)
        self.assertEqual(metrics["counts"]["scored"], 1)
        self.assertEqual(metrics["macro_scores"]["rouge1"], 1)
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertNotIn("PRIVATE REFERENCE SENTINEL", prompt)
            self.assertNotIn("A red ball moves.", prompt)
        records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(records[1]["status"], "error")
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["state"], "completed_with_errors")
        self.assertFalse(config["generation"]["do_sample"])
        self.assertIn("dataset_sha256", config)
        self.assertTrue((output / "dataset.jsonl").exists())
        # New runs cannot overwrite previously recorded results.
        with self.assertRaises(FileExistsError):
            run_experiment(self.dataset, output, ModelConfig(), backend_factory=Backend, sampler=sampler)

    def test_model_load_failure_is_persisted_without_fake_predictions(self):
        def failing_backend(config):
            raise RuntimeError("model unavailable")

        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            run_experiment(self.dataset, output, ModelConfig(), backend_factory=failing_backend)
        self.assertEqual(json.loads((output / "config.json").read_text())["state"], "model_load_failed")
        self.assertFalse((output / "predictions.jsonl").exists())
        self.assertFalse((output / "metrics.json").exists())

    def test_offline_cli_evaluation_reports_missing_and_preserves_output(self):
        predictions = self.root / "predictions.jsonl"
        predictions.write_text(json.dumps({"id": "one", "status": "ok", "summary": "A red ball moves."}))
        self.video.unlink()  # Offline scoring must not require video files or model imports.
        output = self.root / "scores.json"
        arguments = ["evaluate", "--dataset", str(self.dataset), "--predictions", str(predictions),
                     "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 1)
            original = output.read_text()
            self.assertEqual(main(arguments), 1)
        self.assertEqual(output.read_text(), original)
        report = json.loads(original)
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["counts"]["scored"], 1)

    def test_invalid_sampling_rejected_before_loading_model(self):
        for num_frames in (1, 3):
            with contextlib.redirect_stderr(io.StringIO()):
                status = main(["run", "--dataset", str(self.dataset), "--output", str(self.root / "unused"),
                               "--num-frames", str(num_frames)])
            self.assertEqual(status, 1)
            self.assertFalse((self.root / "unused").exists())

    def test_empty_predictions_report_all_missing(self):
        predictions = self.root / "empty.jsonl"
        predictions.write_text("")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            status = main(["evaluate", "--dataset", str(self.dataset),
                           "--predictions", str(predictions)])
        self.assertEqual(status, 1)
        report = json.loads(captured.getvalue())
        self.assertEqual(report["counts"]["missing"], 2)
        self.assertIsNone(report["macro_scores"])


if __name__ == "__main__":
    unittest.main()
