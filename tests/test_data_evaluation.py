import json
from pathlib import Path
import tempfile
import unittest

from video2.data import Example, load_dataset, read_jsonl
from video2.evaluation import evaluate


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "data.jsonl"

    def write_rows(self, rows):
        self.manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_relative_video_paths_resolve_against_manifest_directory(self):
        video = self.root / "clips" / "example.mp4"
        video.parent.mkdir()
        video.touch()
        self.write_rows([
            {"id": "clip", "video": "clips/example.mp4", "references": ["A person runs."]}
        ])
        examples = load_dataset(self.manifest)
        self.assertEqual(examples, [Example("clip", video.resolve(), ["A person runs."])])

    def test_reference_text_cannot_be_a_bare_string(self):
        self.write_rows([{"id": "clip", "video": "missing.mp4", "references": "A cat."}])
        with self.assertRaisesRegex(ValueError, "references must be a list"):
            load_dataset(self.manifest, check_videos=False)

    def test_blank_and_non_string_references_are_rejected(self):
        for references in ([""], [" "], ["valid", "\t"], [None], [42], None):
            with self.subTest(references=references):
                self.write_rows([
                    {"id": "clip", "video": "missing.mp4", "references": references}
                ])
                with self.assertRaisesRegex(ValueError, "references must be a list"):
                    load_dataset(self.manifest, check_videos=False)

    def test_duplicate_ids_are_rejected(self):
        self.write_rows([
            {"id": "same", "video": "one.mp4"},
            {"id": "same", "video": "two.mp4"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            load_dataset(self.manifest, check_videos=False)

    def test_absent_and_empty_references_are_valid(self):
        self.write_rows([
            {"id": "one", "video": "one.mp4"},
            {"id": "two", "video": "two.mp4", "references": []},
        ])
        self.assertEqual(
            [example.references for example in load_dataset(self.manifest, check_videos=False)],
            [[], []],
        )

    def test_video_existence_check_can_be_disabled_for_offline_evaluation(self):
        self.write_rows([{"id": "clip", "video": "missing.mp4"}])
        with self.assertRaisesRegex(ValueError, "video not found"):
            load_dataset(self.manifest)
        self.assertEqual(len(load_dataset(self.manifest, check_videos=False)), 1)

    def test_invalid_or_missing_id_and_video_are_rejected(self):
        for key in ("id", "video"):
            for value in (None, "", " ", 1):
                with self.subTest(key=key, value=value):
                    row = {"id": "clip", "video": "clip.mp4"}
                    row[key] = value
                    self.write_rows([row])
                    with self.assertRaises(ValueError):
                        load_dataset(self.manifest, check_videos=False)

    def test_empty_non_object_and_invalid_json_are_rejected(self):
        for content in ("\n", "[]\n", "{invalid}\n"):
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_jsonl(self.manifest)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def example(identifier, references=None):
        return Example(identifier, Path(f"{identifier}.mp4"), references or [])

    def test_partial_run_counts_success_failure_missing_and_unreferenced(self):
        examples = [
            self.example("scored", ["a cat runs"]),
            self.example("failed", ["a dog sleeps"]),
            self.example("missing", ["a child plays"]),
            self.example("no-reference"),
        ]
        predictions = [
            {"id": "no-reference", "status": "ok", "summary": "A person walks."},
            {"id": "failed", "status": "error", "error": "decoder failed"},
            {"id": "scored", "status": "ok", "summary": "a cat runs"},
        ]
        result = evaluate(examples, predictions, "en")
        self.assertEqual(result["counts"], {
            "total": 4, "successful": 2, "failed": 1, "missing": 1,
            "without_references": 1, "scored": 1,
        })
        self.assertEqual(result["macro_scores"], {
            "rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0,
        })
        self.assertEqual([row["id"] for row in result["per_sample"]], ["scored"])

    def test_predictions_align_by_id_and_macro_average_videos(self):
        examples = [
            self.example("one", ["a cat runs"]),
            self.example("two", ["a dog sleeps"]),
        ]
        predictions = [
            {"id": "two", "status": "ok", "summary": "people walk outside"},
            {"id": "one", "status": "ok", "summary": "a cat runs"},
        ]
        result = evaluate(examples, predictions, "en")
        self.assertEqual([row["id"] for row in result["per_sample"]], ["one", "two"])
        self.assertEqual(result["macro_scores"], {
            "rouge1": 0.5, "rouge2": 0.5, "rougeL": 0.5,
        })

    def test_no_references_yields_no_macro_instead_of_zero(self):
        result = evaluate(
            [self.example("clip")],
            [{"id": "clip", "status": "ok", "summary": "A cat runs."}],
            "en",
        )
        self.assertIsNone(result["macro_scores"])
        self.assertEqual(result["per_sample"], [])
        self.assertEqual(result["counts"]["without_references"], 1)
        self.assertEqual(result["counts"]["scored"], 0)
        self.assertEqual(result["counts"]["successful"], 1)

    def test_failed_and_missing_are_not_scored_as_zero(self):
        examples = [self.example("failed", ["a cat runs"]), self.example("missing", ["a dog sleeps"])]
        result = evaluate(examples, [{"id": "failed", "status": "error"}], "en")
        self.assertIsNone(result["macro_scores"])
        self.assertEqual(result["per_sample"], [])
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["missing"], 1)
        self.assertEqual(result["counts"]["scored"], 0)

    def test_unknown_and_duplicate_prediction_ids_are_rejected(self):
        examples = [self.example("clip", ["a cat runs"])]
        row = {"id": "clip", "status": "ok", "summary": "a cat runs"}
        with self.assertRaisesRegex(ValueError, "Unknown prediction id"):
            evaluate(examples, [{**row, "id": "other"}], "en")
        with self.assertRaisesRegex(ValueError, "Duplicate prediction id"):
            evaluate(examples, [row, row], "en")

    def test_missing_or_invalid_status_and_summary_are_rejected(self):
        examples = [self.example("clip", ["a cat runs"])]
        invalid_rows = [
            {"id": "clip", "summary": "a cat runs"},
            {"id": "clip", "status": "skipped"},
            {"id": "clip", "status": "ok"},
            {"id": "clip", "status": "ok", "summary": "  "},
            {"id": "clip", "status": "ok", "summary": 123},
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate(examples, [row], "en")


if __name__ == "__main__":
    unittest.main()
