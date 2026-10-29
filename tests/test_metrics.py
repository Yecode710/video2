import unittest

from video2.metrics import score_summary


class SummaryMetricTests(unittest.TestCase):
    def test_identical_english_summary_ignores_case_and_punctuation(self):
        scores = score_summary("A person RUNS, then jumps!", ["a person runs then jumps"])
        self.assertEqual(scores, {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0})

    def test_repeated_tokens_use_clipped_counts(self):
        scores = score_summary("cat cat cat", ["cat cat dog"])
        self.assertAlmostEqual(scores["rouge1"], 2 / 3)
        self.assertAlmostEqual(scores["rouge2"], 0.5)
        self.assertAlmostEqual(scores["rougeL"], 2 / 3)

    def test_lcs_preserves_sequence_order(self):
        scores = score_summary("a b c", ["c b a"])
        self.assertEqual(scores["rouge1"], 1.0)
        self.assertEqual(scores["rouge2"], 0.0)
        self.assertAlmostEqual(scores["rougeL"], 1 / 3)

    def test_multiple_references_are_averaged_not_maximized(self):
        scores = score_summary("a cat runs", ["a cat runs", "the dog sleeps"])
        self.assertEqual(scores, {"rouge1": 0.5, "rouge2": 0.5, "rougeL": 0.5})

    def test_chinese_character_and_mixed_word_tokenization(self):
        scores = score_summary("小猫跳。QWEN 2", ["小猫跳，qwen 2!"], language="zh")
        self.assertEqual(scores, {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0})
        partial = score_summary("猫跑", ["猫跳"], language="zh")
        self.assertEqual(partial, {"rouge1": 0.5, "rouge2": 0.0, "rougeL": 0.5})

    def test_empty_prediction_scores_zero(self):
        self.assertEqual(
            score_summary("", ["a person runs"]),
            {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
        )

    def test_no_bigram_scores_zero(self):
        self.assertEqual(
            score_summary("cat", ["cat"]),
            {"rouge1": 1.0, "rouge2": 0.0, "rougeL": 1.0},
        )

    def test_punctuation_only_scores_zero(self):
        self.assertEqual(
            score_summary("!!!", ["..."]),
            {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
        )

    def test_invalid_references_rejected(self):
        for references in ([], [""], ["  "], ["valid", ""], "a reference", [None]):
            with self.subTest(references=references), self.assertRaises(ValueError):
                score_summary("summary", references)

    def test_invalid_language_rejected(self):
        with self.assertRaises(ValueError):
            score_summary("summary", ["reference"], language="fr")

    def test_non_string_prediction_rejected(self):
        with self.assertRaises(TypeError):
            score_summary(None, ["reference"])


if __name__ == "__main__":
    unittest.main()
