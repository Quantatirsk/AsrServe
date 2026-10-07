"""Sentence restoration must preserve source text and chunk ownership."""

import unittest
from unittest.mock import Mock, patch

from app.services.asr.punctuation import restore_punctuation


class PunctuationTest(unittest.TestCase):
    def restore(self, texts, labels):
        model = Mock()
        model.generate.return_value = [
            {"text": "rewritten text must never be used", "punc_array": labels}
        ]
        with patch(
            "app.services.asr.punctuation.get_punctuation_model", return_value=model
        ):
            output = restore_punctuation(texts)
        return output, model.generate

    def test_restores_internal_sentences_and_the_final_mark(self):
        output, generate = self.restore(["甲乙丙丁戊"], [1, 3, 1, 2, 1])
        self.assertEqual(output, ["甲乙。丙丁，戊。"])
        generate.assert_called_once_with(input="甲 乙 丙 丁 戊")

    def test_shared_context_across_chunks_and_empty_results(self):
        output, generate = self.restore(["甲乙", "", "丙丁"], [1, 1, 3, 1])
        self.assertEqual(output, ["甲乙", "", "丙。丁。"])
        generate.assert_called_once_with(input="甲 乙 丙 丁")

    def test_cross_chunk_boundary_and_existing_prefix_mark(self):
        output, _ = self.restore(["hello", ", world"], [3, 1])
        self.assertEqual(output, ["hello", ". world."])

    def test_existing_terminal_at_chunk_edge_survives(self):
        output, _ = self.restore(["hello.", "", "world"], [2, 1])
        self.assertEqual(output, ["hello.", "", "world."])

    def test_preserves_terminals_and_structural_marks(self):
        output, _ = self.restore(["甲：乙；丙？！"], [3, 3, 3])
        self.assertEqual(output, ["甲：乙；丙？！"])
        output, _ = self.restore(["甲乙。丙丁"], [1, 2, 1, 3])
        self.assertEqual(output, ["甲乙。丙丁。"])

    def test_upgrades_commas_without_double_marks(self):
        output, _ = self.restore(["甲乙，丙丁，"], [1, 3, 1, 2])
        self.assertEqual(output, ["甲乙。丙丁。"])

    def test_preserves_casing_spaces_versions_numbers_and_times(self):
        text = " \tqWen-3 API costs 3.14 and 12.5% at 12:30\n"
        output, generate = self.restore([text], [1, 1, 1, 2, 1, 2, 1, 1])
        self.assertEqual(output, [" \tqWen-3 API costs 3.14, and 12.5%, at 12:30.\n"])
        generate.assert_called_once_with(
            input="qWen-3 API costs 3.14 and 12.5% at 12:30"
        )

    def test_preserves_questions_exclamations_digit_grouping(self):
        text = "Is GPT-4 ready? Yes! 1,234.50 at 12:30."
        output, _ = self.restore([text], [1, 1, 3, 3, 1, 1, 1])
        self.assertEqual(output, [text])

    def test_never_uses_reformatted_english_output(self):
        output, _ = self.restore(["oh yeah he said hello"], [1, 3, 1, 1, 1])
        self.assertEqual(output, ["oh yeah. he said hello."])

    def test_quotes_contractions_and_trailing_whitespace(self):
        output, _ = self.restore(["他说“你好” \n"], [1, 1, 1, 1])
        self.assertEqual(output, ["他说“你好。” \n"])
        output, _ = self.restore(['He said "it\'s fine"'], [1, 1, 1, 1])
        self.assertEqual(output, ['He said "it\'s fine."'])

    def test_accented_words_and_combining_marks_cannot_be_split(self):
        output, _ = self.restore(["café déjà vu"], [3] * 7)
        self.assertEqual(output, ["café. déjà. vu."])
        output, _ = self.restore(["Cafe\u0301"], [3, 1])
        self.assertEqual(output, ["Cafe\u0301."])

    def test_typographic_contraction_cannot_be_split(self):
        output, _ = self.restore(["it’s fine"], [3, 3, 1])
        self.assertEqual(output, ["it’s. fine."])

    def test_urls_are_not_split_or_rewritten(self):
        text = "Visit https://example.com/v1.2?q=3.14&n=1,234 now"
        output, generate = self.restore([text], [1, 2, 1])
        self.assertEqual(output, [text.replace(" now", ", now") + "."])
        generate.assert_called_once_with(input=text)

    def test_empty_and_punctuation_only_inputs_skip_the_model(self):
        with patch("app.services.asr.punctuation.get_punctuation_model") as model:
            texts = ["", " \n", "。？！", "😀"]
            self.assertEqual(restore_punctuation(texts), texts)
            self.assertEqual(restore_punctuation([]), [])
            model.assert_not_called()

    def test_invalid_labels_and_model_errors_fail_the_request(self):
        model = Mock()
        with patch(
            "app.services.asr.punctuation.get_punctuation_model", return_value=model
        ):
            for response in (
                [],
                [{}],
                [{"punc_array": [1]}],
                [{"punc_array": [1, 99]}],
                [{"punc_array": [1, "3"]}],
            ):
                model.generate.return_value = response
                with self.subTest(response=response), self.assertRaises(RuntimeError):
                    restore_punctuation(["甲乙"])
            model.generate.side_effect = ValueError("inference failed")
            with self.assertRaisesRegex(ValueError, "inference failed"):
                restore_punctuation(["甲乙"])


if __name__ == "__main__":
    unittest.main()
