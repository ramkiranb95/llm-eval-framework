"""
test_semantic_similarity.py
---------------------------
ROUGE evaluation tests for BSFI CRM Auto-Responder outputs.

What ROUGE measures:
    Word and phrase overlap between a generated reply and a reference reply.

    ROUGE-1 : fraction of single words (unigrams) shared
    ROUGE-2 : fraction of two-word phrases (bigrams) shared
    ROUGE-L : longest common subsequence — respects word order

How ROUGE is computed (no framework, no model, no API):
    1. Tokenise candidate and reference into words
    2. Count overlapping words (or bigrams / LCS)
    3. Compute precision, recall, F1 on those counts
    pure Python math — rouge-score library is just the formula implementation

Why ROUGE alongside the LLM judge metrics:
    RAGAs / DeepEval judge semantic correctness via an LLM call.
    ROUGE checks surface-level phrasing drift with zero API cost.
    If ROUGE drops significantly between model versions, the reply phrasing
    has changed noticeably — even if the LLM judge still rates it highly.
    Use ROUGE for regression tracking between runs, not as a hard gate.

Key limitation:
    ROUGE is surface-level only — it cannot detect meaning.
    A paraphrase ("loan declined" vs "application rejected") scores near-zero
    on ROUGE despite being semantically equivalent.
    This is expected behaviour, not a bug.

Thresholds used here:
    ROUGE-L  >= 0.30  (monitor tier — trend tracking, not a hard block)
    ROUGE-1  >= 0.40  (monitor tier)

    These are non-critical — they do not block the release gate.
    They flag cases where phrasing has drifted significantly from the reference.

Important: reference and candidate must be comparable in length.
    ROUGE measures overlap as a fraction of reference length.
    A short reference (7 words) against a full paragraph reply will always
    score near-zero — that is correct behaviour, not a threshold error.
    Ground truth expected_reply should be a full reply, not a summary phrase.

Run:
    pytest tests/test_semantic_similarity.py -v
    pytest tests/test_semantic_similarity.py -v -k "rouge"
"""

import pytest
from rouge_score import rouge_scorer as rouge_scorer_module


# ── Thresholds ────────────────────────────────────────────────────────────────

ROUGE_L_THRESHOLD = 0.30   # monitor tier — trend tracking
ROUGE_1_THRESHOLD = 0.40   # monitor tier


# ── Shared scorer fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rouge_scorer():
    return rouge_scorer_module.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _rouge_scores(candidate: str, reference: str, scorer) -> dict:
    result = scorer.score(reference, candidate)
    return {
        "rouge1_f": round(result["rouge1"].fmeasure, 4),
        "rouge2_f": round(result["rouge2"].fmeasure, 4),
        "rougeL_f": round(result["rougeL"].fmeasure, 4),
    }


# ── ROUGE tests ───────────────────────────────────────────────────────────────

class TestROUGE:
    """
    ROUGE measures word/phrase overlap — surface-level similarity.

    Used as a monitor-tier regression signal alongside LLM judge metrics.
    Not a hard gate — flags phrasing drift between model versions.
    """

    def test_rouge_scores_present_and_valid(self, mock_pipeline_output, rouge_scorer):
        """
        All three ROUGE variants must be returned as floats in [0, 1].
        Smoke test — confirms the scorer is wired correctly.
        """
        candidate = mock_pipeline_output["generated_reply"]
        reference = mock_pipeline_output["ground_truth"]["expected_reply"]

        scores = _rouge_scores(candidate, reference, rouge_scorer)

        print(f"\n  ROUGE scores: {scores}")
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}: {val} out of range [0, 1]"

    def test_rouge_l_meets_monitor_threshold(self, rouge_scorer):
        """
        ROUGE-L monitor-tier check using matched full-length strings.

        Reference and candidate must be comparable in length for ROUGE to be
        meaningful. Short summaries as ground truth produce near-zero scores
        by design — this test uses full-length reply strings to validate the
        threshold is calibrated correctly.
        """
        candidate = (
            "Dear Ramesh, thank you for reaching out. Based on your details, "
            "you are eligible for a personal loan. Please submit your ID proof, "
            "last 3 months salary slips, and bank statements to proceed. "
            "Regards, Customer Support Team"
        )
        reference = (
            "Dear Ramesh Kumar, you are eligible for a personal loan based on your income. "
            "Please provide your ID proof, salary slips for the last 3 months, and bank statements. "
            "Our team will process your application promptly."
        )

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        rouge_l = scores["rougeL_f"]

        print(f"\n  ROUGE-L: {rouge_l} (threshold: {ROUGE_L_THRESHOLD})")
        assert rouge_l >= ROUGE_L_THRESHOLD, (
            f"ROUGE-L {rouge_l} below monitor threshold {ROUGE_L_THRESHOLD}. "
            f"Reply phrasing may have drifted from expected."
        )

    def test_rouge_1_meets_monitor_threshold(self, rouge_scorer):
        """ROUGE-1 monitor-tier check using matched full-length strings."""
        candidate = (
            "Dear Ramesh, thank you for reaching out. Based on your details, "
            "you are eligible for a personal loan. Please submit your ID proof, "
            "last 3 months salary slips, and bank statements to proceed. "
            "Regards, Customer Support Team"
        )
        reference = (
            "Dear Ramesh Kumar, you are eligible for a personal loan based on your income. "
            "Please provide your ID proof, salary slips for the last 3 months, and bank statements. "
            "Our team will process your application promptly."
        )

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        rouge_1 = scores["rouge1_f"]

        print(f"\n  ROUGE-1: {rouge_1} (threshold: {ROUGE_1_THRESHOLD})")
        assert rouge_1 >= ROUGE_1_THRESHOLD, (
            f"ROUGE-1 {rouge_1} below monitor threshold {ROUGE_1_THRESHOLD}."
        )

    def test_rouge_paraphrase_scores_low(self, rouge_scorer):
        """
        A paraphrased reply with the same meaning but different words must score
        LOW on ROUGE. This validates ROUGE's known limitation — it cannot detect
        semantic equivalence, only word overlap.

        This is why the LLM judge (faithfulness, answer_relevance) is the primary
        quality signal. ROUGE is supplementary.
        """
        candidate = "Your application has been declined due to insufficient income documentation."
        reference = "We regret to inform you that your loan request was rejected because salary proof was inadequate."

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        rouge_l = scores["rougeL_f"]

        print(f"\n  ROUGE-L (paraphrase): {rouge_l} — expected LOW")
        assert rouge_l < 0.40, (
            f"ROUGE-L {rouge_l} unexpectedly high for a paraphrase. "
            f"Candidate and reference may be too similar — update strings."
        )

    def test_rouge_wrong_reply_is_low(self, rouge_scorer):
        """
        A reply on a completely different topic must score low on ROUGE.
        ROUGE discriminates well on topic mismatch — this is where it is reliable.
        """
        candidate = "Please visit our nearest branch for gold loan enquiries."
        reference = "Your EMI payment failed. Please retry via UPI or NEFT."

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        rouge_l = scores["rougeL_f"]

        print(f"\n  ROUGE-L (wrong reply): {rouge_l} — expected below {ROUGE_L_THRESHOLD}")
        assert rouge_l < ROUGE_L_THRESHOLD, (
            f"ROUGE-L {rouge_l} should be low for an off-topic reply."
        )

    def test_rouge_out_of_scope_reply(self, mock_oos_pipeline_output, rouge_scorer):
        """
        ROUGE must run without errors on short out-of-scope reply texts.
        Scores are not thresholded here — short texts naturally score near-zero.
        """
        candidate = mock_oos_pipeline_output["generated_reply"]
        reference = mock_oos_pipeline_output["ground_truth"]["expected_reply"]

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        print(f"\n  ROUGE scores (OOS): {scores}")

        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}: {val} out of range [0, 1]"

    def test_rouge_empty_candidate_scores_zero(self, rouge_scorer):
        """
        An empty generated reply must score 0.0 on all ROUGE variants.
        Guards against scorer returning None or crashing on empty input.
        """
        candidate = ""
        reference = "Your EMI payment failed. Please retry via UPI or NEFT."

        scores = _rouge_scores(candidate, reference, rouge_scorer)
        print(f"\n  ROUGE scores (empty candidate): {scores}")

        for key, val in scores.items():
            assert val == 0.0, f"{key}: expected 0.0 for empty candidate, got {val}"

    def test_rouge_identical_strings_score_one(self, rouge_scorer):
        """
        Identical candidate and reference must score 1.0 on all ROUGE variants.
        Validates the upper boundary of the metric.
        """
        text = "Your EMI payment of Rs 5,000 failed. Please retry via UPI or contact support."

        scores = _rouge_scores(text, text, rouge_scorer)
        print(f"\n  ROUGE scores (identical): {scores}")

        for key, val in scores.items():
            assert val == 1.0, f"{key}: expected 1.0 for identical strings, got {val}"
