# LLM Syndromes — Bug Taxonomy for Language Models

## What Are LLM Syndromes?

LLM Syndromes are named, observable, repeatable failure patterns in language models.
Coined and documented by James Bach and Michael Bolton as part of extending
Rapid Software Testing (RST) thinking to AI systems.

Unlike traditional software bugs (which are deterministic), LLM syndromes are
**probabilistic tendencies** — the model is statistically biased toward a
failure mode under certain conditions.

This taxonomy is used to design adversarial test cases in Tier 2 and beyond.

---

## The 10 Core Syndromes

### 1. Confabulation (Hallucination)
**What it is:** The model generates plausible-sounding but factually incorrect
information, presented with confidence.

**In BSFI context:** Stating an interest rate, penalty clause, or RBI guideline
that does not exist in the retrieved context.

**Detection metric:** `hallucination` (DeepEval), `faithfulness` (RAGAs)

**Test signal:** Reply contains facts not present in `retrieved_context`

---

### 2. Sycophancy
**What it is:** The model agrees with the user's stated position even when
the user is wrong. Prioritises approval over accuracy.

**In BSFI context:** Customer claims their interest rate should be 18%, model
agrees without checking the context — even if context says 22% is correct.

**Detection metric:** `sycophancy_resistance` (Tier 2)

**Test signal:** Introduce incorrect customer claim → model should correct,
not validate

---

### 3. Reversal Curse
**What it is:** The model fails to reverse a learned relationship.
If trained on "A → B", it cannot reliably derive "B → A".

**In BSFI context:** Can state "foreclosure after 12 months has 2% charge"
but cannot reliably answer "when does foreclosure become free?"

**Detection metric:** `reversal_consistency` (Tier 2)

**Test signal:** Ask the same question in forward and reverse form →
compare answers for logical consistency

---

### 4. Context Blindness
**What it is:** The model ignores provided context and answers from
training data instead.

**In BSFI context:** Retrieved context says processing time is 5 days,
but model responds with a different number from its training data.

**Detection metric:** `faithfulness`, `context_utilisation`

**Test signal:** Provide context that contradicts training knowledge →
model should follow context, not training

---

### 5. Instruction Amnesia
**What it is:** The model forgets instructions given earlier in the prompt
as the conversation or context grows longer.

**In BSFI context:** System prompt says "do not provide investment advice"
but after a long email thread, the model starts offering investment tips.

**Detection metric:** `instruction_drift` (Tier 2)

**Test signal:** Long-context multi-turn tests → check if system prompt
constraints are maintained throughout

---

### 6. Overconfident Refusal
**What it is:** The model refuses to answer a legitimate question, citing
safety or scope concerns that do not apply.

**In BSFI context:** Customer asks about their loan balance; model refuses
claiming it "cannot access account information" even when context is provided.

**Detection metric:** `over_refusal` (Tier 2)

**Test signal:** Legitimate in-scope queries → model should answer,
not refuse

---

### 7. Role Collapse
**What it is:** The model breaks character or persona under pressure —
either from adversarial prompting or complex multi-turn context.

**In BSFI context:** System prompt defines the model as a BSFI CRM agent.
After jailbreak attempts or confusing context, model stops behaving as an agent.

**Detection metric:** `persona_maintenance`, `jailbreak_resistance`

**Test signal:** Adversarial prompts designed to break the CRM agent role

---

### 8. Format Regression
**What it is:** The model reverts to a generic or inconsistent output format
when the input is ambiguous or complex, ignoring format instructions.

**In BSFI context:** System prompt specifies a structured reply format.
For complex grievance emails, model produces unstructured output.

**Detection metric:** `format_adherence` (Tier 2)

**Test signal:** Complex input emails → verify output structure remains consistent

---

### 9. Length Anchoring
**What it is:** The model calibrates response length to the input length
rather than the task complexity. Long inputs get long outputs regardless
of what the question requires.

**In BSFI context:** A simple "what is the processing time?" question
embedded in a long email triggers a full-page response.

**Detection metric:** `conciseness`, `token_efficiency`

**Test signal:** Vary input length for the same core question →
check if output length varies inappropriately

---

### 10. Voldemort Syndrome
**What it is:** The model avoids naming a topic, entity, or answer
even when it is clearly the correct response — dancing around it
without committing.

**In BSFI context:** Customer asks if they are eligible for a loan.
Model describes eligibility criteria without ever saying "yes, you are eligible."

**Detection metric:** `completeness`, `answer_correctness`

**Test signal:** Questions requiring a direct yes/no or named answer →
check if model commits to a clear response

---

## Usage in Test Design

| Syndrome | Tier | Test Case Type |
|---|---|---|
| Confabulation | Tier 1 | All 10 test cases check for this |
| Sycophancy | Tier 2 | Adversarial: incorrect customer claim |
| Reversal Curse | Tier 2 | Same question, reversed form |
| Context Blindness | Tier 1 | Context vs training data conflict |
| Instruction Amnesia | Tier 2 | Long-context / multi-turn |
| Overconfident Refusal | Tier 2 | Legitimate in-scope queries |
| Role Collapse | Tier 2 | Jailbreak attempts |
| Format Regression | Tier 2 | Complex input, format check |
| Length Anchoring | Tier 3 | Input length variation |
| Voldemort Syndrome | Tier 2 | Direct-answer questions |

---

## References

- James Bach — [Rapid Software Testing](https://www.satisfice.com/rapid-software-testing)
- Michael Bolton — [DevelopSense Blog](https://developsense.com/blog)
- LLM Syndromes originated in RST community discussions on AI system testing
