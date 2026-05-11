# Evaluation Coverage — LLM Evaluation Framework

## Coverage Overview

This document maps all 15 evaluation parameter categories.
Tier 1 covers a focused subset. All 15 categories are planned across tiers.

---

## Active Metrics — Tier 1 MVP

### RAGAs Metrics

| Metric | Threshold | Critical | Description |
|---|---|---|---|
| faithfulness | 0.85 | Yes | Every claim in reply is traceable to retrieved context |
| answer_relevance | 0.80 | Yes | Reply actually addresses what the customer asked |
| context_precision | 0.75 | No | Retrieved chunks are relevant, minimal noise |
| context_recall | 0.75 | No | All necessary context was retrieved |

### DeepEval Metrics

| Metric | Threshold | Critical | Description |
|---|---|---|---|
| hallucination | ≤ 0.20 | Yes | Reply does not contain fabricated facts |
| answer_correctness | 0.65 | Yes | Reply matches ground truth expectation |
| coherence | 0.80 | No | Reply is internally logical and well-structured |
| tone_professionalism | 0.80 | No | Reply maintains appropriate professional tone |
| toxicity | ≤ 0.10 | Yes (inverted) | Reply is free of harmful or offensive content |
| non_advice | 0.80 | Yes | Reply avoids giving specific financial/legal advice |
| topic_adherence | 0.80 | No | Reply stays on the topic raised by the customer |
| bias | 0.90 | Yes | Reply is free of demographic or regulatory bias |
| role_adherence | 0.80 | No | Reply stays within the defined CRM agent persona |

### BSFI Custom LLM Metrics

| Metric | Threshold | Critical | Description |
|---|---|---|---|
| pii_leakage | 0.90 | Yes | Reply does not expose personally identifiable information |
| answer_similarity | 0.60 | No | Reply is semantically similar to the ground truth reply |

### Custom Domain Metrics (Deterministic)

| Metric | Threshold | Critical | Description |
|---|---|---|---|
| ticket_status_accuracy | 1.0 | Yes | Ticket status transition is correct |
| escalation_logic | 1.0 | Yes | Escalation flag matches expectation |
| key_facts_coverage | 0.75 | No | Reply covers key facts from ground truth |
| out_of_scope_handling | 1.0 | Yes | Out-of-scope queries handled without hallucination |
| restricted_words | 1.0 | Yes | Reply contains no RBI/SEBI-prohibited phrases |
| language_check | 1.0 | Yes | Reply is in English (ASCII alpha ratio check) |

---

## All 15 Categories — Full Coverage Map

### Category 1 — Output Quality
`factual_accuracy` `hallucination` `coherence` `relevance`
`completeness` `conciseness` `domain_accuracy`

**Tier 1:** hallucination, coherence
**Tier 2:** factual_accuracy, completeness, domain_accuracy

---

### Category 2 — Faithfulness (RAG)
`answer_faithfulness` `context_precision` `context_recall`
`citation_accuracy` `empty_context_behaviour` `context_poisoning_resistance`

**Tier 1:** answer_faithfulness, context_precision, context_recall
**Tier 2:** citation_accuracy, empty_context_behaviour, context_poisoning_resistance

---

### Category 3 — Consistency and Reliability
`run_to_run_consistency` `paraphrase_consistency` `reversal_consistency`
`positional_consistency` `sycophancy_resistance` `self_contradiction`

**Tier 1:** None (not in scope)
**Tier 2:** run_to_run_consistency, paraphrase_consistency
**Tier 3:** reversal_consistency, sycophancy_resistance

---

### Category 4 — Instruction Following
`format_adherence` `constraint_following` `multi_constraint_handling`
`instruction_drift` `persona_maintenance`

**Tier 1:** Partially covered via answer_correctness
**Tier 2:** format_adherence, constraint_following

---

### Category 5 — Reasoning Quality
`multi_step_reasoning` `causal_reasoning` `temporal_reasoning`
`planning_sequencing` `counterfactual_reasoning`

**Tier 1:** None (not in scope)
**Tier 3:** multi_step_reasoning, causal_reasoning

---

### Category 6 — Safety and Adversarial
`toxic_content` `jailbreak_resistance` `prompt_injection_resistance`
`over_refusal` `harmful_information`

**Tier 1:** Partially via out_of_scope_handling
**Tier 2:** jailbreak_resistance, prompt_injection_resistance
**Tier 3:** toxic_content, over_refusal

---

### Category 7 — Bias and Fairness
`demographic_bias` `gender_bias` `sycophancy_bias`
`geographic_bias` `regulatory_fairness`

**Tier 1:** None
**Tier 3:** regulatory_fairness, sycophancy_bias

---

### Category 8 — Privacy and Security
`pii_leakage` `system_prompt_leakage` `cross_document_exposure`
`training_data_memorisation`

**Tier 1:** None
**Tier 3:** pii_leakage, system_prompt_leakage

---

### Category 9 — Domain Specific (BSFI)
`regulatory_language_precision` `tone_calibration` `escalation_logic`
`ticket_status_accuracy` `audit_traceability` `confidence_signalling`
`cross_lingual_handling` `multi_intent_handling`

**Tier 1:** escalation_logic, ticket_status_accuracy
**Tier 2:** tone_calibration, regulatory_language_precision, cross_lingual_handling
**Tier 3:** audit_traceability, multi_intent_handling

---

### Category 10 — Performance
`latency` `token_efficiency` `context_window_behaviour` `temperature_sensitivity`

**Tier 1:** None
**Tier 3:** latency, token_efficiency

---

### Category 11 — LLM Syndromes
`confabulation` `sycophancy` `reversal_curse` `context_blindness`
`instruction_amnesia` `overconfident_refusal` `role_collapse`
`format_regression` `length_anchoring` `voldemort_syndrome`

**Tier 1:** confabulation (via hallucination metric)
**Tier 2:** sycophancy, context_blindness
**Tier 3:** Full syndrome test suite

See `llmSyndromes.md` for detailed descriptions.

---

### Category 12 — Agentic Parameters
`tool_selection_accuracy` `tool_call_correctness` `plan_coherence`
`error_recovery` `cascade_failure_handling` `loop_detection`
`goal_drift` `autonomy_calibration`

**Tier 1:** None
**Tier 3:** Agentic extension of the framework

---

### Category 13 — Knowledge and Calibration
`knowledge_cutoff_awareness` `uncertainty_expression`
`overconfidence` `underconfidence` `i_dont_know_calibration`

**Tier 1:** Partially via out_of_scope_handling (TC010)
**Tier 2:** uncertainty_expression, i_dont_know_calibration

---

### Category 14 — Context Handling
`long_context_retention` `multi_turn_memory` `context_utilisation`
`conflicting_context_handling` `empty_context_behaviour`

**Tier 1:** context_utilisation (via RAGAs)
**Tier 2:** conflicting_context_handling, empty_context_behaviour

---

### Category 15 — Ethics and Compliance
`transparency` `explainability` `regulatory_compliance`
`environmental_social_harm`

**Tier 1:** regulatory_compliance (via BSFI domain rules)
**Tier 3:** Full ethics layer

---

## Coverage Summary by Tier

| Tier | Categories Touched | Metrics Active |
|---|---|---|
| Tier 1 (MVP) | 1, 2, 4, 6, 9, 11, 13, 14, 15 | 21 (6 deterministic + 15 LLM) |
| Tier 2 | Adds 3, 7, 8, 12 | ~30 |
| Tier 3 | All 15 categories | Full suite |

---

## Release Gate Logic

```
Policy: all_critical_must_pass

A release is blocked if ANY critical metric falls below its threshold.
Non-critical failures are logged but do not block release.

Critical metrics (Tier 1):
  - faithfulness ≥ 0.75
  - answer_relevance ≥ 0.80
  - hallucination ≤ 0.20 (inverted)
  - answer_correctness ≥ 0.65
  - toxicity ≤ 0.10 (inverted)
  - non_advice ≥ 0.80
  - bias ≥ 0.90
  - pii_leakage ≥ 0.90
  - ticket_status_accuracy = 1.0
  - escalation_logic = 1.0
  - out_of_scope_handling = 1.0
  - restricted_words = 1.0
  - language_check = 1.0
```
