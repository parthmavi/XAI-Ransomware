# Limitations and future work

Aligned with the two student papers. Nothing here is an experimental claim.

## Current limitations

1. **No verified public ransomware dataset is bundled.** The sample CSV is synthetic and labelled `SYNTHETIC_DEMO`.
2. **Classification unit is a feature row**, not a live process, PE binary, memory dump, or network flow.
3. **No sandbox / dynamic analysis.** Behavioural columns must already exist in the CSV.
4. **Split is stratified-random**, not family-disjoint or temporal. Test scores can be optimistic if related rows leak across splits.
5. **Hyperparameters are not nested-CV tuned.**
6. **SHAP and LIME explain the model**, not causes. LIME can change between runs; correlated features distort attributions. No systematic stability study is included (paper experiment E7).
7. **Not real-time.** Inference and explanation times are measured per batch in evaluation/XAI pages, not against a streaming EDR pipeline.
8. **Concept drift, adversarial evasion, uncertainty quantification, and federated learning** from the review paper are out of scope.
9. **Literature accuracies** (for example 99.08% on BitcoinHeist, 97.85% on a memory-dump study) belong to those authors’ settings. This app must not display them as its own results.

## Future work (from the papers)

- Dataset audit on a verified labelled corpus (experiment E1)
- Grouped / nested cross-validation (E2)
- Family-disjoint and temporal splits vs random split (E3)
- Repeated runs with variability (E4–E5)
- SHAP–LIME agreement and seed stability (E6–E7)
- Cost reporting as a first-class table (E8)
- Pre-encryption / streaming features
- Drift-aware training (FeSAD-style)
- Uncertainty-aware alerts
- Analyst workflow integration (EDR), still without executing malware in this UI
