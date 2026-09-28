# Production-v0 implementation plan and package audit

The complete current package is `qnormuon/core.py` and its exports in
`qnormuon/__init__.py`. `QNorMuon`, `quotient_polar_update`, and
`balance_shared_metric` use independent up/down polars. `gauge_metric` and
weight canonicalization clamp row norms at epsilon. The shared `x` state,
paired-leverage residual, target `2n/m`, and logdet balancing belong to the
historical construction. Thin SVD completion moves at zero momentum. Existing
adversarial tests deliberately record these limitations. Canonical gradients
in that code are transformed once; the double-H mistake is a documented
misinterpretation, not an extra transform currently in `core.py`.

1. Preserve historical code and tests. Add separate production solver and
   optimizer modules, with explicitly named `SwiGLUPair` registration.
2. Port the validated cached-SVD Newton-CG equations to device-aware code,
   retaining float64 centering/certification, original-coordinate warm lambda,
   and explicit CPU float64 reference fallback. No rank inference or truncation.
3. Add stable regular canonicalization, canonical EMA, raw update lifting,
   dtype promotion, named topology checkpoint validation, and diagnostics.
4. Compare against independent reference oracles; exercise gauge, precision,
   zero, failure, checkpoint, and finite-step contracts. Run numerical work
   only in SLURM, then the entire historical and new suite on one H100.
5. Document the API, numerical contract, fallback selection limitation, and
   measured H100 results in `docs/PRODUCTION_V0.md`.

No contradiction was found in the accepted contract. Generic fallback is a
primary-value solve and is explicitly **not** an implementation of the exact
minimum-Frobenius tie-break on every deficient optimal face.
