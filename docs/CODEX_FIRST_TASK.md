# First Codex task: theory + implementation audit

Before implementing new features, perform a mathematical and implementation audit of this repository.

The main research specification is:

    docs/QNORMUON_THEORY.md

Do not assume the theory is correct merely because it is written there.

Your task is:

1. Read `QNORMUON_THEORY.md` completely.
2. Read the current QNorMuon implementation and every existing test.
3. Re-derive independently Lemma 1 and Theorems 1–7.
4. For every theorem, classify it as:
   - correct as stated;
   - correct under additional assumptions;
   - incomplete;
   - false.
5. Pay particular attention to:
   - dimensions and transpose conventions;
   - transformation law of Euclidean gradients;
   - the distinction between canonical coordinates and lifted updates;
   - rank-deficient momentum matrices;
   - uniqueness claims;
   - existence/uniqueness of the paired-leverage solution;
   - the full-spark assumption;
   - whether the variational characterization uses the correct matrix after canonicalization;
   - whether approximate polar solvers preserve gauge equivariance;
   - zero or near-zero neuron norms.
6. Build a theorem-to-test table:
   theorem -> assumptions -> implementation -> automated test -> numerical tolerance.
7. Try to falsify each important claim numerically using adversarial random examples.
8. Do not redesign the optimizer yet.

Produce the audit report in:

    docs/THEORY_AUDIT.md

and add missing mathematical tests where appropriate.

All existing tests must continue to pass unless you find that a test encodes a mathematically incorrect claim. In that case, explain the issue before changing it.

## High-value missing tests to consider

- canonical gradients are gauge invariant;
- exact functional invariance of an actual toy SwiGLU block under diagonal gauge;
- `∇Phi(x)` from the leverage formula vs PyTorch autograd vs finite differences;
- Hessian PSD checks for random full-rank instances;
- gauge stress over a large dynamic range;
- rank-deficient and nearly rank-deficient momentum matrices;
- zero / near-zero paired neuron norms;
- float64 reference vs float32 / bfloat16 error measurements.
