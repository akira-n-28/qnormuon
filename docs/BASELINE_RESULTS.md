# Verified baseline results

These results were produced from the packaged repository before creating the archive.
They are reference sanity checks, not benchmark claims.

## Unit tests

```text
5 passed
```

## Gauge stress test

```text
gauge dynamic range: 1e-3.93 .. 1e3.79
relative equivariance error, up:   7.953e-16
relative equivariance error, down: 1.145e-15
canonical polar mismatch, up:      1.152e-15
canonical polar mismatch, down:    8.517e-16
```

## Functional trajectory stress test

```text
QNorMuon   | initial=2.225e-16 final=4.824e-16 max=5.499e-16
raw Muon   | initial=2.225e-16 final=1.616e+00 max=3.056e+00
```

Exact floating-point values can vary by PyTorch version, BLAS backend, device and platform. The important reference behavior is that QNorMuon remains near floating-point error under the tested gauge reset, whereas the raw-coordinate spectral baseline does not.
