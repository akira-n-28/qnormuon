"""Independent checks of paired seed inference and frozen experiment plumbing."""
import copy
import json
from pathlib import Path
import math
import pytest
from benchmarks.multiseed import (SEEDS, configuration, verify_pair,
    validation_metrics, paired_statistics, analyse)


def test_seed_inference_and_missing_seed_are_not_checkpoint_samples():
    result = paired_statistics([-1., -2., -3., -4., -5.])
    assert result["mean"] == -3 and result["sample_sd"] == math.sqrt(2.5)
    assert result["standard_error"] == math.sqrt(.5)
    assert result["ci95"] == pytest.approx([-4.9632431615, -1.0367568385])
    with pytest.raises(ValueError): paired_statistics([-1]*4)


def test_failure_gate_does_not_drop_failed_seed():
    pairs = [dict(seed=s, qso_status="passed", adamw_status="passed",
                  differences=dict(final=-.1, last3=-.1, post_initial_mean=-.1, auc_mean=-.1)) for s in SEEDS]
    assert analyse(pairs)["classification"] == "A"
    pairs[-1]["qso_status"] = "failed"
    assert analyse(pairs)["classification"] == "D" and analyse(pairs)["inference"] is None
    with pytest.raises(ValueError): analyse(pairs[:-1])


def test_fixed_configuration_and_pairing():
    base = json.loads(Path("configs/tiny_transformer/lr_study.json").read_text())
    def provenance(method):
        return dict(seed=2028, config=configuration(base,2028,method), data={"order":"same"},
            initialization_sha256="same", expected_batch_hashes=["same"],
            validation_batch_hashes=["fixed"], parameter_count=11457408, clipping="none")
    a,q = provenance("adamw"),provenance("qso")
    verify_pair(a,q)
    assert a["config"]["data"]["seed"] == 2026 and a["config"]["warmup_steps"] == 51
    for key in ("initialization_sha256", "expected_batch_hashes", "validation_batch_hashes"):
        bad=copy.deepcopy(q);bad[key]="different"
        with pytest.raises(ValueError): verify_pair(a,bad)
    bad=copy.deepcopy(q);bad["config"]["beta"] = .9
    with pytest.raises(ValueError): verify_pair(a,bad)
    with pytest.raises(ValueError): configuration(base,2031,"qso")


def test_common_checkpoint_metric_and_predeclared_gate():
    assert validation_metrics({s:float(s) for s in range(0,513,32)})["last3"] == 480
    with pytest.raises(ValueError): validation_metrics({0:1.,512:0.})
    pairs=[dict(seed=s,qso_status="passed",adamw_status="passed",
        differences=dict(final=-.1,last3=d,post_initial_mean=-.1,auc_mean=-.1))
        for s,d in zip(SEEDS,[-.4,-.3,-.2,-.1,.1])]
    assert analyse(pairs)["classification"] == "B"  # Mean negative, CI crosses zero.
