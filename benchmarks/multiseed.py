"""Predeclared seed-level analysis; no checkpoint pseudo-replication or retuning."""
import copy
import math
import statistics

SEEDS = (2026, 2027, 2028, 2029, 2030)
METHODS = ("adamw", "qso")
T975_DF4 = 2.7764451051977987


def configuration(base, seed, method):
    if seed not in SEEDS or method not in METHODS:
        raise ValueError("only the five declared paired seeds/methods are allowed")
    config = copy.deepcopy(base)
    # data.seed stays 2026: the validation batches stay fixed. Training's
    # SinglePassStream uses the experiment seed independently.
    config.update(seed=seed, steps=512, warmup_steps=51,
                  adam_lr=2e-4 if method == "adamw" else 3e-4,
                  qso_lr=.001 if method == "adamw" else 0.0012247448713915891)
    return config


def verify_pair(a, q):
    for key in ("seed", "data", "initialization_sha256", "expected_batch_hashes",
                "validation_batch_hashes", "parameter_count", "clipping"):
        if a[key] != q[key]:
            raise ValueError("paired provenance differs: " + key)
    ac, qc = copy.deepcopy(a["config"]), copy.deepcopy(q["config"])
    if ac.pop("adam_lr") != 2e-4 or qc.pop("adam_lr") != 3e-4:
        raise ValueError("frozen AdamW rates changed")
    if ac.pop("qso_lr") != .001 or qc.pop("qso_lr") != 0.0012247448713915891 or ac != qc:
        raise ValueError("paired configuration changed")
    if a["seed"] not in SEEDS or a["config"]["seed"] != a["seed"]:
        raise ValueError("undeclared or mismatched seed")


def validation_metrics(values):
    v = {int(k): x for k, x in values.items()}
    if sorted(v) != list(range(0, 513, 32)) or not all(math.isfinite(x) for x in v.values()):
        raise ValueError("complete finite equal-token validation required")
    post = [v[s] for s in range(32, 513, 32)]
    return dict(final=v[512], last3=statistics.mean(v[s] for s in (448, 480, 512)),
                post_initial_mean=statistics.mean(post),
                auc_mean=(v[0]/2 + sum(post[:-1]) + v[512]/2)/16)


def paired_statistics(values):
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError("all five seed differences are required; no subset inference")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd / math.sqrt(5)
    return dict(values=values, mean=mean, median=statistics.median(values), sample_sd=sd,
                standard_error=se, ci95=[mean-T975_DF4*se, mean+T975_DF4*se],
                seeds_favoring_qso=sum(v < 0 for v in values), n=5, df=4)


def analyse(pairs):
    if [p["seed"] for p in pairs] != list(SEEDS):
        raise ValueError("all declared seeds, in order, must be reported")
    failures = [p["seed"] for p in pairs if p["qso_status"] != "passed"]
    if failures:
        return dict(classification="D", failed_qso_seeds=failures,
                    inference=None, reason="planned equal-token comparison prevented by QSO failure")
    if any(p["adamw_status"] != "passed" for p in pairs):
        return dict(classification=None, inference=None, reason="AdamW reference incomplete")
    final = paired_statistics([p["differences"]["final"] for p in pairs])
    last3 = paired_statistics([p["differences"]["last3"] for p in pairs])
    if (last3["mean"] < 0 and final["mean"] < 0 and last3["seeds_favoring_qso"] >= 4
            and last3["ci95"][1] < 0):
        classification = "A"
    elif last3["mean"] < 0 and final["mean"] < 0:
        classification = "B"
    else:
        classification = "C"
    return dict(classification=classification, failed_qso_seeds=[],
                inference=dict(last3=last3, final=final),
                post_initial_mean=paired_statistics([p["differences"]["post_initial_mean"] for p in pairs]),
                auc_mean=paired_statistics([p["differences"]["auc_mean"] for p in pairs]))
