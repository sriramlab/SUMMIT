"""Qualify identical whole-marker public jobs using the private native launcher."""
import argparse, json, time, resource
from pathlib import Path
from summit.epistasis.cli import main as cli, _jsonable
from summit.prediction.genotype import native_module
from scripts.epistasis.benchmark_robust_workflow import io


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    a.out.chmod(0o700)
    root = a.inputs.resolve()
    out = a.out.resolve()
    records = []
    flags = ["--num-threads", "2", "--memory-gib", "8", "--block-size", "128"]

    def run(name, args):
        t = time.perf_counter()
        cpu = time.process_time()
        before = io()
        cli(args)
        after = io()
        records.append(
            dict(
                stage=name,
                seconds=time.perf_counter() - t,
                cpu_seconds=time.process_time() - cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                io={k: after[k] - before[k] for k in before},
            )
        )
        print(name, records[-1]["seconds"], flush=True)

    spec = json.loads((root / "prespecified.json").read_text())
    spec["samples"] = str(root / spec["samples"])
    spec["phenotypes"]["file"] = str(root / spec["phenotypes"]["file"])
    spec["jobs"][0]["inference"]["save_reference"] = True
    path = out / "prespecified.json"
    path.write_text(json.dumps(spec))
    run(
        "private_whole_marker_3_trait_prepare",
        ["prepare", str(path), "--out", str(out / "prepared"), *flags],
    )
    run(
        "private_summary_fit",
        [
            "fit",
            str(out / "prepared/prespecified.robust-score.npz"),
            "--out",
            str(out / "fitted.json"),
        ],
    )
    train = json.loads((root / "train.json").read_text())
    for key in ("samples", "variants", "interaction_variants"):
        train[key] = str(root / train[key])
    train["phenotype"]["file"] = str(root / train["phenotype"]["file"])
    path = out / "train.json"
    path.write_text(json.dumps(train))
    run(
        "private_complete_training_6000",
        ["train-direction", str(path), "--out", str(out / "trained"), *flags],
    )
    confirm = json.loads((root / "confirm.json").read_text())
    confirm["samples"] = str(root / confirm["samples"])
    confirm["phenotypes"]["file"] = str(root / confirm["phenotypes"]["file"])
    for score in confirm["frozen_scores"]:
        score["direction"] = str(out / "trained/direction.json")
    path = out / "confirm.json"
    path.write_text(json.dumps(confirm))
    run(
        "private_independent_confirmation",
        ["prepare", str(path), "--out", str(out / "confirmed"), *flags],
    )
    run(
        "private_confirmation_fit",
        [
            "fit",
            str(out / "confirmed/learned.robust-score.npz"),
            "--out",
            str(out / "confirmed.fit.json"),
        ],
    )
    native = native_module()
    with a.receipt.open("x") as f:
        json.dump(
            _jsonable(
                dict(
                    n=10060,
                    m=454207,
                    training_n=6000,
                    confirmation_n=4060,
                    records=records,
                    native_path=native.__file__,
                    native_build=native.build_info(),
                    result=json.loads((out / "fitted.json").read_text()),
                    confirmation=json.loads((out / "confirmed.fit.json").read_text()),
                )
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
