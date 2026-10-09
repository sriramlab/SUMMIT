"""Functional checks for command routing and shared CLI option names."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from summit.cli_options import explicit_options


def test_shared_options_preserve_workflow_defaults():
    from summit.cli import build_parser, _provided_long_options, _GXE_BATCH_REFERENCE_OPTIONS
    parser = build_parser()
    binary = parser.parse_args(["--binary-method", "pcgc"])
    assert (binary.nvecs, binary.seed, binary.step_size, binary.memory_gib, binary.genome_build) == (256, 0, 256, 1., None)
    assert binary.binary_output_format == "separate"
    ordinary = parser.parse_args([])
    assert (ordinary.nvecs, ordinary.seed, ordinary.step_size, ordinary.memory_gib, ordinary.njack) == (1000, None, 1000, "auto", "chr")
    args = parser.parse_args(["--binary-method", "pcgc", "--nvecs=31", "--seed", "19", "--block-size=7", "--memory-gib", "2"])
    assert (args.nvecs, args.seed, args.step_size, args.memory_gib) == (31, 19, 7, 2.)
    assert parser.parse_args(["--block-size", "auto"]).step_size == "auto"
    assert _provided_long_options(["--block-size=10"]) & _GXE_BATCH_REFERENCE_OPTIONS == {"--block-size"}
    assert parser.parse_args([]).nvecs == 1000
    assert explicit_options(["--seed=1", "--", "--nvecs"]) == {"--seed"}


@pytest.mark.parametrize("flag", ["--binary-probes", "--binary-seed", "--binary-block-size",
    "--binary-memory-gib", "--binary-genome-build", "--step_size", "--target-mem",
    "--target-xz-mem", "--trace", "--save-trace", "--collapse-reg-ld", "--intercept-rg",
    "--intercept-weight-mode", "--intercept-chisq-thr", "--skip-kmoments", "--write-ld-mc-ci",
    "--align-alleles", "--nvec", "--num-thread", "--force_affinity_all", "--decode_threads_cap"])
def test_removed_or_abbreviated_options_fail_clearly(flag):
    from summit.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, "1"])
    assert error.value.code == 2
    assert flag not in parser._option_string_actions


def test_help_shows_shared_controls():
    from summit.cli import build_parser
    help_text = build_parser().format_help()
    for current in ("--nvecs", "--seed", "--block-size", "--memory-gib", "--genome-build", "--gxe-total-memory-gib"):
        assert current in help_text
    assert "HE/LDSC regression:" in help_text
    assert "summit pgs" in help_text


@pytest.mark.parametrize("flag,value", [
    ("--nvecs", "10"), ("--seed", "10"), ("--block-size", "10"),
    ("--memory-gib", "10"), ("--binary-output-format", "combined"),
])
def test_binary_fit_rejects_preparation_controls(tmp_path, flag, value):
    from summit.cli import build_parser
    from summit.pcgc.cli import run
    argv = ["--binary-method", "pcgc", "--h2", "missing", "--out", str(tmp_path/"fit"), flag, value]
    with pytest.raises(ValueError, match="remove preparation options") as error:
        run(build_parser().parse_args(argv), argv)
    assert flag in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_pgs_dispatch_and_help_do_not_import_the_legacy_runtime():
    code = '''
import sys
# Select the checkout explicitly: a developer environment may have an
# editable installation for a different worktree. No numerical imports.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_gwldcore_editable']
sys.path.insert(0, sys.argv[1])
from summit.entrypoint import main
assert 'summit.cli' not in sys.modules and 'numpy' not in sys.modules
try:
    main(['pgs', '--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('expected help exit')
assert 'summit.cli' not in sys.modules and 'numpy' not in sys.modules
'''
    source = Path(__file__).resolve().parents[1]/"src"
    result = subprocess.run([sys.executable, "-B", "-c", code, str(source)], env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "summit pgs" in result.stdout


def test_reference_plan_converts_gib_and_uses_shared_options(capsys):
    from summit.entrypoint import main
    from summit.ldscore.generalized_gxe_variant_cli import build_parser
    options = ["plan", "--samples", "200", "--variants", "1000", "--basis", "3", "--annotations", "2", "--genotype-format", "bed",
               "--nvecs", "16", "--num-threads", "2", "--block-size", "128", "--memory-gib", "2"]
    assert build_parser().parse_args(options).memory_limit_bytes == 2*2**30
    assert main(["reference", *options]) == 0
    assert json.loads(capsys.readouterr().out)["planned_reference_genotype_passes"] == 2
    with pytest.raises(SystemExit):
        main(["reference", *options, "--memory-bytes", str(2*2**30)])


def test_zpass_shared_options_reach_execution(monkeypatch, capsys):
    from summit.context import reference_zpass_cli as cli
    from summit.entrypoint import main
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {"seconds": 0}
    monkeypatch.setattr(cli, "run_zpass", capture)
    main(["reference", "zpass", "--manifest", "manifest", "--reference-root", "ref", "--chromosome", "22", "--master-input", "master",
          "--geno", "study", "--annot", "a.npy", "--out", "z.npz", "--block-size", "9"])
    assert calls[0]["bed_prefix"] == Path("study")
    assert calls[0]["width"] == 9
    assert calls[0]["output"] == Path("z.npz")
