"""Compatibility checks for command routing and shared CLI option names."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from summit.cli_options import explicit_options


def test_shared_binary_options_preserve_defaults_and_alias_values():
    from summit.cli import build_parser
    parser = build_parser()
    binary = parser.parse_args(["--binary-method", "pcgc"])
    assert (binary.nvecs, binary.seed, binary.step_size, binary.memory_gib) == (256, 0, 256, 1.)
    ordinary = parser.parse_args([])
    assert (ordinary.nvecs, ordinary.seed, ordinary.step_size, ordinary.njack) == (1000, None, 1000, "chr")
    old = ["--binary-probes=31", "--binary-seed", "19", "--binary-block-size=7", "--binary-memory-gib", "2", "--binary-genome-build", "test"]
    new = ["--nvecs=31", "--seed", "19", "--block-size=7", "--memory-gib", "2", "--genome-build", "test"]
    assert vars(parser.parse_args(["--binary-method", "pcgc", *old])) == vars(parser.parse_args(["--binary-method", "pcgc", *new]))
    assert explicit_options(parser, old) == explicit_options(parser, new)
    assert parser.parse_args(["--step_size", "auto"]).step_size == "auto"
    assert parser.parse_args(["--block-size", "auto"]).step_size == "auto"
    # Reusing a parser must not retain a previous mode's defaults.
    assert parser.parse_args([]).nvecs == 1000


@pytest.mark.parametrize("flags", [
    ["--nvecs", "31", "--binary-probes", "32"],
    ["--binary-seed=1", "--seed=2"],
    ["--step_size", "10", "--block-size", "11"],
    ["--binary-block-size", "10", "--step_size", "11"],
    ["--memory-gib", "1", "--binary-memory-gib", "2"],
])
def test_conflicting_alias_values_fail_before_computation(flags):
    from summit.cli import build_parser
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(flags)
    assert error.value.code == 2


def test_identical_alias_values_are_accepted_and_help_shows_one_spelling():
    from summit.cli import build_parser, _provided_long_options, _GXE_BATCH_REFERENCE_OPTIONS
    parser = build_parser()
    assert parser.parse_args(["--block-size=10", "--step_size=10"]).step_size == 10
    assert _provided_long_options(["--step_size=10"], parser=parser) & _GXE_BATCH_REFERENCE_OPTIONS == {"--block-size"}
    help_text = parser.format_help()
    for old in ("--binary-probes", "--binary-seed", "--binary-block-size", "--binary-memory-gib", "--binary-genome-build", "--step_size", "--collapse-reg-ld", "--save-trace", "--target-xz-mem"):
        assert old not in help_text
    for current in ("--nvecs", "--seed", "--block-size", "--memory-gib", "--genome-build", "--target-mem", "--gxe-total-memory-gib"):
        assert current in help_text
    assert "HE/LDSC regression:" in help_text
    assert "summit pgs" in help_text


def test_legacy_memory_override_precedence_is_preserved():
    from summit.cli import build_parser
    for flags in (["--target-xz-mem", "2", "--target-mem", "3"],
                  ["--target-mem", "3", "--target-xz-mem", "2"]):
        args = build_parser().parse_args(flags)
        assert (args.target_xz_mem, args.target_mem) == (2., 3.)


def test_binary_shared_preparation_options_are_not_silently_ignored(tmp_path, monkeypatch):
    from summit import cli
    for flag, value in (("--memory-gib", "1"), ("--genome-build", "test")):
        monkeypatch.setattr(sys, "argv", ["summit", "--h2", "missing", flag, value, "--out", str(tmp_path/"bad")])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", ["--nvecs", "--binary-probes", "--seed", "--binary-seed", "--block-size", "--binary-block-size", "--memory-gib", "--binary-memory-gib"])
def test_binary_fit_rejects_preparation_controls_even_with_aliases(tmp_path, flag):
    from summit.cli import build_parser
    from summit.pcgc.cli import run
    parser = build_parser()
    argv = ["--binary-method", "pcgc", "--h2", "missing", "--out", str(tmp_path/"fit"), flag, "10"]
    with pytest.raises(ValueError, match="sealed artifact"):
        run(parser.parse_args(argv), argv, parser=parser)
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


def test_unified_reference_plan_matches_legacy_units_and_options(capsys):
    from summit.entrypoint import main
    from summit.ldscore.generalized_gxe_variant_cli import main as legacy
    common = ["plan", "--samples", "200", "--variants", "1000", "--basis", "3", "--annotations", "2", "--genotype-format", "bed"]
    old = [*common, "--probes", "16", "--threads", "2", "--variant-block-width", "128", "--memory-bytes", str(2*2**30)]
    new = [*common, "--nvecs", "16", "--num-threads", "2", "--block-size", "128", "--memory-gib", "2"]
    assert legacy(old) == 0
    before = json.loads(capsys.readouterr().out)
    assert main(["reference", *new]) == 0
    assert json.loads(capsys.readouterr().out) == before
    with pytest.raises(SystemExit):
        main(["reference", *new, "--memory-bytes", str(2*2**30)])


def test_zpass_aliases_reach_identical_authenticated_inputs(monkeypatch, capsys):
    from summit.context import reference_zpass_cli as cli
    from summit.entrypoint import main
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {"seconds": 0}
    monkeypatch.setattr(cli, "run_zpass", capture)
    common = ["reference", "zpass", "--manifest", "manifest", "--reference-root", "ref", "--chromosome", "22", "--master-input", "master"]
    main([*common, "--geno", "study", "--annot", "a.npy", "--out", "z.npz", "--block-size", "9"])
    capsys.readouterr()
    main([*common, "--bed-prefix", "study", "--annotations", "a.npy", "--output", "z.npz", "--width", "9"])
    capsys.readouterr()
    assert calls[0] == calls[1]
