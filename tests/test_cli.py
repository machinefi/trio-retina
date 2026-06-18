"""Tests for the `retina` CLI (retina.cli.main)."""

import pytest

from retina import __version__
from retina.cli import main


def _run(argv, capsys):
    """Run main(argv), return (rc, stdout, stderr)."""
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_version(capsys):
    # argparse `--version` prints to stdout and raises SystemExit(0).
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_no_args_prints_help(capsys):
    rc, out, _ = _run([], capsys)
    assert rc == 0
    assert "demo" in out and "validate" in out


def test_demo_emits_json_events(capsys):
    rc, out, err = _run(["demo"], capsys)
    assert rc == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines, "demo produced no events"
    # Every line is valid JSON with the required event keys.
    import json

    for ln in lines:
        ev = json.loads(ln)
        assert {"type", "t", "src"} <= ev.keys()
    assert "emitted" in err  # summary on stderr


def test_demo_quiet_has_no_summary(capsys):
    rc, out, err = _run(["demo", "-q"], capsys)
    assert rc == 0
    assert out.strip()
    assert err.strip() == ""


def test_validate_all_valid(tmp_path, capsys):
    p = tmp_path / "events.jsonl"
    p.write_text(
        '{"type":"zone.enter","t":1.0,"src":"cam"}\n'
        '{"type":"line.cross","t":2.0,"src":"cam","dir":"a_to_b"}\n'
    )
    rc, out, _ = _run(["validate", str(p)], capsys)
    assert rc == 0
    assert "2 valid, 0 invalid" in out


def test_validate_reports_invalid_and_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "events.jsonl"
    p.write_text(
        '{"type":"zone.enter","t":1.0,"src":"cam"}\n'  # valid
        '{"t":2.0,"src":"cam"}\n'  # missing type
        '{"type":"x","t":3.0,"src":"cam","conf":2.5}\n'  # conf out of range
        "not json\n"  # bad JSON
    )
    rc, out, _ = _run(["validate", str(p)], capsys)
    assert rc == 1
    assert "1 valid, 3 invalid" in out
    assert "line 2" in out
    assert "type" in out


def test_validate_missing_file(tmp_path, capsys):
    rc, _, err = _run(["validate", str(tmp_path / "nope.jsonl")], capsys)
    assert rc == 2
    assert "cannot open" in err


def test_bench_prints_msframe(capsys):
    rc, out, _ = _run(["bench", "--frames", "100", "--tracks", "5", "--warmup", "10"], capsys)
    assert rc == 0
    assert "ms/frame" in out
    assert "Retina overhead" in out
