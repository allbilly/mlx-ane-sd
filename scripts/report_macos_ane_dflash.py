"""Audit completed native-ANE receipts and write their macOS experiment report."""
import hashlib
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    receipt = root / "notes/m1_macos_ane_dflash.json"
    r = json.loads(receipt.read_text())
    assert r["status"] == "complete"
    for name, expected in r["source_sha256"].items():
        assert hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() == expected, name
    assert len(r["configs"]) == 3
    lines = ["# M1 macOS: native ANE DFlash + MLX target", "",
             "The separate ANE runtime works, but this experiment did not reproduce a 3× speedup. "
             "MLX itself runs on CPU/GPU; the draft calls ANEForge's native e5rt runtime with "
             "ANE-only device mask `0x4`. The bf16 target remains on MLX/Metal.", "",
             "Hardware: base M1 MacBook Air, 8 GB, macOS 27.0.1. Target: pinned "
             "`mlx-community/Qwen3-0.6B-bf16`. Draft: public "
             "`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`, converted from bf16 to fp16 for ANE. "
             "Three draft layers, 8-token trained block, seven proposed tokens, fixed context "
             "capacity 256. The full vocabulary ANE head is tiled at 8,192 columns; no reduced-vocabulary approximation.", "",
             "Four repository prompts × two passes × 100 tokens per configuration. Baseline/SD order "
             "alternates between passes. Both device locks are held for the complete sweep. "
             "All paths use stock mlx-lm's recommended wired-memory limit. Loading, compilation and "
             "prefill are excluded; drafting, transfers, verification, cache commits and rollback "
             "are included in decode timing.", "",
             "| Configuration | Stock MLX mean tok/s | SD mean tok/s | Mean paired speedup | Best trial | Matching greedy trials |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    profiles = []
    for c in r["configs"]:
        assert c["status"] == "complete" and c["ane_device_mask"] == 4
        assert c["draft_parity"]["qualified"] and len(c["draft_parity"]["rows"]) == 4
        assert len(c["rows"]) == 8
        for row in c["rows"]:
            d = row["dflash"]
            assert len(d["tokens"]) == len(row["baseline"]["tokens"]) == len(row["stock_mlx_lm"]["tokens"]) == 100
            assert row["stock_token_identity"]
            assert d["cycles"] == len(d["verification_trace"])
            proposes = sum(bool(t["proposals"]) for t in d["verification_trace"])
            assert d["ane_evaluations"] == proposes * (2 if c["head"] == "ane" else 1)
            assert d["ane_evaluations"] > 0
        s = c["summary"]
        assert abs(s["mean_speedup_vs_stock"] - statistics.mean(row["speedup_vs_stock"] for row in c["rows"])) < 1e-12
        lines.append(f"| {c['name']} | {s['stock_mean_tps']:.2f} | {s['dflash_mean_tps']:.2f} | "
                     f"{s['mean_speedup_vs_stock']:.2f}× | {s['max_speedup_vs_stock']:.2f}× | {s['identity_trials']}/8 |")
        cycles = sum(row["dflash"]["cycles"] for row in c["rows"])
        profiles.append((c, {k: 1000 * sum(row["dflash"]["draft_profile_s"][k] for row in c["rows"]) / cycles
                             for k in ("body", "head", "context")},
                         1000 * sum(row["dflash"]["target_verify_s"] for row in c["rows"]) / cycles))
    serial = r["configs"][-1]
    assert serial["summary"]["qualified"] and serial["summary"]["identity_trials"] == 8
    lines += ["", "The serial-verification configuration matches all **800/800 generated token IDs** "
              "against both scalar greedy decoding and stock mlx-lm. Batched GPU verification "
              "changes near-tie argmax results on three prompts and is retained as a numerical variant, "
              "not a byte-identical reproduction.", "",
              "| Configuration | ANE body ms/cycle | Head ms/cycle | Context/transfer ms/cycle | GPU verify ms/cycle | Native ANE calls |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c, p, verify in profiles:
        lines.append(f"| {c['name']} | {p['body']:.2f} | {p['head']:.2f} | {p['context']:.2f} | {verify:.2f} | {c['summary']['ane_evaluations']} |")
    speeds = [row["stock_mlx_lm"]["generation_tps"] for c in r["configs"] for row in c["rows"]]
    cosines = [row["cosine"] for c in r["configs"] for row in c["draft_parity"]["rows"]]
    lines += ["", "## Correctness and placement", "",
              f"Each compiled draft passed real-input comparison with the MLX fp16 reference "
              f"on all four prompts (minimum hidden-state cosine similarity {min(cosines):.8f}). "
              "The first proposed token matched the reference on every prompt. Both native "
              "program constructors reject a device mask other than `0x4`; successful native "
              "evaluation counts are retained for every trial. The body fuses 263 graph operations. "
              "ANE-head variants execute the full vocabulary through a second native program.", "",
              "The initial graph returned zero hidden states: ANEForge's FP16 `reduce_l2_norm` "
              "overflowed on real projector/residual values reaching about 8,000. Power-of-two "
              "rescaling before normalization fixed the issue. Zero-output runs are marked invalid "
              "in `m1_macos_ane_dflash_zero_output.*`; the superseded unwired control is retained "
              "in `m1_macos_ane_dflash_unwired.*`. These are not speedup evidence.", "",
              "## Limits", "",
              f"Stock MLX rates ranged from {min(speeds):.2f} to {max(speeds):.2f} tok/s during "
              "the desktop session. Exclusive GPU/ANE reservations do not eliminate background "
              "CPU activity, paging or clock variation. These are paired measurements of this "
              "implementation, not an isolated hardware ceiling or a claim that M1 ANE strength "
              "alone explains the result. Matching the wired-memory setting did not recover "
              "batched verification performance.", "",
              "This tests an ANE draft with a GPU target. The README's best result also offloads "
              "the much larger Qwen3-4B target to ANE on an M4 Pro/64 GB machine. That full-ANE "
              "stack has not been reproduced here. Decode timing includes the first target forward "
              "after prompt prefill; stock mlx-lm's own timing convention differs by roughly one "
              "token at this generation length.", "",
              "Existing Asahi source files and reference repositories were not modified.", "",
              "## Reproduce", "", "```bash",
              "cd /Users/yeren/Desktop/mlx-ane-sd",
              ".asahi/venv-metal/bin/python -u scripts/bench_macos_ane_dflash.py",
              ".asahi/venv-metal/bin/python scripts/report_macos_ane_dflash.py", "```", "",
              "Uses the already downloaded, pinned HF target and public draft plus the existing "
              "built ANEForge runtime in `~/Desktop/ANEForge`. Source hashes, model revisions/weight "
              "hashes, runtime library hash, host states, token IDs and per-cycle traces are in "
              "[the raw receipt](m1_macos_ane_dflash.json). See [the reference survey](m1_macos_ane_reference_survey.md) "
              "for MLX/Core ML/direct-ANE distinctions.", ""]
    (root / "notes/m1_macos_ane_dflash.md").write_text("\n".join(lines))
    audit = {"receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
             "source_hashes_match": True, "complete_trials": 24, "native_ane_calls":
             sum(c["summary"]["ane_evaluations"] for c in r["configs"]),
             "serial_identity_tokens": 800, "draft_parity_passed": True,
             "speedup_3x_reproduced": False}
    (root / "notes/m1_macos_ane_dflash_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
