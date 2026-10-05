"""Compare single-token and block target logits on real benchmark prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=Path("notes/asahi/bench.json"))
    ap.add_argument("--out", type=Path, default=Path("notes/asahi/verify_probe.json"))
    args = ap.parse_args()
    bench = json.loads(args.bench.read_text())
    meta = bench["models"]["target"]
    root = Path.home() / ".cache/huggingface/hub"
    path = root / ("models--" + meta["model"].replace("/", "--")) / "snapshots" / meta["revision"]
    actual = hashlib.file_digest((path / "model.safetensors").open("rb"), "sha256").hexdigest()
    assert actual == meta["files"]["model.safetensors"]
    model, tok = load(str(path))
    results = []
    for row in bench["rows"]:
        if row["repeat"] != 1:
            continue
        index = row["first_mismatch_index"]
        index = 20 if index is None else index
        prompt = tok.encode(row["prompt"])
        continuation = [prompt[-1], *row["baseline"]["tokens"][:index]]
        width = min(4, len(continuation))
        caches = [make_prompt_cache(model), make_prompt_cache(model)]
        for cache in caches:
            model(mx.array(prompt[:-1])[None], cache=cache)
            mx.eval([c.state for c in cache])
            for token in continuation[:-width]:
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
        # These two cache histories are identical. Only the final query width
        # changes; the draft model and speculative acceptance loop are absent.
        for token in continuation[-width:]:
            single = model(mx.array([[token]]), cache=caches[0])[0, -1]
            mx.eval(single)
        block = model(mx.array([continuation[-width:]]), cache=caches[1])[0, -1]
        mx.eval(block)
        assert [c.offset for c in caches[0]] == [c.offset for c in caches[1]]

        def top(logits):
            # argsort may order ties differently from the sampler's argmax.
            winner = int(mx.argmax(logits).item())
            others = [i for i in mx.argsort(logits)[-5:].tolist() if i != winner]
            others.sort(key=lambda i: (-float(logits[i].item()), i))
            ids = [winner, *others[:4]]
            return [{"token": i, "text": tok.decode([i]), "logit": float(logits[i].item())}
                    for i in ids]

        s_top, b_top = top(single), top(block)
        delta = single.astype(mx.float32) - block.astype(mx.float32)
        result = {
            "name": row["name"], "token_index": index, "block_width": width,
            "single_top5": s_top, "block_top5": b_top,
            "single_top1_gap": s_top[0]["logit"] - s_top[1]["logit"],
            "block_top1_gap": b_top[0]["logit"] - b_top[1]["logit"],
            "max_abs_logit_difference": mx.max(mx.abs(delta)).item(),
            "rms_logit_difference": mx.sqrt(mx.mean(delta * delta)).item(),
            "top1_equal": s_top[0]["token"] == b_top[0]["token"],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.out.write_text(json.dumps({
        "target": meta, "mlx_provenance": bench["mlx_provenance"],
        "harness_sha256": hashlib.file_digest(open(__file__, "rb"), "sha256").hexdigest(),
        "procedure": "Actual greedy benchmark prefixes; identical cached histories; "
                     "last up-to-four input tokens evaluated sequentially or in one block. "
                     "No draft model, quantized weights, or accept/trim loop.",
        "rows": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
