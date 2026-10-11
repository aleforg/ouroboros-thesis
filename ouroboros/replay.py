"""Rigenera con prompt/semi salvati; non sovrascrive mai le immagini originali."""

from pathlib import Path

from .analysis import bootstrap
from .storage import environment, read_json, save_json, sha256, verify_images


def replay(root, target, limit=8):
    root = Path(root)
    state = read_json(root / "checkpoint.json")
    verify_images(root, state)
    if limit < 1:
        raise ValueError("limit deve essere positivo")
    results = []
    try:
        for record in state["records"]:
            for image in record["images"]:
                if len(results) >= limit:
                    break
                destination = root / "replay" / image["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                regenerated = target.generate(record["prompt"], image["seed"])
                regenerated.save(destination, format="PNG")
                digest = sha256(destination)
                results.append({"profession": record["profession"], "path": image["path"],
                                "original_sha256": image["sha256"], "replay_sha256": digest,
                                "match": digest == image["sha256"]})
    finally:
        target.unload()
    groups = [[r["match"] for r in results if r["profession"] == p]
              for p in sorted({r["profession"] for r in results})]
    report = {"environment": environment(), "images": results,
              "match_rate": bootstrap(groups, lambda sampled: sum(sum(g) for g in sampled) / sum(len(g) for g in sampled)),
              "note": "IC ricampionando professioni; riproducibilità bit-level non garantita tra hardware/backend diversi"}
    save_json(root / "replay.json", report)
    return report
