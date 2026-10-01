"""Run two predefined point-count experiments sequentially, then evaluate both."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    suffix = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    names = {n: f"points{n}_{suffix}_seed42" for n in (256, 512)}
    manifest_path = ROOT / "results/metrics" / f"point_count_suite_{suffix}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    steps = []
    # 先完成两组训练并按验证集选权重，再运行任何新增测试评估。
    for n, name in names.items():
        steps.append({"name": name, "kind": "train", "num_points": n,
                      "command": [sys.executable, "-u", "train.py", "--num-points", str(n),
                                  "--augmentation", "none", "--epochs", "50", "--batch-size", "32",
                                  "--learning-rate", "0.001", "--dropout", "0.3", "--seed", "42",
                                  "--val-fraction", "0.2", "--num-workers", "0", "--cpu-threads", "4",
                                  "--device", args.device, "--run-name", name]})
    for n, name in names.items():
        steps.append({"name": f"test_{name}", "kind": "test", "num_points": n,
                      "command": [sys.executable, "-u", "test.py", "--checkpoint",
                                  f"checkpoints/{name}/best_model.pth", "--device", args.device,
                                  "--run-name", f"test_{name}"]})
    suite = {"status": "running", "factor": "num_points", "values": [256, 512, 1024],
             "baseline_training": "results/metrics/baseline_20261001_seed42_r2.json",
             "baseline_test": "results/metrics/test_baseline_20261001_seed42.json",
             "sampling_note": "Same sampler and seed per mesh; clouds at different N are not nested subsets. Each cloud is normalized independently.",
             "schedule": "Sequential training of both configurations, then test evaluation of both selected checkpoints",
             "steps": steps}

    def save():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(suite, indent=2) + "\n")
        temporary.replace(manifest_path)

    save()
    print(f"Suite: {manifest_path}", flush=True)
    try:
        for step in steps:
            step["status"] = "running"
            step["log"] = f"logs/{step['name']}.log"
            save()
            print(f"Starting {step['kind']}: {step['name']}", flush=True)
            with (ROOT / step["log"]).open("x") as log:
                subprocess.run(step["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            step["status"] = "completed"
            step["report"] = f"results/metrics/{step['name']}.json"
            save()
            print(f"Completed: {step['name']}", flush=True)
        suite["status"] = "completed"
    except Exception as error:
        step["status"] = "failed"
        suite.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
