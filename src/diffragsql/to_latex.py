"""Export QA evaluation metrics to a LaTeX table. No modeling claims implied."""
import argparse
import json
from pathlib import Path
import re
import yaml

_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    text = _CONTROL.sub("", str(value))
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        text = text.replace(a, b)
    return text


def export_metrics(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results_path = Path(cfg["outputs"]["results_json"])
    out_path = Path(cfg["outputs"]["latex_table"])
    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)
    lines = [r"\begin{tabular}{l r}", r"\hline", "Metric & Value " + r"\\", r"\hline"]
    for key, value in data.items():
        if not key.startswith("n_"):
            lines.append(f"{clean_value(key)} & {clean_value(value)} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad_demo.yaml")
    args = ap.parse_args()
    print(f"Wrote {export_metrics(args.config)}")


if __name__ == "__main__":
    main()
