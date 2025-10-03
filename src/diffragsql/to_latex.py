import argparse, json, yaml, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(os.path.dirname(cfg['outputs']['latex_table']), exist_ok=True)
    if not os.path.exists(cfg['outputs']['results_json']):
        raise FileNotFoundError(f"Results JSON not found at {cfg['outputs']['results_json']}. Run evaluate first.")
    res = json.load(open(cfg['outputs']['results_json']))
    table = (
        "\\begin{tabular}{l r}\n"
        "\\hline\n"
        "Metric & Value \\\\ \\hline\n"
        f"EM & {res['EM']:.4f} \\\\ \n"
        f"F1 & {res['F1']:.4f} \\\\ \n"
        f"Faithfulness@k & {res['Faithfulness@k']:.4f} \\\\ \n"
        f"AbstentionRate@thr & {res['AbstentionRate@thr']:.4f} \\\\ \\hline\n"
        "\\end{tabular}\n"
    )
    open(cfg['outputs']['latex_table'], 'w').write(table)
    print('Wrote LaTeX table to', cfg['outputs']['latex_table'])

if __name__ == '__main__':
    main()
