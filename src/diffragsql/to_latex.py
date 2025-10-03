import argparse, json, yaml, os
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); res = json.load(open(cfg['outputs']['results_json']))
    os.makedirs(os.path.dirname(cfg['outputs']['latex_table']), exist_ok=True)
    table = f"""%% Auto-generated
\begin{{tabular}}{{l r}}
\hline
Metric & Value \\ \hline
EM & {res['EM']:.4f} \\
F1 & {res['F1']:.4f} \\
Faithfulness@k & {res['Faithfulness@k']:.4f} \\
AbstentionRate@thr & {res['AbstentionRate@thr']:.4f} \\ \hline
\end{{tabular}}
"""
    open(cfg['outputs']['latex_table'], 'w').write(table)
    print('Wrote LaTeX table to', cfg['outputs']['latex_table'])
if __name__ == '__main__': main()
