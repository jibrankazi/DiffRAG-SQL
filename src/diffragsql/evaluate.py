import argparse, os, json, numpy as np, yaml, matplotlib.pyplot as plt
from diffragsql.data import load_squad_splits
from diffragsql.retriever import TFIDFRetriever
from diffragsql.reader import QAReader
from diffragsql.metrics import exact_match, f1_score
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); out_dir = cfg['outputs']['run_dir']; os.makedirs(out_dir, exist_ok=True)
    ds_train, ds_eval = load_squad_splits(cfg['data']['dataset_id'], cfg['data']['split_train'], cfg['data']['split_eval'])
    contexts = list({ex['context'] for ex in ds_train}); retr = TFIDFRetriever(contexts)
    reader = QAReader(cfg['reader']['model_name'], cfg['reader']['max_length'], cfg['reader']['doc_stride'])
    ems, f1s, confs, faiths, eval_records = [], [], [], [], []
    for i, ex in enumerate(ds_eval.select(range(min(200, len(ds_eval))))):
        q = ex['question']; answers = ex['answers']['text'] if isinstance(ex['answers'], dict) else ex['answers']
        hits = retr.search(q, k=cfg['retriever']['max_docs'])
        pred = reader.answer(q, [h['doc'] for h in hits])
        em = exact_match(pred['answer'], answers); f1 = f1_score(pred['answer'], answers); conf = pred['score']
        faithful = any(pred['answer'] and pred['answer'].lower() in h['doc'].lower() for h in hits)
        ems.append(em); f1s.append(f1); confs.append(conf); faiths.append(float(faithful))
        eval_records.append({'id': ex.get('id', i), 'pred': pred['answer'], 'score': conf, 'em': em, 'f1': f1, 'faithful': faithful})
    results = {'EM': float(np.mean(ems)), 'F1': float(np.mean(f1s)), 'Faithfulness@k': float(np.mean(faiths)), 'AbstentionRate@thr': float(np.mean([1.0 if c < cfg['eval']['abstain_threshold'] else 0.0 for c in confs]))}
    json.dump(results, open(cfg['outputs']['results_json'], 'w'), indent=2)
    json.dump(eval_records, open(os.path.join(out_dir, 'eval_records.json'), 'w'))
    os.makedirs(cfg['outputs']['fig_dir'], exist_ok=True)
    plt.figure(); plt.scatter(ems, faiths, alpha=0.5); plt.xlabel('Exact Match (per item)'); plt.ylabel('Faithful (1/0)'); plt.title('Faithfulness vs EM'); plt.savefig(os.path.join(cfg['outputs']['fig_dir'], 'faith_vs_em.png'), dpi=200)
    print('Results:', results)
if __name__ == '__main__': main()
