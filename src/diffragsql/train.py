import argparse, os, json, random, numpy as np, yaml, torch
from diffragsql.data import load_squad_splits
from diffragsql.retriever import TFIDFRetriever
from diffragsql.reader import QAReader
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); set_seed(cfg.get('seed', 1337))
    out_dir = cfg['outputs']['run_dir']; os.makedirs(out_dir, exist_ok=True)
    ds_train, ds_eval = load_squad_splits(cfg['data']['dataset_id'], cfg['data']['split_train'], cfg['data']['split_eval'])
    contexts = list({ex['context'] for ex in ds_train}); retr = TFIDFRetriever(contexts)
    reader = QAReader(cfg['reader']['model_name'], cfg['reader']['max_length'], cfg['reader']['doc_stride'])
    results = []
    for i, ex in enumerate(ds_train.select(range(min(100, len(ds_train))))):
        q = ex['question']; answers = ex['answers']['text'] if isinstance(ex['answers'], dict) else ex['answers']
        hits = retr.search(q, k=cfg['retriever']['max_docs'])
        pred = reader.answer(q, [h['doc'] for h in hits])
        results.append({'id': ex.get('id', i), 'question': q, 'pred': pred['answer'], 'score': pred['score'], 'gts': answers, 'contexts': hits})
    json.dump(results, open(os.path.join(out_dir, 'train_preds.json'), 'w'))
    json.dump(cfg, open(os.path.join(out_dir, 'config_snapshot.json'), 'w'), indent=2)
    print('Training pipeline complete (light demo).')
if __name__ == '__main__': main()
