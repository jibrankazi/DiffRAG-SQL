"""Real Stanford SQuAD questions + pretrained extractive QA integration.

No SQL, backpropagation or model training is claimed. Measures retrieval hit,
exact match and token F1 for held-out published SQuAD dev questions.
"""
import argparse
import json
import random
from pathlib import Path
import requests
from diffragsql.retriever import TFIDFRetriever
from diffragsql.reader import QAReader
from diffragsql.metrics import exact_match, f1_score

SOURCE="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
CHECKPOINT="distilbert/distilbert-base-cased-distilled-squad"


def load_actual_squad(count=16):
    if count<3:
        raise ValueError("Need at least three real questions")
    r=requests.get(SOURCE,timeout=75);r.raise_for_status()
    source=r.json()
    candidates=[]
    for article in source["data"]:
        for par in article["paragraphs"]:
            for record in par["qas"]:
                answers=[x["text"] for x in record["answers"] if x.get("text")]
                if answers:
                    candidates.append({"question":record["question"],"context":par["context"],"answers":answers})
    if len(candidates)<count:
        raise ValueError("Insufficient real SQuAD questions")
    sample=random.Random(42).sample(candidates,count)
    return sample


def evaluate(count=16):
    records=load_actual_squad(count)
    contexts=list(dict.fromkeys([r["context"] for r in records]))
    if len(contexts)<3:
        raise ValueError("Insufficient independent official SQuAD passage contexts")
    retr=TFIDFRetriever(contexts)
    reader=QAReader(CHECKPOINT,max_length=384,doc_stride=128)
    results=[]
    for r in records:
        hits=retr.search(r["question"],k=min(3,len(contexts)))
        retrieved=[h["doc"] for h in hits]
        prediction=reader.answer(r["question"],retrieved)
        results.append({
            "id":len(results),
            "grounded_context_retrieved":r["context"] in retrieved,
            "exact_match":exact_match(prediction["answer"],r["answers"]),
            "token_f1":f1_score(prediction["answer"],r["answers"]),
            "prediction":prediction["answer"],
            "confidence":prediction["score"]
        })
    metrics={
        "source":SOURCE,"model_checkpoint":CHECKPOINT,"real_squad_dev_questions":len(records),
        "unique_reference_passages":len(contexts),
        "retrieval_hit_at_3":sum(x["grounded_context_retrieved"] for x in results)/len(results),
        "answer_exact_match":sum(x["exact_match"] for x in results)/len(results),
        "answer_token_f1":sum(x["token_f1"] for x in results)/len(results),
        "no_sql_execution":True,
        "no_training_or_joint_gradient_updates":True,
        "individual_results":results
    }
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--count",type=int,default=16)
    p.add_argument("--output",default="runs/real_squad/results.json")
    args=p.parse_args()
    metrics=evaluate(args.count)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps({k:v for k,v in metrics.items() if k!="individual_results"},indent=2))

if __name__=="__main__":
    main()
