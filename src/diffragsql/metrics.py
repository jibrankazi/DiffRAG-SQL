import re
def normalize_text(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return re.sub(r'[^\w\s]', '', text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
def exact_match(prediction, ground_truths):
    if not prediction: return 0.0
    pred = normalize_text(prediction)
    return float(any(normalize_text(gt) == pred for gt in ground_truths))
def f1_score(prediction, ground_truths):
    pred = normalize_text(prediction).split()
    def _f1(p, gt):
        gt = normalize_text(gt).split()
        common = set(p) & set(gt)
        if len(p) == 0 or len(gt) == 0: return float(p == gt)
        if len(common) == 0: return 0.0
        prec = len(common)/len(p); rec = len(common)/len(gt)
        return 2*prec*rec/(prec+rec)
    return max(_f1(pred, gt) for gt in ground_truths) if ground_truths else 0.0
