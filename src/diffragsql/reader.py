import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import List
class QAReader:
    def __init__(self, model_name='distilbert-base-uncased', max_length=384, doc_stride=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_length = max_length
        self.doc_stride = doc_stride
    def answer(self, question: str, contexts: List[str]):
        if len(contexts) == 0:
            return {'answer': '', 'score': 0.0, 'start': 0, 'end': 0}
        context = contexts[0]
        inputs = self.tokenizer(question, context, return_tensors='pt', truncation=True, max_length=self.max_length, stride=self.doc_stride)
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits[0].softmax(dim=-1)
            end_scores = outputs.end_logits[0].softmax(dim=-1)
            start_idx = int(torch.argmax(start_scores))
            end_idx = int(torch.argmax(end_scores))
            if end_idx < start_idx: end_idx = start_idx
            score = float((start_scores[start_idx] * end_scores[end_idx]).item())
            ans_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(ans_tokens, skip_special_tokens=True)
        return {'answer': answer.strip(), 'score': score, 'start': start_idx, 'end': end_idx}
