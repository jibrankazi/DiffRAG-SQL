from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDFRetriever:
    def __init__(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
        self.doc_mat = self.vectorizer.fit_transform(docs)
        self.docs = docs

    def search(self, query: str, k: int = 5) -> List[Dict]:
        qv = self.vectorizer.transform([query])
        scores = (qv @ self.doc_mat.T).toarray().flatten()
        idx = np.argsort(-scores)[:k]
        return [{'doc': self.docs[i], 'score': float(scores[i])} for i in idx]
