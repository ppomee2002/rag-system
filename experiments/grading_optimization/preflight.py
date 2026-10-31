import os
from pathlib import Path
root=Path(__file__).resolve().parents[2]
print('API key present:',bool(os.getenv('OPENAI_API_KEY')))
print('Model:',os.getenv('EXPERIMENT_MODEL','gpt-4.1-mini-2025-04-14'))
for name in ['data/grading_optimization/selected_203.jsonl','data/grading_optimization/corpus_5/index.faiss','data/grading_optimization/corpus_5/chunks.jsonl','data/grading_optimization/corpus_5/manifest.json']:
    print(name,'READY' if (root/name).exists() else 'MISSING')
print('No API request sent; no key written to disk.')
