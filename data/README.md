# data

- `raw/` : 원본 데이터셋 (커밋 X)
- `processed/` : 실험 입력용 jsonl (커밋 X)

QA jsonl 형식 (`rag.data.loader.load_qa_jsonl`):

```json
{"id": "q1", "question": "...", "answers": ["..."]}
```

데이터셋 다운로드/전처리 방법은 여기에 기록해서 팀원이 재현할 수 있게 한다.
