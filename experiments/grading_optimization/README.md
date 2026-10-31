# Grading Optimization — current workflow

Current user-selected size: **50 measured PopQA questions + 3 separate warmup questions**.
Current execution permission: **prepare documents/index only; wait before Grading or Scheduling measurement**.

## Prepare only (default; no LLM answers generated)
```powershell
uv run python experiments/grading_optimization/start_local.py --limit 50
```
This prepares the selected-query Wikipedia full-text corpus and BGE/FAISS index, then stops.
Existing document cache is reused. Embedding checkpoints are saved every 64 chunks.

Files: data/grading_optimization/corpus_50/pages.jsonl, chunks.jsonl, index.faiss, manifest.json.
Manifest records question IDs, corpus scope, embedding model and index hash.

## Grading measurement — only after authorization
```powershell
uv run python experiments/grading_optimization/start_local.py --limit 50 --run-measurement
```
This starts the local model server, verifies actual GPU memory placement, runs Full/Lazy-hit/Lazy-rest (50 queries each; 3 warmups per variant excluded), saves measurements and reports, then stops before Scheduling.
Default repetitions: 1 (matches the existing profiling approach). Use --repeats 3 for repeated measurement if time permits.
To explicitly include scheduling trace replay after Grading, add --with-scheduling.

## Local execution and connection to team code
- Native Ollama API on 127.0.0.1:11434, llama3.1:8b, temperature 0, max_tokens 512, num_ctx 6144.
- Uses Vulkan on RTX 3050; driver is unchanged. 4GB VRAM means CPU offloading may occur. Model placement is recorded.
- LangGraph Full uses original team prompts and reproduces the shared CRAG class's calls/actions/contexts under deterministic mocks. Verify with verify_local.py.
- Reuses shared FAISS retriever, stage timing, quality metrics and original token-chunking function. Shared source and configs/base.yaml are unchanged.
- BGE encoder: BAAI/bge-base-en-v1.5 on CPU. Normalized embeddings, FAISS IndexFlatIP, top 5 chunks, 480-token chunks with overlap 64.
- Corpus: full Wikipedia subject/object pages for selected questions. This is smaller than the teammate's all-longtail corpus; absolute comparisons with their results are not controlled.
- Correction: original query-rewrite prompt followed by local FAISS re-search, not live web search and no replayed Tavily delay.

## Outputs and interpretation
results/grading_optimization/TIMESTAMP/: incremental results.jsonl, measurement_summary.csv, report.md, figures/, local_gpu_samples.csv, local_gpu_summary.csv, model_placement.json, config.json and STATUS.txt.
Native calls record server total/load/prompt/decode durations and token counts. Server time includes any CPU offload and is not pure GPU kernel time. Whole-device GPU utilization/memory/power are separate telemetry and may include other applications.
Full repetition 0 exports profile.jsonl for the existing scheduling trace schema. Scheduling replay is not live vLLM validation.

## Resume a measured run
```powershell
uv run python experiments/grading_optimization/run_live.py --config experiments/grading_optimization/config.local.json --limit 50 --repeats 1 --resume results/grading_optimization/TIMESTAMP
```
Configuration and index hash must match; successful query/variant/repeat combinations are skipped. Warmup is rerun. Telemetry appends. Audit errors before drawing conclusions.

## Earlier API pilot
run.py and config.openai.json retain the prepared-evidence/API prototype. OpenAI results are a separate experiment and must not be mixed with local Llama measurements. API keys belong only in terminal environment variables; never commit or send them in chat.
_local dependencies: _deps/, _extras/, _cache/ are ignored and must not be committed._
