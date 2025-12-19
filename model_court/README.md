# Model Court

Court-style multi-agent simulation to evaluate a candidate model, produce adversarial critique + defense + judge ruling, and persist a reusable reasoning dataset (canonical JSONL) + a local vector database for retrieval and analysis.

## 1) Create From Scratch (PowerShell)

```powershell
# from your workspace root
New-Item -ItemType Directory -Force model_court | Out-Null
Set-Location model_court
git init

# folders
New-Item -ItemType Directory -Force `
  configs, datasets/raw, datasets/golden, scripts, tests, runs, `
  src/model_court, src/model_court/agents/prompts, src/model_court/llm, `
  src/model_court/question_gen, src/model_court/simulation, src/model_court/storage, `
  src/model_court/vector_store, src/model_court/evals, src/model_court/utils, src/model_court/datasets | Out-Null

# stubs (filled in by this repo)
New-Item -ItemType File -Force pyproject.toml, requirements.txt, README.md, configs/default.yaml | Out-Null
```

## 2) Roles + Contracts (JSON I/O)

- Candidate (initial answer): `CandidateAnswer` (`src/model_court/agents/contracts.py`)
- Prosecutor (critique): `ProsecutorCritique` (`src/model_court/agents/contracts.py`)
- Defense (rebuttal + optional improved answer): `DefenseRebuttal` (`src/model_court/agents/contracts.py`)
- Judge (verdict + rubric scoring): `JudgeDecision` (`src/model_court/agents/contracts.py`)

Prompts live in `src/model_court/agents/prompts/*.txt` and are schema-constrained: each agent must output a single JSON object.

## 3) Repository Layout

```
model_court/
  configs/               # YAML configs (models, rounds, thresholds)
  datasets/
    raw/                 # experiment JSONL dumps
    golden/              # filtered exports (high-confidence accepts)
  runs/                  # per-experiment vector DBs
  scripts/               # CLI wrappers
  src/model_court/
    agents/              # prompts + wrappers + contracts
    llm/                 # provider interface + mock + optional OpenAI skeleton
    question_gen/        # synthetic generators + benchmark adapters
    simulation/          # orchestrator + stopping logic
    storage/             # canonical case schema + JSONL store
    vector_store/        # embedder + sqlite index + retrieval
    datasets/            # golden export logic
    utils/               # config/time/ids
  tests/                 # schema + orchestrator smoke tests
```

## 4) Questions At Scale

- Synthetic generation: `SyntheticMathV1` (`src/model_court/question_gen/synthetic.py`) is a runnable example; add additional generators per domain.
- Benchmark sampling: `JSONLBenchmarkAdapter` (`src/model_court/question_gen/benchmark.py`) reads a local JSONL benchmark dump (adapter interface is `QuestionSource`).
  - Example conversion pipeline (outside this repo): export GSM8K/TruthfulQA into JSONL with fields `id`, `question`, `answer`, `metadata`, then point config to it.

## 5) Simulation Loop (Runnable)

Pseudocode:
- `questions = source.sample(n, seed)`
- for each `q`:
  - `candidate = Candidate(q, prior_rounds)`
  - `prosecutor = Prosecutor(q, candidate)`
  - `defense = Defense(q, candidate, prosecutor)`
  - `judge = Judge(q, candidate, prosecutor, defense, rubric)`
  - stop if `judge.verdict == accept` else iterate up to `max_rounds`

Implementation: `src/model_court/simulation/orchestrator.py` (`run_experiment`, `_run_case`).

## 6) Artifact Storage (Canonical JSONL)

Each line in `datasets/raw/<experiment_id>/cases.jsonl` is a `CaseRecord` (`src/model_court/storage/schema.py`) containing:
- `question` (id/text/answer_key/source/metadata)
- per-round transcript: `candidate`, `prosecutor`, `defense`, `judge`
- raw LLM call records per role: prompt, response, parsed JSON (`LLMCallRecord`)
- `final_answer`, `final_verdict`, `final_overall_score`
- `metadata.config_snapshot`, timestamps, seeds, experiment ids

## 7) Vector DB Integration (SQLite)

- Embedding strategy: embed `question`, `final_answer`, `prosecutor.objections`, `judge.justification`; chunk by `chunk_chars` with `chunk_overlap`.
- Embedder: `HashingEmbedder` (deterministic local baseline) (`src/model_court/vector_store/embedder.py`).
- DB schema: `vectors(id, content, metadata_json, embedding, dim)` (`src/model_court/vector_store/sqlite_store.py`).
- Indexing: `CaseIndexer.index_case(case)` (`src/model_court/vector_store/indexing.py`).
- Retrieval use cases:
  - Similar past cases by question/objections (failure pattern mining)
  - Judge consistency checks by retrieving near-duplicate questions and comparing verdicts
  - Training data selection by pulling high-scoring, high-confidence “golden” cases

## 8) Golden Dataset Creation

Golden export filters cases with rules (`src/model_court/datasets/golden.py`):
- `final_verdict == "accept"`
- `final_overall_score >= accept_threshold`
- final-round `judge.confidence >= confidence_threshold`
- optional: `defense.unresolved_objections` must be empty

Export format: JSONL of full `CaseRecord` (versioned by `schema_version="case_v1"`). Parquet export can be added as an optional dependency later.

## 9) Iterative Improvement (“Court Curriculum”)

Recommended loop (config-driven):
- Mine failure tags + unresolved objections from `cases.jsonl`
- Generate targeted follow-up questions (harder variants) with a generator per tag
- Update prompts/rubric weights and re-run the court
- Gate changes with automated checks: accept-rate on fixed seed set, judge consistency on near-duplicates, regression reports

## How To Run

### Install

```powershell
cd model_court
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Small demo (mock LLMs, no keys)

```powershell
model-court-demo --config configs/default.yaml --n 3
```

### Larger run

```powershell
model-court-run --config configs/default.yaml --n 50
```

### Export golden

```powershell
model-court-export-golden --cases datasets/raw/<experiment_id>/cases.jsonl --out datasets/golden/golden_v1/cases.jsonl
```

### Query similar past cases

```powershell
model-court-query --db runs/<experiment_id>/vectors.sqlite --text "word problem multiplication" --k 5
```
