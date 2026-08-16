# Multi-Agent ACSA Annotation System

A human-in-the-loop, multi-agent pipeline for Vietnamese Aspect Category Sentiment Analysis (ACSA). Two annotators label reviews independently; disagreements are resolved through debate and a judge panel. In guideline-update mode, the system also analyzes root causes and proposes a reviewed update to the active annotation guideline.

## Workflow

1. Load and normalize one review per line from the local input dataset.
2. Load an existing guideline or adapt one to a target domain.
3. Build a local Chroma vector database from the guideline.
4. Run two annotators and store immediately agreed labels.
5. Route disagreements through debate, summary, and judge agents.
6. Optionally analyze conflict causes and ask a human to approve a guideline update.
7. Save progress and generated artifacts under `system_data/`.

## Repository layout

```text
agents/               Agent implementations
config/               LLM and embedding configuration
core_engine/          Data loading and workflow orchestration
data/guideline.txt    Default ACSA guideline
memory_and_history/   Debate history management
models/               Shared data schemas
prompts/              Prompt templates
rag_system/           Guideline and agreed-case retrieval
utils/                Preprocessing, retry, and usage helpers
main.py               Interactive pipeline entry point
```

Datasets, API keys, checkpoints, vector databases, and generated results are intentionally not tracked.

## Requirements

- Python 3.10 or newer
- An Azure OpenAI resource with compatible chat deployments
- Internet access on the first run to download `intfloat/multilingual-e5-small`

The current environment was validated with Python 3.13. A CUDA GPU is not required; embeddings run on CPU by default.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/Namem2006/Multi_Agent_Paper.git
cd Multi_Agent_Paper
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux, use `cp .env.example .env`. Then fill in at least:

```dotenv
OPENAI_API_KEY=...
BASE_URL=https://your-resource.openai.azure.com
API_VERSION=2024-08-01-preview
DEPLOYMENT_NAME=your-azure-deployment-name
```

Never commit `.env` or real API keys.

## Add input data

Create `data/Data_Foody_Final.txt`. Put exactly one raw review on each non-empty line, for example:

```text
Phòng sạch sẽ và nhân viên thân thiện.
Đồ ăn ngon nhưng phục vụ hơi chậm.
```

At startup, the pipeline converts this file into the ID-based format expected by the agents. The prepared file and all other dataset files remain ignored by Git.

To use a different domain guideline, replace `data/guideline.txt` or select the Adapt Agent when prompted.

## Run

From the repository root:

```bash
python main.py
```

On the first cycle, choose the guideline source and target domain. Then choose one of two modes:

- **Annotation-only** processes a selected number of samples and skips root-cause analysis and guideline updates.
- **Guideline-update** processes one fixed cycle, resolves conflicts, and opens a human review before appending an approved rule.

Start with a small batch because every sample can require multiple paid LLM calls. The pipeline saves progress after each completed cycle.

## Generated files

Runtime artifacts are written to `system_data/`, including:

- `progress.json`: last completed sample and active guideline
- `agreed_samples.jsonl`: samples where annotators agreed
- `conflict_samples.jsonl`: samples routed to debate
- `result/`: final per-sample decisions
- `cause/`: root-cause and guideline-update proposals
- `chroma_db*`: local vector databases
- `llm_token_usage*.json*`: token-usage logs

These files are reproducible run outputs and are excluded from version control.

To restart from the first sample, remove `system_data/progress.json`. To rebuild all local state, remove the entire `system_data/` directory.

## Security

- Keep all credentials in `.env`.
- Revoke and rotate a key immediately if it is ever committed or shared.
- Review estimated API usage before processing a large dataset.
