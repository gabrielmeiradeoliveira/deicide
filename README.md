# DEICIDE

**DEICIDE** is an open-source, human-controlled, modular AI software assurance and code intelligence platform. Its local CLI engine inspects Git changes, retrieves bounded surrounding context, identifies concrete correctness bugs, validates findings against actual changed lines and source evidence, deduplicates overlapping reports, enforces persistent API rate and daily quotas, and outputs terminal or machine-readable JSON reports.

---

## Architecture & Workflow

```text
Git Repository
      |
      v
Detect Changes (GitAnalyzer)
      |
      v
Extract Diff & Line Mapping (DiffBundle)
      |
      v
Retrieve Bounded Source Context (FileContext)
      |
      v
Atomic Quota & Budget Reservation (BudgetTracker + SQLite UsageStore)
      |
      v
AI Code Review Module (module_id: "code_review")
      |
      v
Reconcile Actual Token Usage & Retries
      |
      v
Validate Findings (Schema, Changed Lines, Verbatim Evidence)
      |
      v
Deduplicate Findings (Stable IDs & Overlap Resolution)
      |
      v
Terminal / JSON Report
```

### Core Components (`src/deicide/`)

- **`deicide.core`**: Configuration (`DeicideConfig`), per-analysis and persistent execution budgets (`ExecutionBudget`, `BudgetTracker`), shared schemas (`Finding`, `CandidateFinding`, `AnalysisReport`), validation and deduplication (`validation.py`), module registry (`ModuleRegistry`), and stateless orchestration (`AnalysisOrchestrator`).
- **`deicide.quota`**: Provider-independent SQLite usage ledger (`UsageStore`, `QuotaSnapshot`, `RequestStatus`), atomic `BEGIN IMMEDIATE` request reservations, stale reservation expiration, rolling 60-second (`RPM`/`TPM`) and UTC daily (`RPD`) quota windows, and API key scope isolation.
- **`deicide.git`**: Safe read-only Git repository detection, diff extraction, unified diff hunk/line mapping, and bounded working-tree context retrieval (`GitAnalyzer`) with path traversal and symlink-escape protection.
- **`deicide.providers`**: Provider-neutral `LLMProvider` protocol, deterministic offline `FakeProvider`, and production-grade `NvidiaProvider` (using Python's standard library `urllib.request`, per-attempt quota reservation, bounded retries, `Retry-After` handling, request timeouts, and token usage reconciliation).
- **`deicide.modules.code_review`**: First analysis module (`module_id = "code_review"`), decoupled from provisional mythological display names. Focuses strictly on functional correctness bugs rather than stylistic nitpicks.
- **`deicide.reporting`**: Human-readable terminal formatters and strict machine-readable JSON formatters for both review reports and usage/quota snapshots.
- **`deicide.evaluation`**: Offline evaluation runner (`evaluate_corpus`) measuring True Positives, False Positives, False Negatives, Precision, Recall, False Positive Rate, Latency, and Token Consumption against curated fixtures (`evaluations/fixtures/code_review_corpus.json`).

---

## Installation

Requires **Python 3.12+**:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Configuration

DEICIDE is configured via CLI flags, environment variables, or a `.env` file in the DEICIDE application directory. **DEICIDE never reads `.env` or configuration files from a repository being analyzed.**

Copy `.env.example` to `.env` and configure your provider, model, API key, and quota limits:

```dotenv
DEICIDE_PROVIDER=nvidia
DEICIDE_MODEL=z-ai/glm-5.3-flash

DEICIDE_MAX_REQUESTS=3
DEICIDE_MAX_TOKENS=8000

DEICIDE_QUOTA_ENABLED=true
DEICIDE_QUOTA_SCOPE=nvidia-glm

NVIDIA_API_KEY=your_nvidia_api_key_here
DEICIDE_ALLOW_EXTERNAL_CODE=1
```

| Environment Variable | Default | Description |
|---|---|---|
| `DEICIDE_PROVIDER` | `nvidia` | AI provider (`nvidia` or `fake`). |
| `DEICIDE_MODEL` | `z-ai/glm-5.3-flash` (or `fake-deterministic-v1`) | Model name override. |
| `DEICIDE_MAX_REQUESTS` | `3` | Maximum API request attempts allowed per analysis run. |
| `DEICIDE_MAX_TOKENS` | `8000` | Maximum total tokens allowed per analysis run. |
| `DEICIDE_MAX_RETRIES` | `0` | Maximum automatic retries on transient HTTP 429/5xx or timeout (`0` = 1 initial attempt, no automatic retries). |
| `DEICIDE_RATE_LIMIT_RPM` | *None* | Persistent requests-per-minute limit (rolling 60s window). |
| `DEICIDE_RATE_LIMIT_TPM` | *None* | Persistent input-tokens-per-minute limit (rolling 60s window). |
| `DEICIDE_RATE_LIMIT_RPD` | *None* | Persistent requests-per-day limit (UTC calendar day). |
| `DEICIDE_QUOTA_ENABLED` | `true` | Enable persistent SQLite quota enforcement (`true`/`false`). |
| `DEICIDE_QUOTA_SCOPE` | Key SHA-256 fingerprint | Optional shared quota bucket name across keys in the same project. |
| `DEICIDE_USAGE_DB_PATH` | OS user data dir | Optional custom path for the SQLite usage database. |
| `NVIDIA_API_KEY` / `DEICIDE_API_KEY` | *None* | API key for NVIDIA API (resolved with precedence: process `NVIDIA_API_KEY` > process `DEICIDE_API_KEY` > trusted `.env` `NVIDIA_API_KEY` > trusted `.env` `DEICIDE_API_KEY`). |
| `DEICIDE_ALLOW_EXTERNAL_CODE` | `0` | Explicit authorization to send repository diffs to an external AI provider (`1` to authorize). |
| `DEICIDE_TIMEOUT_SECONDS` | `60.0` | Per-request HTTP socket timeout in seconds passed directly to the HTTP transport (`--timeout` on CLI). |

---

## Persistent Quota & Usage Tracking

- **Database Location**: Stored outside analyzed repositories in the OS user data directory:
  - **Windows**: `%LOCALAPPDATA%\deicide\usage.sqlite3`
  - **macOS**: `~/Library/Application Support/deicide/usage.sqlite3`
  - **Linux**: `${XDG_DATA_HOME:-~/.local/share}/deicide/usage.sqlite3`
- **Privacy**: The SQLite ledger stores only `provider`, `model`, `quota_scope` (a non-reversible truncated SHA-256 hash when derived from an API key), UTC timestamps, request lifecycle status, and token counts. **API keys, prompts, diffs, filenames, source code, and model responses are never stored.**
- **Window Calculations & Quota Limitations**:
  - **Daily Window (`RPD`)**: Calculated per UTC calendar day (`00:00:00Z` through `23:59:59Z`). Note that a provider's server-side daily quota may reset on a different timezone (such as Pacific Time) or rolling window.
  - **Per-Minute Windows (`RPM` & `TPM`)**: Calculated over a rolling 60-second window (`[now_utc - 60s, now_utc]`).
  - **Local Estimation**: Usage tracked in SQLite reflects requests initiated by this local DEICIDE installation. Requests made by other tools using the same API key/project are not visible locally.
  - **Uncertain Network Failures & Retries**: Every retry that reaches the provider reserves and consumes an attempt. Network timeouts where it is unknown whether the provider received the request are recorded as `uncertain` and are conservatively retained against quota rather than refunded. Crashed reservations automatically expire after `reservation_ttl_seconds` (120s).

---

## Usage

### 1. Show CLI help

```bash
deicide --help
```

### 2. Inspect persistent API usage and remaining quota (`deicide usage`)

```bash
# Human-readable terminal report:
deicide usage

# Machine-readable JSON report:
deicide usage --format json
```

### 3. Analyze local uncommitted changes (`deicide review`)

By default, `deicide review` inspects all local uncommitted changes in the current Git repository, including **both staged and unstaged changes**:

```bash
# Offline deterministic review (no API key required; isolated from external quota):
deicide review --provider fake

# Real review using NVIDIA (requires NVIDIA_API_KEY and explicit external authorization):
deicide review --provider nvidia --allow-external
```

### 4. Compare against a specific Git revision (`--base`)

```bash
deicide review --base HEAD~1 --provider fake
```

### 5. Structured JSON output (`--format json`)

```bash
deicide review --provider fake --format json
```

### 6. Run the evaluation benchmark (`deicide evaluate`)

```bash
deicide evaluate --provider fake
```

---

## Process Exit Codes

| Exit Code | Meaning |
|---|---|
| `0` | Command completed successfully |
| `1` | Unexpected internal error |
| `2` | Configuration error (missing credentials, unsupported provider, invalid flags/limits) |
| `3` | Repository or Git error (not a Git repository, invalid revision) |
| `4` | AI provider failure or exhausted rate-limit retries |
| `5` | Execution budget or persistent API quota (`RPD`/`RPM`/`TPM`) exceeded |
| `6` | Security or trust boundary violation (path traversal, symlink escape) |
