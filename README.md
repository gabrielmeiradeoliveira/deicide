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
- **`deicide.context`**: Deterministic repository context engine (MVP 1.2) featuring secure Python AST symbol extraction (`PythonAstAnalyzer`), cross-file import and call-site dependency indexing (`DependencyGraph`), depth-based context ranking, token-budgeted snippet slicing, and deletion-only diff symbol recovery.
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
| `DEICIDE_CONTEXT_ENABLED` | `true` | Enable or disable AST-based cross-file context engine (`--context` / `--no-context`). |
| `DEICIDE_CONTEXT_MAX_TOKENS` | `2000` | Maximum token budget allocated for supplementary repository context (`--context-tokens`). |
| `DEICIDE_CONTEXT_MAX_FILES` | `8` | Maximum number of supplementary context files to select (`--context-files`). |
| `DEICIDE_CONTEXT_MAX_SNIPPETS` | `16` | Maximum number of supplementary code snippets to select (`--context-snippets`). |
| `DEICIDE_CONTEXT_MAX_SNIPPET_LINES` | `60` | Maximum line count per sliced snippet before truncation (`60`). |
| `DEICIDE_CONTEXT_MAX_GRAPH_DEPTH` | `2` | Maximum traversal depth in the dependency graph (`2`). |

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

## Context Engine (MVP 1.2)

DEICIDE MVP 1.2 introduces an autonomous, deterministic, AST-powered repository context engine. When reviewing code changes, the engine maps modified lines to syntactic symbols, traces cross-file references across the repository, deterministically ranks relevant context, and fits extracted snippets into a strict token budget alongside the diff.

### Capabilities

1. **Python AST Symbol Extraction**:
   - Parses modified and repository Python files using the standard library `ast` module.
   - Extracts top-level and nested classes, methods, synchronous functions, and `async def` coroutines.
   - Accurately tracks exact line spans (`start_line`, `end_line`), parent scopes, and qualified names (e.g., `Class.method`).
2. **Cross-File Dependency Discovery**:
   - Bounded discovery of repository Python files with symlink-escape and size protections.
   - Discovers import statements (`import foo`, `from foo.bar import baz, qux as q`).
   - Identifies intra-file and cross-file call sites, resolving them against the indexed module symbol table.
   - Distinguishes confirmed `RESOLVED` relationships from `AMBIGUOUS` or `UNRESOLVED` references, never fabricating hallucinated graph targets.
3. **Deterministic Ranking**:
   - Scores candidate contexts using deterministic priority ordering: direct callers/callees > imported definitions > class-associated methods > module-level definitions > transitive dependencies.
   - Breaks ties deterministically by relationship depth, file path, and start line, avoiding non-deterministic hashing or ordering.
4. **Token-Budgeted Snippet Selection**:
   - Slices exact source code ranges with safety boundaries (`max_snippet_lines`, `max_context_files`, `max_snippets`).
   - Dynamically checks available token headroom against the overall run budget and baseline prompt tokens.
   - Omitted candidates are explicitly recorded with reasons (e.g., `token_budget_exceeded`, `file_limit_exceeded`, `duplicate`).
5. **Deletion-Only Diff Detection**:
   - Accurately maps symbol boundaries in diff hunks where code was purely deleted (`-` lines without `+` lines).
   - Recovers caller and dependent context even when functions, methods, or entire files are deleted.

### CLI Options

The Context Engine is enabled by default and can be configured or disabled via CLI options:

- `--context` / `--no-context`: Enable or disable the AST cross-file context engine (default: enabled).
- `--context-tokens <int>`: Set the maximum token budget allocated for supplementary repository context (default: 2000).
- `--context-files <int>`: Set the maximum number of supplementary context files to select (default: 8).
- `--context-snippets <int>`: Set the maximum number of supplementary code snippets to select (default: 16).

### Known MVP 1.2 Limitations

- **Python Only**: The AST parser and dependency graph currently support Python source files (`.py`). Multi-language AST support is planned for future MVPs.
- **Static Resolution Only**: Symbol resolution relies on deterministic static analysis. Dynamic runtime behaviors (such as `getattr()`, `importlib`, dynamic `*args` / `**kwargs` dispatch, or monkey-patching) are not dynamically executed.
- **Single-Repository Boundary**: Context extraction is strictly bounded to the target Git repository root. External third-party packages in virtual environments or system paths are omitted from snippet extraction.

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

### 4. Configure AST context selection

By default, cross-file AST context discovery is automatically performed. You can tune context limits or disable context extraction entirely:

```bash
# Review with customized context token and snippet limits:
deicide review --provider fake --context-tokens 1500 --context-files 4 --context-snippets 8

# Review with AST context engine disabled (diff only):
deicide review --provider fake --no-context
```

### 5. Compare against a specific Git revision (`--base`)

```bash
deicide review --base HEAD~1 --provider fake
```

### 6. Structured JSON output (`--format json`)

```bash
deicide review --provider fake --format json
```

### 7. Run the evaluation benchmark (`deicide evaluate`)

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
