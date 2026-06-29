# FacultyAI

AI-driven async CLI tool that autonomously scrapes university faculty information. Given a university name and optionally a department, it discovers the faculty listing page, extracts structured data, and writes results to an Excel file.

## Quick Start

```bash
pip install -e .
playwright install chromium

# Set API key
export OPENAI_API_KEY="sk-..."

# Validate and run
facultyai config validate
facultyai run
```

## How It Works

```
universities.xlsx ──► Input sync ──► SQLite DB
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    Discovery Jobs   Scrape Jobs   Resume/Retry
                          │             │
                          ▼             ▼
                    LangGraph Agent (checkpointed)
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
          DuckDuckGo   LLM call   ScrapeGraphAI / Playwright
                │         │         │
                └─────────┼─────────┘
                          ▼
                    faculty_data.xlsx
```

## Configuration (`config.yaml`)

```yaml
llm:
  provider: deepseek          # openai, azure, anthropic, google, local
  model: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com/v1

search:
  provider: duckduckgo        # or bing (needs api_key)

scraping:
  max_concurrent_jobs: 3      # parallel scraping
  max_retries_per_step: 3     # auto-retry on failure

department:
  discovery_enabled: true     # auto-discover departments
```

## Input File (`universities.xlsx`)

Sheet `universities` with columns:

| university_name | department_name | extra_info |
|---|---|---|
| MIT | Electrical Engineering & CS | |
| Stanford | | discover all |
| UC Berkeley | Computer Science | berkeley.edu/cs |

Blank `department_name` → auto-discovers all departments.
`extra_info` → search hints.

## Schema File (`schema.json`)

```json
{
  "columns": [
    {"name": "Full Name",   "type": "extracted", "hint": "Professor's full name"},
    {"name": "Last Name",   "type": "formula",   "formula": "=TEXTAFTER([@[Full Name]],\" \")"},
    {"name": "Email",       "type": "extracted", "hint": "Email address"},
    {"name": "Institution", "type": "static",    "value_from": "university_name"}
  ]
}
```

- **extracted** — scraped by AI (`hint` guides the prompt)
- **formula** — Excel formula using `[@[Column Name]]` references
- **static** — constant value or `value_from` university name

## Commands

| Command | Description |
|---|---|
| `facultyai run` | Run all pending jobs |
| `facultyai run --retry-failed` | Run pending + retry failed jobs |
| `facultyai resume` | Resume after interruption |
| `facultyai resume --retry-failed` | Resume + retry failed |
| `facultyai retry MIT EECS` | Retry a specific job |
| `facultyai status` | Job statuses + run history |
| `facultyai export` | Regenerate output Excel |
| `facultyai chat` | Interactive REPL |
| `facultyai config validate` | Validate config |
| `facultyai config show` | Show masked config |

## Typical Workflow

```bash
facultyai run                    # first run
facultyai status                 # check progress
facultyai run --retry-failed     # retry failures
facultyai resume                 # recover from crash
facultyai export                 # manual export
```

## Chat REPL

`facultyai chat` — slash commands: `/list`, `/jobs`, `/schema`,
`/add-col`, `/export`, `/config`, `/help`. Natural language also works.

## Resume & Crash Recovery

LangGraph SQLite checkpointing persists state after each node to `facultyai.db`.
`facultyai resume` picks up from the last checkpoint — no duplicate work.

## LLM Providers

| Provider | Config notes |
|---|---|
| DeepSeek | `provider: deepseek`, set `base_url` |
| OpenAI | `provider: openai`, set `api_key` |
| Azure | `provider: azure`, set `azure_endpoint` |
| Anthropic | `provider: anthropic` |
| Google | `provider: google` |
| Local/Ollama | `provider: openai_compatible`, `base_url: http://localhost:11434/v1` |

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All settings |
| `universities.xlsx` | Input list |
| `schema.json` | Output column definitions |
| `faculty_data.xlsx` | Output spreadsheet |
| `facultyai.db` | SQLite database (source of truth) |
| `cache/` | Page and LLM response cache |

## Troubleshooting

| Problem | Solution |
|---|---|
| "Another process is running" | Delete `facultyai.lock` |
| API key errors | Check env var or `config.yaml` |
| No search results | Add `request_delay_sec` or switch to Bing |
| No faculty extracted | Add `extra_info` hints in Excel input |
| Excel formula broken | Verify column names match `schema.json` |
| ScrapeGraphAI error | Set `use_scrapegraphai: false` in config |

## Requirements

- Python 3.11+
- Chromium browser (auto-installed by `playwright install chromium`)
- API key for an LLM provider (or a local model)

## License

MIT
