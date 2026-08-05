# docsum

Document summarizer for texts too large for an LLM's context window.

Uses token-aware chunking and summarization algorithms (map-reduce, refine, hierarchical) to process documents of any size through any OpenAI-compatible API — including the Hermes proxy, so it works with your existing Nous Portal subscription and any model you're authenticated with.

## Quick Start

```bash
# Install dependencies
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Start the Hermes proxy (if not already running)
hermes proxy start &

# Summarize a document
PYTHONPATH=. .venv/bin/python -m docsum.cli --input transcript.txt --model z-ai/glm-5.2

# Extract themes instead of summary
PYTHONPATH=. .venv/bin/python -m docsum.cli --input transcript.txt --model z-ai/glm-5.2 --prompt themes
```

## Algorithms

| Mode | How It Works | Best For |
|---|---|---|
| `map-reduce` (default) | Chunk → summarize each → combine all summaries | Quick overview, parallelizable chunks |
| `refine` | Chunk → summarize → refine with next chunk → repeat | Narrative content (transcripts, stories) |
| `hierarchical` | Map-reduce with recursive reduction when summaries are still too large | Very long documents, structured content |

## Built-in Prompts

| Prompt | What It Extracts |
|---|---|
| `summary` (default) | Key ideas and important details |
| `themes` | Main themes, motifs, and central ideas |
| `characters` | Characters, roles, actions, relationships |
| `key_events` | Events and plot points in order |

## Usage

```
docsum --input FILE --model MODEL [options]

Required:
  --input, -i FILE          Input text file
  --model, -m MODEL         Model ID (e.g., z-ai/glm-5.2)

LLM Connection:
  --base-url URL           API endpoint (default: http://127.0.0.1:8645/v1 — Hermes proxy)
  --api-key KEY            API key (default: "proxy" — any string works with Hermes proxy)

Algorithm:
  --mode MODE              map-reduce | refine | hierarchical (default: map-reduce)
  --max-tokens N           Max tokens per chunk (default: 2000)
  --overlap-tokens N       Token overlap between chunks (default: 0)

Prompt:
  --prompt NAME            Built-in: summary | themes | characters | key_events (default: summary)
  --prompt-file FILE       Custom prompt template with {text} placeholder

Output:
  --output, -o FILE       Write to file (default: stdout)

Other:
  --tokenizer-model MODEL  Model for token counting (default: gpt-4)
```

## Examples

### Summarize a Star Trek TNG transcript

```bash
# Basic summary with map-reduce
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input episode.txt --model z-ai/glm-5.2 --mode map-reduce

# Extract themes
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input episode.txt --model z-ai/glm-5.2 --prompt themes

# Extract characters
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input episode.txt --model z-ai/glm-5.2 --prompt characters
```

### Use a small model (Nemotron)

```bash
# Nemotron has a small context window — use smaller chunks
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input large_doc.txt \
  --model nvidia/nemotron-3-super-120b-a12b \
  --max-tokens 500
```

### Custom prompt

Create a file `my_prompt.txt`:
```
Extract all dates and timeline events from the following text. 
List them chronologically.

{text}
```

```bash
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input report.txt --model z-ai/glm-5.2 \
  --prompt-file my_prompt.txt
```

### Use refine mode for narrative content

```bash
PYTHONPATH=. .venv/bin/python -m docsum.cli \
  --input story.txt --model z-ai/glm-5.2 \
  --mode refine --overlap-tokens 100
```

## Testing

```bash
# Unit tests (fast, no LLM calls)
PYTHONPATH=. .venv/bin/python -m pytest

# Integration tests (calls real LLM via Hermes proxy, costs tokens)
PYTHONPATH=. .venv/bin/python -m pytest -m integration
```

## Project Layout

```
docsum/
├── docsum/
│   ├── __init__.py
│   ├── chunker.py       # Token-aware text chunking
│   ├── prompts.py        # Built-in and custom prompt templates
│   ├── llm_client.py     # OpenAI-compatible API client
│   ├── algorithms.py     # map_reduce, refine, hierarchical
│   └── cli.py           # Command-line interface
├── tests/
│   ├── test_chunker.py       # 13 tests
│   ├── test_prompts.py       # 12 tests
│   ├── test_llm_client.py     # 5 tests
│   ├── test_algorithms.py    # 17 tests
│   ├── test_cli.py           # 7 tests
│   └── test_integration.py   # 4 integration tests (Nemotron via proxy)
├── requirements.txt
├── pytest.ini
└── README.md
```

## License

Unlicense (public domain).
