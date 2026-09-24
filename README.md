# EduMCTS: reasoning data synthesis

EduMCTS constructs reasoning trajectories by searching over autonomous student
steps and three pedagogical interventions: hint, critique and Socratic question.
Each transition is scored for realized reasoning progress; terminal solutions
are checked against a reference answer. Selected teacher interventions are
preserved as explicit `<reflection type="...">` tags in the training outputs.

This release contains the synthesis pipeline, its input seeds and the final
1,080-example mathematical reasoning dataset.

## Project layout

```text
.
├── main.py                  # Batch synthesis and SFT export
├── config.py                # Model, search and data configuration
├── llm_client.py            # Model client and role prompts
├── mcts.py                  # Search, progress shaping and path selection
├── harvester.py             # Verified-trajectory filtering and formatting
├── run.sh              # Full synthesis launcher
├── .env.example             # API configuration placeholders
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── seed_problems.json    # original input problems
│   └── EduMCTS.json          # retained Alpaca examples
└── tests/                   # Offline tests using mock model responses
```

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The only third-party runtime dependency is the OpenAI Python client. The service
you configure must support OpenAI-compatible chat completions.

## API configuration

```bash
cp .env.example .env
# Edit .env and replace both API placeholders with your own settings.
set -a
source .env
set +a
```

| Variable | Purpose |
| --- | --- |
| `EDUMCTS_API_KEY` | Your API credential; the shipped value is `<YOUR_API_KEY>` |
| `EDUMCTS_BASE_URL` | Your service's API base URL; the shipped value is `<YOUR_API_BASE_URL>` |
| `EDUMCTS_MODEL` | Model identifier; defaults to `qwen3-235b-a22b-instruct-2507` |

The program reads these environment variables; it does not automatically load
`.env`. Unconfigured placeholders are rejected before synthesis starts. No API
credential or service address is included in this release. `.env` and generated
logs are excluded by `.gitignore`.

## Run synthesis

Check the available arguments without calling the API:

```bash
python main.py --help
```

Run a small demo after configuring your API service:

```bash
python main.py --mode demo --problems 1 --rollouts 4 --depth 3 \
  --workers 1 --output-dir output/demo
```

Run the provided input set:

```bash
bash run.sh
```

The launcher uses 20 rollouts per problem, maximum depth 10, UCT coefficient 1.4
and progress weight 0.25. Depth counts student tree steps, including the initial
autonomous step. `WORKERS` controls parallel problems (default: 4); set it to a
concurrency supported by your service. Extra arguments can override the launcher's
defaults, for example:

```bash
WORKERS=2 bash run.sh --problems 10 --output-dir output/small_run
```

To use your own seeds:

```bash
python main.py --mode full --input data/my_problems.json --problems 100 \
  --rollouts 20 --depth 4 --workers 4 --output-dir output/custom
```

The input is a JSON list of objects with `problem` and `answer` fields and an
optional `id`. `--problems N` selects the first N entries; `--problems START END`
selects an inclusive, one-based range. Relative input and output paths resolve
against the project directory, so they also work when the script is launched
from another working directory. The demo and synthesis commands make API calls;
the tests below use mock responses and run offline.

Each run writes to its chosen output directory:

- `training_data_alpaca.json`: instruction/input/output examples with metadata;
- `training_data_sharegpt.json`: human/assistant conversations with metadata;
- `training_data_cot.json`: problem/reasoning/reference-answer examples;
- `raw_search_results.json`: search results and diagnostics for the new run;
- `run.log`: progress messages for the new run.

## Released data

`data/EduMCTS.json` contains 1,080 retained Alpaca examples from a synthesis run
over 1,800 seed problems, with a rollout budget of 20 and maximum depth of 10.
The released outputs and their `_meta` fields are preserved from that run.
`data/seed_problems.json` preserves the corresponding original problems,
reference answers and zero-based IDs; only the fields needed for synthesis are
included. The seeds are the actual inputs to that run, including problems for
which synthesis did not retain a trajectory. They are not a subsequently
corrected reference-answer set.

Alpaca fields:

| Field | Contents |
| --- | --- |
| `instruction` | Instruction to solve the problem step by step |
| `input` | Problem statement |
| `output` | Selected reasoning trajectory, including any reflection tags |
| `_meta` | Search and verification statistics from the synthesis run |

The harvester keeps trajectories marked correct by the terminal verifier,
meeting the minimum search correctness rate (default: 0.05), and ending in a
complete student response with `FINAL ANSWER:`. These are model-verifier labels,
not claims of independent human verification. Generation and search are
stochastic, so a rerun need not reproduce the released outputs byte for byte.

## Offline checks

```bash
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

The tests cover core search and harvesting behavior, placeholder configuration,
portable paths and mocked synthesis. They do not require an API key or access
to the synthesis service.
