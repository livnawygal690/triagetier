# Repository plan

## Layout

The package is organized as compiler-style stages: a case is lowered from raw inputs
through the stages to a triage verdict (`ir` is the shared record base; `toolchain` is the
top-level command surface).

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Dockerfile
├── .gitignore
├── settings/
│   ├── model/      qwen25vl7b.yaml qwen25vl3b.yaml internvl25_8b.yaml llava_med.yaml mock.yaml
│   ├── data/       ratic.yaml cq500.yaml mimic_iv_ed.yaml synthetic.yaml
│   ├── pipeline/   default.yaml
│   └── experiment/ main.yaml ablation_*.yaml supplementary_edge_*.yaml slice_*.yaml _smoke.yaml
├── triagetier/
│   ├── ir/       records.py
│   ├── frontends/      registry.py ratic.py cq500.py mimic_iv_ed.py synthetic.py
│   ├── lowering/    localize.py sampling.py windowing.py
│   ├── oracles/     base.py factory.py qwen25vl.py internvl25.py llava_med.py awq.py imaging.py mock.py
│   ├── passes/ base.py detection.py grading.py aggregation.py triage.py qa.py
│   ├── lexicon/    graph.py loader.py ais.py iss.py esi.py confidence.py assets/{aast,iss_regions,esi}.json
│   ├── driver/   orchestrator.py react.py text_reading.py
│   ├── reports/      stats.py detection_eval.py triage_eval.py
│   ├── host/        config.py seed.py logsetup.py artifacts.py
│   └── toolchain/       run.py evaluate.py inspect.py _common.py
├── scripts/         launch_run.sh launch_eval.sh prepare_data.sh
├── tests/
└── docs/            project-context.md implementation-map.md repo-plan.md deviations.md
```

## Stage responsibilities

- **ir/** — the typed records that flow between stages (Case, OrganFinding,
  AISGrade, PartialISS, TriageDecision, CaseResult). No logic, no dependencies.
- **frontends/** — adapters yielding cases (volume or clinical text + ground truth);
  the registry resolves a config name to an adapter.
- **lowering/** — converts a 3-D CT volume into the small set of multi-window
  tiled 2-D images the visual agents consume. Pure array math; numpy only.
- **oracles/** — uniform `generate` interface over Qwen2.5-VL, InternVL2.5, and
  LLaVA-Med (lazy HF imports), AWQ loading, DICOM/NIfTI I/O, and a deterministic mock
  for tests.
- **passes/** — one class per ATLS reasoning step plus the QA cross-checker.
  Each agent owns a prompt builder and a parser that maps backend text to a typed
  record.
- **lexicon/** — the clinical substrate: JSON rules (AAST, ISS regions, ESI) with
  a typed accessor, and the deterministic arithmetic (AAST→AIS, Baker ISS, ESI logic,
  confidence aggregation). No model calls; fully unit-testable.
- **driver/** — the orchestrator that wires agents + QA into a `CaseResult`,
  the ReAct loop, and the text-mode pathway.
- **reports/** — AUC/CI, triage accuracy, ISS/κ/ECE, and the significance tests.
- **host/** — config loading (OmegaConf), single seeding entry point, logging
  setup, atomic run-manifest writes.
- **toolchain/** — `run` (one case / dataset), `evaluate` (metrics + stats), `inspect`
  (dump slice selection / knowledge lookups for a case).

## Dependencies (pinned ranges)

Core (CI, CPU): `numpy>=1.26,<2.1`, `scipy>=1.11,<1.14`, `scikit-learn>=1.3,<1.6`,
`omegaconf>=2.3,<2.4`, `pillow>=10.0,<11.0`; dataclasses only (no pydantic).

Inference extras (optional group `gpu`): `torch>=2.1,<2.5`,
`transformers>=4.49,<4.52` (Qwen2.5-VL support), `accelerate>=0.30`,
`autoawq>=0.2.5`, `qwen-vl-utils>=0.0.8`, `einops`.

Imaging extras (optional group `imaging`): `pydicom>=2.4`, `nibabel>=5.2`,
`SimpleITK>=2.3`.

Dev: `pytest>=8.0`, `ruff>=0.6`, `black>=24.0`, `isort>=5.13`, `mypy>=1.11`,
`pre-commit>=3.7`.

## Expected test coverage

- **lexicon/** — exhaustive unit coverage (ISS, AIS mapping, ESI, confidence) +
  the §2.7 golden regression + knowledge schema/lookup tests.
- **lowering/** — geometry/pixel-math unit + shape/contract tests.
- **reports/stats.py** — cross-checked against scipy/sklearn references.
- **driver/** — one end-to-end integration test (MockVLM) covering the full
  four-agent + QA pipeline on the synthetic case.
- Targeted floor: every public function in `lexicon`, `lowering`,
  `reports` exercised; pipeline covered end-to-end on the synthetic case.
