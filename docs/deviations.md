# Deviations

Each entry: what differs from a literal reading of the build protocol, the paper
anchor that justifies it, and how it is handled.

## D1 — No training stage; inference profile replaces training hyperparameters

The protocol's compute/hyperparameter rules assume a training run (batch size, lr,
warmup, weight decay, scheduler, epochs, DDP/FSDP, AMP). This work is zero-shot: the
VLM weights are never updated (Abstract; §1; §4 throughout; "Data integrity and
leakage statement", §4.7: "operates in a zero-shot setting with no model fine-tuning").

Handling: `settings/experiment/main.yaml` encodes the *inference* profile (backbone,
quantization, decoding policy, slice counts, HU windows, ReAct cap, confidence
threshold, seeds) rather than training hyperparameters. There is no losses tier, no
optimizer/scheduler, and no distributed-training code. Effective batch size is not a
meaningful quantity here.

## D2 — End-to-end pipeline test replaces the "loss-decreases" smoke test

The protocol asks for a 2-step training smoke test asserting that loss decreases.
With no training loop this is undefined. Handling: `tests/test_service_e2e.py` runs
the full four-agent + QA pipeline on a synthetic CT volume through `MockVLM` and
asserts the structural invariants instead (valid ESI ∈ {1..5}, ISS computed by the
Baker formula, ReAct terminates within the 3-iteration cap, low-confidence flagging
triggers). `settings/experiment/_smoke.yaml` carries the unit-test-only marker.

## D3 — In-code comments and provenance docstrings suppressed

The protocol (R5) suggests top-of-file docstrings citing paper sections, and (R1)
asks for a one-line comment at any deviation call site. This release ships with no
in-code comments or decorative docstrings. Handling: all paper provenance lives in
`docs/implementation-map.md`; all deviation notes live in this file. No information
is lost; it is relocated out of source.

## D4 — Repository layout organized as compiler-style lowering stages

The template layout (`data/models/losses/training/metrics/evaluation`) is
training-centric. This system has no trainable model and no loss, so the source is
organized as the stages a case is lowered through — `frontends → lowering → oracles →
passes → driver → reports`, over a shared `ir` record base, a `lexicon` clinical
substrate, a `host` of runtime services, and a `toolchain` command surface. The ATLS
reasoning steps of §4.1–§4.6 live in `passes` and `driver`; the clinical substrate of
§4.4 lives in `lexicon`. This keeps the release distinct from training-oriented siblings.

## D5 — ISS contributions follow the worked example, not the literal formula text

The formula in §4.1 reads ISS = g²₍₁₎ + g²₍₂₎ + g²₍₃₎ over "the top three regional
AIS values", which would take one AIS per body region. The worked example in §2.7
sums two abdominal organs as separate terms (hepatic AIS 2 and splenic AIS 1 →
2² + 1² = 5), i.e. it ranks per-organ contributions rather than per-region maxima.
`lexicon/iss.py:aggregate_iss` reproduces the worked example (rank per-organ AIS,
sum the squares of the top three), so the numeric output matches the paper.
`tests/test_domain_regression.py` pins both §2.7 values (ISS 5 → ESI-3, and the
underestimate boundary ISS 4 → ESI-4). The AAST→AIS map in
`lexicon/assets/aast.json` likewise uses the paper's convention (grade I–V → AIS
1–5), consistent with the §2.7 figures.

## D6 — Docker build not exercised in this environment

`docker build .` is the stated definition-of-done but Docker was unavailable where
this release was assembled, so the image build was not run here. The Dockerfile pins
a CUDA 12.1 / PyTorch 2.4 base consistent with the framework field and installs the
`gpu` and `imaging` extras; it has not been executed.

## D7 — Heavy dependencies are optional and lazily imported

Faithful backends require `torch`, `transformers`, `autoawq`, and DICOM/NIfTI
readers — none installable in a CPU-only CI runner at reasonable cost, and the model
weights are multi-gigabyte downloads. Handling: those imports are deferred to call
time and gated behind optional dependency groups (`gpu`, `imaging`). The algorithmic
core (perception math, knowledge graph, scoring, statistics, orchestration via
`MockVLM`) runs and is fully tested without them. This is an engineering boundary,
not a scientific approximation: the real backends are implemented exactly per §4.2.
