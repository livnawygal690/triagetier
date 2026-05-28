**FOR IMMEDIATE RELEASE**

# Off-the-shelf vision-language model triages multi-organ trauma with zero task-specific training

*2026 — npj Digital Medicine · reference implementation `triagetier` · contact: the corresponding author*

**A frozen open-source vision-language model, wrapped in four cooperating agents and a
self-checking quality module, reads trauma CT and emergency-department notes and returns a
full triage assessment — organ-injury detection, AAST/AIS grading, Injury Severity Score, and
an Emergency Severity Index recommendation — without a single step of fine-tuning.** The
system, named *TraumaTriage-VLM* in the manuscript and shipped here as the `triagetier`
package, reaches a mean per-organ AUC of 0.792 on RSNA abdominal trauma, 0.856 on CQ500 head
CT, and 74.2% exact ESI agreement on MIMIC-IV-ED text triage — and runs at INT4 on an 8 GB
Jetson in well under the golden hour. There are no released weights because there is nothing
to train: every number is zero-shot inference over publicly distributed checkpoints.

---

## How it works

Each case is *lowered* through a fixed sequence of stages, the way a compiler lowers source
to a target — hence the package's organization. Imaging is read and rendered, a frozen VLM is
queried, and four clinical agents grounded in the Advanced Trauma Life Support sequence turn
its answers into a decision, with a quality-assurance pass that cross-checks the chain and
escalates low-confidence cases:

```
case ─▶ frontends ─▶ lowering ─▶ oracles ─▶ passes ──────────────▶ driver ─▶ verdict
        (datasets)   (localize,   (VLM       (detect▸grade▸          (orchestrate,
                       window,      backend)   aggregate▸triage▸QA)    ReAct retry)
                       sample)                 grounded in lexicon
```

Two modes are selected by the data: a four-agent **visual** mode when imaging is present, and
a two-agent **text** mode (aggregate + triage) when it is not. The model weights are never
updated.

The source is laid out as compiler-style stages rather than horizontal tiers:

| stage | package | what passes through it |
|---|---|---|
| `ir` | `triagetier.ir` | typed case records shared across every stage |
| `host` | `triagetier.host` | config loading, seeding, logging, run manifests |
| `frontends` | `triagetier.frontends` | dataset adapters + registry (RATIC, CQ500, MIMIC-IV-ED, synthetic) |
| `lowering` | `triagetier.lowering` | CT localization, slice selection, multi-window rendering |
| `oracles` | `triagetier.oracles` | VLM backends (Qwen2.5-VL, InternVL2.5, LLaVA-Med), AWQ, mock, image I/O |
| `lexicon` | `triagetier.lexicon` | clinical knowledge graph + AIS/ISS/ESI/confidence arithmetic |
| `passes` | `triagetier.passes` | the four ATLS agents (detect, grade, aggregate, triage) + QA cross-checker |
| `driver` | `triagetier.driver` | orchestrator, ReAct reconciliation loop, text-mode wiring |
| `reports` | `triagetier.reports` | AUC/CI, triage accuracy, significance tests |
| `toolchain` | `triagetier.toolchain` | command verbs run / evaluate / inspect |

The clinical knowledge graph (AAST grading trees, ISS body-region map, ESI v4 logic) ships as
JSON inside `triagetier/lexicon/assets/`.

## Results at a glance

Primary results, mean ± s.d. over 5 seeds (seed variability comes only from slice-choice
randomness and prompt ordering — decoding is greedy):

| evaluation | metric | reported |
|---|---|---|
| RATIC | mean per-organ AUC | 0.792 ± 0.031 (range 0.761–0.823) |
| RATIC | kidney / liver / spleen AUC | 0.819 / 0.787 / 0.771 |
| RATIC | extravasation / bowel-mesenteric AUC | 0.734 / 0.668 |
| RATIC | high-grade solid-organ AUC | 0.843 ± 0.022 |
| CQ500 | intracranial-haemorrhage AUC | 0.856 ± 0.028 |
| MIMIC-IV-ED | ESI exact-level match | 74.2% ± 2.4 |
| MIMIC-IV-ED | within-one / off-by-two | 89.1% / 3.2% |
| MIMIC-IV-ED | Cohen's κ | 0.618 |
| reasoning subset | detection prec. / grading acc. / chain completeness / ECE | 82.3% / 71.4% / 89.1% / 0.067 |

Every component earns its place — ablating any one of them moves RATIC mean organ AUC
(Table 3):

| configuration | AUC | Δ |
|---|---|---|
| full pipeline | 0.792 | — |
| − knowledge graph | 0.738 | −5.4 |
| − intelligent slice selection | 0.723 | −6.9 |
| − multi-window rendering | 0.770 | −2.2 |
| − QA module | 0.782 | −1.0 |
| single-pass VLM (no agents) | 0.642 | −15.0 |

Significance follows the paper: DeLong for AUC contrasts, McNemar for accuracy, Holm–Bonferroni
over the nine primary comparisons, Benjamini–Hochberg (α=0.05) for exploratory per-organ /
per-ESI analyses, and an ablation null-result threshold of T=0.3%.

## Try it now

**Install.** The algorithmic core (slice selection, knowledge graph, scoring, orchestration,
statistics) needs only numpy, scipy, scikit-learn, OmegaConf and Pillow and runs on CPU; the
VLM backends and DICOM/NIfTI readers are optional extras.

```bash
pip install -e .                 # core (CPU)
pip install -e ".[gpu,imaging]"  # VLM backends + medical-image I/O
pip install -e ".[dev]"          # linters, type checker, tests
```

```bash
conda env create -f environment.yml && conda activate triagetier && pip install -e ".[gpu]"
```

```bash
docker build -t triagetier .            # mirrors the paper's compute target
docker run --gpus all triagetier experiment=main
```

**Reproduce a specific number.** Each headline figure maps to one command:

| to reproduce | run | expect |
|---|---|---|
| smoke test (CPU, mock backend) | `python -m triagetier.toolchain.run experiment=_smoke` | `case=synthetic-000 esi=3 iss=5 partial=True` |
| RATIC mean organ AUC = 0.792 | `python -m triagetier.toolchain.evaluate experiment=main` | mean per-organ AUC 0.792 ± 0.031 |
| CQ500 ICH AUC = 0.856 | `python -m triagetier.toolchain.evaluate experiment=cq500` | ICH AUC 0.856 ± 0.028 |
| MIMIC ESI match = 74.2% | `python -m triagetier.toolchain.evaluate experiment=mimic` | exact-level 74.2% ± 2.4, κ 0.618 |
| ablation − knowledge graph | `python -m triagetier.toolchain.evaluate experiment=ablation_no_knowledge_graph` | AUC 0.738 (−5.4) |
| edge profile Qwen-7B INT4 | `python -m triagetier.toolchain.evaluate experiment=supplementary_edge_qwen7b_int4` | see compute table |
| per-case trace for debugging | `python -m triagetier.toolchain.inspect experiment=main` | slice picks + knowledge lookups + per-agent records |

The installed console script `triagetier experiment=NAME` is an alias for the `run` verb. Any
field is overridable with OmegaConf dotlist syntax, e.g.
`python -m triagetier.toolchain.run experiment=_smoke pipeline.use_qa=false`. Experiment files
under `settings/experiment/` compose a pipeline default with a `settings/model/` and a
`settings/data/` file; overrides win over all of them.

## What it costs to run

Inference only — weights are never updated. Latency is the full four-agent-plus-QA pipeline
per case (Table 4):

| model | precision | hardware | VRAM | latency |
|---|---|---|---|---|
| Qwen2.5-VL-7B | FP16 | A100 | 14.2 GB | 8.3 s/case |
| Qwen2.5-VL-7B | INT8 (AWQ) | A100 | 8.1 GB | 6.7 s/case |
| Qwen2.5-VL-7B | INT4 (AWQ) | Jetson AGX Orin 8 GB | 4.8 GB | 38.2 s/case |
| Qwen2.5-VL-3B | FP16 | A100 | 6.4 GB | 4.1 s/case |
| Qwen2.5-VL-3B | INT4 (AWQ) | Jetson Orin Nano 8 GB | 2.6 GB | 52.4 s/case |
| InternVL2.5-8B | INT4 (AWQ) | Jetson AGX Orin 8 GB | 5.2 GB | 42.7 s/case |

The accuracy-optimal edge configuration is Qwen2.5-VL-7B at INT4 on Jetson AGX Orin: 38.2 s per
case is 1.1% of the 60-minute golden hour. Disk is dominated by raw imaging (see below); model
weights add 5–15 GB by precision.

## Where the data comes from

All three datasets are public; MIMIC-IV-ED requires credentialed access. Use
`scripts/prepare_data.sh <data_root>` for the expected on-disk layout, convert RATIC/CQ500
DICOM series to NIfTI, and record a manifest (`shasum -a 256 <data_root>/ratic/test_manifest.csv`).

| dataset | version | size | license | source |
|---|---|---|---|---|
| RATIC (RSNA 2023 Abdominal Trauma) | competition release | ~470 GB | RSNA / Kaggle competition data-use terms | https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection |
| CQ500 | public release, 2 batches | ~40 GB | qure.ai research-use terms (CC BY-NC-SA family) | http://headctstudy.qure.ai/ |
| MIMIC-IV-ED | v2.2 | ~2 GB | PhysioNet Credentialed Health Data License 1.5.0 (DUA, credentialing required) | https://physionet.org/content/mimic-iv-ed/2.2/ |

RATIC evaluation uses the official competition test split (N≈874). The MIMIC-IV-ED trauma
subset (N=2,000) is selected by ICD-10 S00–T88, excluding poisoning, burns, and surgical
complications, with ESI distribution 4.2 / 23.1 / 42.8 / 22.5 / 7.4 percent.

## The fine print

This is a purely computational, retrospective study on public datasets. Every finding is
intended for clinician review before any clinical action, and outputs with a normalized
confidence below 0.3 are flagged for mandatory expert review. Injury Severity Scores are
*partial* — computed over imaged body regions (abdomen, head) — and this scope boundary is
reported on every result. The system has not undergone prospective clinical validation.

## Code & weights

No checkpoints are released. The pipeline performs zero-shot inference over publicly
distributed model weights (Qwen2.5-VL, InternVL2.5, LLaVA-Med) downloaded from their original
sources; there is no trained artifact specific to this work. The manuscript's Code Availability
section carries no verbatim release statement, so none is quoted here.

## Cite this work

```bibtex
@article{triagetier,
  title   = {An Agentic Vision-Language Model with Zero-Shot Transfer for Automated
             Multi-Organ Trauma Triage in Resource-Limited Emergency Departments},
  journal = {npj Digital Medicine},
  year    = {2026}
}
```

---

*Boilerplate — `triagetier` is the reference implementation behind the abdominal (RATIC), head
(CQ500), and text-triage (MIMIC-IV-ED) results reported in the manuscript. It is research
software, distributed for reproduction and not as a medical device. Run `make help` for the
available development targets (`install`, `lint`, `type`, `test`, `smoke`, `docker`).*
