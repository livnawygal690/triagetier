# Project Context

Source: *An Agentic Vision-Language Model with Zero-Shot Transfer for Automated
Multi-Organ Trauma Triage in Resource-Limited Emergency Departments* (npj Digital
Medicine). The system described is **TraumaTriage-VLM**.

    project_name       : triagetier                                [HIGH]
    domain             : zero-shot multi-organ trauma triage from  [HIGH]
                         CT via agentic VLM orchestration
    framework          : PyTorch 2.x + HF Transformers (inference) [HIGH]
                         no training stack
    venue              : npj Digital Medicine                      [HIGH]
    primary_datasets   : 3 datasets (see §6)                       [HIGH]
    compute_target     : inference-only; A100 / Jetson AGX Orin    [HIGH]
    hparams_reference   : inference profile, Tables 1-4 + §4.2-4.5  [HIGH]
    supp_path          : none (S1-S9 referenced in text only)

    NEEDS_USER_DECISION: 0
    Ready to proceed.

The defining property of this work is that it performs **no model training**. Every
reported number is zero-shot inference over a frozen open-source VLM. There is no
loss function, optimizer, learning-rate schedule, epoch budget, or gradient step.
What the kickoff calls the "compute profile" and "hyperparameter set" are here an
*inference profile*: backbone choice, quantization precision, decoding policy, slice
counts, Hounsfield-Unit (HU) windows, ReAct iteration cap, confidence threshold, and
seed count. The consequences for repository layout and the kickoff training rules are
recorded in `docs/deviations.md`.

---

## 1. project_name — `triagetier` [HIGH]

The paper names the system "TraumaTriage-VLM" throughout (Abstract; Introduction last
paragraph; §2.1). This reference implementation is packaged as `triagetier`; the source is organized as
compiler-style lowering stages (a case is lowered to a triage verdict), see
`docs/repo-plan.md` and deviation D4.

## 2. supp_path — none [HIGH]

No supplementary file accompanies the manuscript. Supplementary Tables S1–S9 are
cross-referenced in §2.1 (Table 1 footnote), §4.6 ("Baseline methods"), and §4.2
(Table S2), but only as text references; no SI document is present. A search of
sibling locations returned no `*supp*`, `*_si.*`, `appendix*`, or `SI.*` file for
this manuscript.

## 3. domain — zero-shot multi-organ trauma triage from CT via agentic VLM orchestration [HIGH]

Computational emergency radiology / clinical decision support. The framework reads
trauma CT (abdominal: RATIC; head: CQ500) and emergency-department clinical text
(MIMIC-IV-ED) and produces organ-injury detection, AAST/AIS severity grading, Injury
Severity Score aggregation, and an Emergency Severity Index triage recommendation
(Abstract; §1 last paragraph; §4.1).

## 4. framework — PyTorch 2.x + HF Transformers (inference), no training stack [HIGH]

Evidence: the pipeline orchestrates open-source VLMs Qwen2.5-VL-7B/3B,
InternVL2.5-8B, and LLaVA-Med (§4.2), all distributed as HuggingFace Transformers
checkpoints. Edge deployment uses Activation-Aware Weight Quantization (AWQ) to
INT4/INT8 (§4.2), i.e. the `autoawq` toolchain. CT volumes require DICOM/NIfTI I/O
and HU windowing (§4.3) → `pydicom` / `nibabel` / `SimpleITK` + `numpy` + `Pillow`.
Statistics (§4.7) use SciPy 1.11 + scikit-learn 1.3 (stated verbatim). Decoding is
greedy (temperature=0). There is no `torch.optim`, no DataLoader-driven training, no
distributed training; "DDP/FSDP/AMP" from the kickoff do not apply.

## 5. venue — npj Digital Medicine [HIGH]

Structured abstract, Nature-style numbered references with DOIs, and the section set
(Data Availability, Code Availability, Acknowledgements, Author Contributions,
Competing Interests) match the npj Digital Medicine article template. Several
references are themselves npj Digital Medicine papers ([22], [23], [29], [55]).

## 6. primary_datasets — 3 datasets [HIGH]

| name | version | role | N | access |
|---|---|---|---|---|
| RATIC (RSNA 2023 Abdominal Trauma Detection) | competition release | primary visual triage (4 agents) | 4,274 studies; test N≈874 | https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection |
| CQ500 | public release (2 batches) | cross-anatomical head transfer | 491 scans / 193,317 slices | http://headctstudy.qure.ai/ |
| MIMIC-IV-ED | v2.2 | text-only triage reasoning (2 agents) | trauma subset N=2,000 of >425,000 stays | https://physionet.org/content/mimic-iv-ed/2.2/ |

Source: §4.6 and Data Availability. Ground truth: RATIC ASER/SAR radiologist
consensus (liver/spleen/kidney healthy/low/high-grade + binary bowel-mesenteric +
extravasation); CQ500 three-radiologist consensus (ICH five subtypes, calvarial
fracture, midline shift, mass effect); MIMIC-IV-ED triage notes + vitals + ESI.

Licensing (to be confirmed verbatim against each canonical URL during the README
step; recorded, not invented — R6):
- RATIC: governed by the Kaggle competition rules / RSNA data-use terms.
- CQ500: qure.ai public research release terms (CC BY-NC-SA family) — confirm.
- MIMIC-IV-ED: PhysioNet Credentialed Health Data License 1.5.0 + DUA; **restricted
  access** (credentialing required). Flagged restricted.

## 7. compute_target — inference-only [HIGH]

The paper reports no training compute. Deployment/inference profile (Table 4):

| model | precision | hardware | VRAM | latency |
|---|---|---|---|---|
| Qwen2.5-VL-7B | FP16 | A100 | 14.2 GB | 8.3 s/case |
| Qwen2.5-VL-7B | INT8 | A100 | 8.1 GB | 6.7 s/case |
| Qwen2.5-VL-7B | INT4 (AWQ) | Jetson AGX Orin 8 GB | 4.8 GB | 38.2 s/case |
| Qwen2.5-VL-3B | FP16 | A100 | 6.4 GB | 4.1 s/case |
| Qwen2.5-VL-3B | INT4 | Jetson Orin Nano 8 GB | 2.6 GB | 52.4 s/case |
| InternVL2.5-8B | INT4 | Jetson AGX Orin 8 GB | 5.2 GB | 42.7 s/case |

The accuracy-optimal edge configuration is Qwen2.5-VL-7B INT4 on Jetson AGX Orin:
38.2 s is 1.1% of the 60-minute golden hour (§2.4). No GPU-hours/storage for training
exist because the model weights are never updated.

## 8. hparams_reference — inference profile (Tables 1-4 + §4.2-4.5) [HIGH]

- Backbone: Qwen2.5-VL-7B primary; InternVL2.5-8B cross-architecture; LLaVA-Med
  medical baseline; Qwen2.5-VL-3B edge variant (§4.2).
- Quantization: AWQ, top-1% salient channels protected by per-channel scaling;
  INT4 (≈4.8 GB) and INT8 evaluated (§4.2).
- Decoding: greedy, temperature=0 (§4.2) — output deterministic for fixed input.
- Slice selection: K=7 solid-organ slices at percentiles {10,25,40,50,60,75,90};
  K=10 for bowel (full abdominopelvic coverage); organ HU bands liver[40,80],
  spleen[45,65], kidney[30,60]; min connected-component area 500 px (§4.3).
- Multi-window render: soft-tissue W400/L50, liver W150/L30, bone W1800/L400;
  head: brain W?/L? at [0,80] HU and subdural/epidural [50,200] HU (§4.3).
- QA ReAct loop: ≤3 iterations; confidence threshold 0.3 (normalized) for mandatory
  expert review; 12 physiological-plausibility rules (§4.5).
- Seeds: 5; reported as mean ± s.d.; variability arises from slice-choice randomness
  and prompt-template ordering only (§4.6 "Seed variance protocol").
- Statistics: bootstrap 1,000 resamples; DeLong (AUC); McNemar (accuracy);
  Holm-Bonferroni over 9 primary contrasts; Benjamini-Hochberg FDR α=0.05
  (exploratory); null-result threshold T=0.3% (§4.7).

## 9. extra_signals

- No released checkpoints: the system uses external, publicly distributed VLMs and
  performs zero-shot inference; nothing to release as a trained artifact.
- Article template sections Code Availability / Acknowledgements / Author
  Contributions appear with empty bodies (manuscript p. with References onset);
  Competing Interests states authors declare none.
- Numbered equations: one (ISS Baker formula, §4.1 / §4.4). No formal algorithm
  boxes; the 3-stage slice selection (§4.3) and the ReAct QA loop (§4.5) are
  procedural specifications implemented as such.
- SI-only material: Tables S1–S9 (reproduction protocols, per-seed results, baseline
  adaptation details) — referenced, not provided.
- Clinical-safety framing: every finding is flagged for physician review before
  clinical action; low-confidence (<0.3) outputs escalate to mandatory expert
  review; partial-ISS scope boundary reported in all evaluation tables (§4.1, §4.5).
