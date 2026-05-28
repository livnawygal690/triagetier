# Implementation map

Every numbered result, equation, table, ablation, and protocol element traced to the
module that realizes it. Because in-code comments are suppressed for this release,
this table is the authoritative provenance record (paper → code). Paths are relative
to `triagetier/`.

## Architecture and agents

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.1, Fig. 3 | four-agent ATLS pipeline | `driver/orchestrator.py` | `Orchestrator` | sequential detect→grade→aggregate→triage; visual agents skipped in text mode |
| §4.1 (1) | Organ Detection Agent A_det | `passes/detection.py` | `DetectionAgent` | per-organ structured finding from K slices |
| §4.1 (2) | Severity Grading Agent A_grade | `passes/grading.py` | `GradingAgent` | finding → AIS via knowledge graph |
| §4.1 (3) | ISS Aggregation Agent A_iss | `passes/aggregation.py` | `AggregationAgent` | partial ISS over imaged regions |
| §4.1 (4) | Triage Decision Agent A_triage | `passes/triage.py` | `TriageAgent` | ESI 1–5 + reasoning chain |
| §4.5 | QA module (ReAct) | `passes/qa.py`, `driver/react.py` | `QualityAssurance`, `react_loop` | consistency + plausibility + confidence |
| §4.1 | agent I/O contracts | `ir/records.py` | dataclasses/TypedDict | OrganFinding, AISGrade, PartialISS, TriageDecision, CaseResult |

## Equations and scoring

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.1, §4.4 (Baker) | ISS = g²₍₁₎+g²₍₂₎+g²₍₃₎ | `lexicon/iss.py` | `injury_severity_score`, `partial_iss` | top-3 region AIS squared; partial over imaged regions |
| §4.4 (AAST) | AAST grade I–V → AIS | `lexicon/ais.py` | `aast_to_ais`, `match_grade` | select highest grade meeting all imaging criteria |
| §4.4 (ESI v4) | ESI decision logic | `lexicon/esi.py` | `esi_level` | acuity 1 (life-saving), 2 (high-risk), 3/4/5 (resources) |
| §4.5 | confidence scoring | `lexicon/confidence.py` | `confidence_score` | log-prob of key terms × multi-window agreement; threshold 0.3 |

## Perception (slice selection, §4.3)

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.3 stage 1 | HU-band organ localization | `lowering/localize.py` | `localize_organ` | connected components, min area 500 px, bands liver[40,80] spleen[45,65] kidney[30,60] |
| §4.3 stage 2 | percentile slice sampling | `lowering/sampling.py` | `select_slices` | K=7 at {10,25,40,50,60,75,90}; K=10 bowel |
| §4.3 stage 3 | multi-window tiled render | `lowering/windowing.py` | `window`, `tile_windows` | soft W400/L50, liver W150/L30, bone W1800/L400; head brain[0,80], subdural[50,200] |

## Knowledge graph (§4.4)

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.4 | AAST decision trees | `lexicon/assets/aast.json` | — | liver/spleen/kidney grades I–V imaging criteria |
| §4.4 | ISS 6-region map | `lexicon/assets/iss_regions.json` | — | head/neck, face, thorax, abdomen, extremities, external |
| §4.4 | ESI v4 logic | `lexicon/assets/esi.json` | — | acuity rules + resource thresholds |
| §4.4 | graph access | `lexicon/graph.py`, `lexicon/loader.py` | `KnowledgeGraph`, `load_graph` | typed access to the three assets |

## Backends and quantization (§4.2)

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.2 | backend interface | `oracles/base.py` | `VLMBackend` | generate(text,images)→text + token log-probs |
| §4.2 | Qwen2.5-VL-7B/3B | `oracles/qwen25vl.py` | `Qwen25VLBackend` | lazy HF import, dynamic resolution |
| §4.2 | InternVL2.5-8B | `oracles/internvl25.py` | `InternVL25Backend` | cross-architecture check |
| §4.2 | LLaVA-Med | `oracles/llava_med.py` | `LlavaMedBackend` | medical-domain baseline |
| §4.2 | AWQ INT4/INT8 | `oracles/awq.py` | `load_awq` | top-1% salient channels; ~55% compression |
| (test infra) | deterministic mock | `oracles/mock.py` | `MockVLM` | seed-driven; powers CPU tests |

## Datasets and protocol (§4.6)

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.6 RATIC | abdominal CT adapter | `frontends/ratic.py` | `RaticDataset` | official test split N≈874; 3 organs + bowel + extravasation |
| §4.6 CQ500 | head CT adapter | `frontends/cq500.py` | `CQ500Dataset` | cranial windows; ICH subtypes |
| §4.6 MIMIC | ED text adapter | `frontends/mimic_iv_ed.py` | `MimicEdDataset` | Gaber inclusion: ICD-10 S00–T88 excl. T20–T65/T80–T88; ESI dist check |
| §4.6 | registry | `frontends/registry.py` | `build_dataset` | name → adapter |

## Metrics and statistics (§4.6, §4.7)

| paper anchor | item | file | module | notes |
|---|---|---|---|---|
| §4.6 | per-organ AUC + bootstrap CI | `reports/stats.py` | `auc`, `bootstrap_ci` | 1,000 resamples |
| §4.6 | ESI exact / within-one acc | `reports/triage_eval.py` | `esi_match`, `within_one` | |
| §4.6 | ISS Spearman | `reports/stats.py` | `spearman` | r=0.763 reported |
| §4.6 | Cohen's κ | `reports/stats.py` | `cohen_kappa` | κ=0.618 reported |
| §2.5/§4.5 | ECE | `reports/stats.py` | `expected_calibration_error` | 0.067 reported |
| §4.7 | DeLong test | `reports/stats.py` | `delong_test` | paired ROC AUC |
| §4.7 | McNemar test | `reports/stats.py` | `mcnemar_test` | continuity correction |
| §4.7 | Holm-Bonferroni | `reports/stats.py` | `holm_bonferroni` | 9 primary contrasts |
| §4.7 | Benjamini-Hochberg FDR | `reports/stats.py` | `benjamini_hochberg` | α=0.05 |
| §4.6 | detection precision / grading acc / chain completeness | `reports/detection_eval.py` | `agent_reasoning_metrics` | 82.3 / 71.4 / 89.1 |

## Reported numbers wired as config/README expected values

| paper anchor | result | where encoded |
|---|---|---|
| §2.1, Table 1 | RATIC 0.792±0.031; CQ500 0.856±0.028; MIMIC ESI 74.2±2.4, κ=0.618, within-one 89.1 | `settings/experiment/main.yaml`, README expected table |
| Table 2 | per-organ AUC (kidney .819 … bowel .668; high-grade .843); CQ500 subtypes | README expected table |
| Table 3 | ablation drops (KG −5.4, slice −6.9, QA −1.0, window −2.2, single-pass −15.0) | `settings/experiment/ablation_*.yaml` |
| Table 4 | edge VRAM/latency/accuracy | `settings/experiment/supplementary_edge_*.yaml` |
| §2.5 | slice strategy: landmark +10.2 pp; KG-prompt +5.4 pp | `settings/experiment/*slice*.yaml` |
| §2.7 | illustrative case (hepatic II + splenic I → partial ISS 5 → ESI-3) | `tests/test_domain_regression.py` golden |
