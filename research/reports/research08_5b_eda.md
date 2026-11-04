# Research 08.5b — Corpus and QA Dataset Exploratory Analysis

**Project:** Scientific Research Assistant
**Checkpoint:** November 10, 2026
**Report status:** Preliminary, v0.3
**Analysis snapshot:** November 4, 2026

## 1. Research Objective

The Scientific Research Assistant project aims to support
evidence-grounded question answering over scientific publications
and their citation networks.

Research 08.5b focuses on preparing and examining the data
required for future retrieval and RAG experiments.

The current research objectives are:

1. Characterize the extracted scientific text corpus.
2. Identify data-quality limitations affecting downstream QA.
3. Prepare source-grounded QA candidates.
4. Evaluate local LLM-based QA generation.
5. Analyze technical failures and scientific correctness.
6. Establish a reproducible basis for subsequent dataset expansion.

This stage does not evaluate retrieval or RAG performance.

## 2. Data Sources

The current EDA covers seven arXiv publications.

The corpus consists of two previously prepared subsets:

| Subset | Articles | Extracted paragraphs |
|---|---:|---:|
| Initial QA pilot | 3 | 230 |
| Research 08.5b | 4 | 252 |
| Total | 7 | 482 |

### 2.1. Article-level statistics

| arXiv ID | Paragraphs | Median length | Paragraphs <250 chars |
|---|---:|---:|---:|
| 1406.1078v3 | 99 | 299 | 47 |
| 1409.0473v7 | 104 | 298.5 | 44 |
| 1508.07909v5 | 57 | 455 | 9 |
| 1512.00567v3 | 42 | 583 | 9 |
| 1607.06450v1 | 84 | 342.5 | 31 |
| 1610.02357v3 | 34 | 569 | 4 |
| 1706.03762v7 | 62 | 380.5 | 19 |
| Total | 482 | 380.5 | 163 |

Paragraph lengths are measured in characters of the extracted
raw LaTeX, not normalized plain text or model tokens.

### 2.2. Source provenance

Each exported paragraph contains:

- arXiv paper identifier and version;
- source archive SHA-256;
- main TeX filename;
- section index and section title;
- assembled document offsets;
- original source-file spans;
- raw LaTeX content.

Paragraph identifiers are derived from assembled document offsets.

Consequently, they should not be treated as permanent identifiers
independent of the extraction pipeline.

## 3. Corpus Integrity

The combined corpus contains:

| Integrity check | Result |
|---|---:|
| Total records | 482 |
| Unique source IDs | 482 |
| Unique articles | 7 |
| Duplicate source IDs | 0 |
| Records with missing required fields | 0 |
| Invalid coordinate ranges | 0 |
| Paragraph length mismatches | 0 |
| Articles with inconsistent source hashes | 0 |

For every exported paragraph, the following invariant holds:

`len(raw_latex) == assembled_end - assembled_start`

These checks confirm structural consistency of the exported
paragraph records.

They do not establish complete preservation of all scientific
content from the original papers.

Equations, tables, figures, lists, and other structures may be
represented by separate extraction blocks.

## 4. Paragraph Length Distribution

The combined corpus contains 482 paragraphs.

Summary statistics:

- Minimum length: 3 characters.
- Median length: 380.5 characters.
- Maximum length: 2024 characters.
- Sections containing at least one paragraph: 136.
- Paragraphs containing multiple source spans: 1.

### 4.1. Length distribution

| Character length | Paragraphs | Percentage |
|---|---:|---:|
| <100 | 79 | 16.4% |
| 100–249 | 84 | 17.4% |
| 250–499 | 149 | 30.9% |
| 500–999 | 149 | 30.9% |
| 1000–1999 | 20 | 4.1% |
| >=2000 | 1 | 0.2% |

A total of 163 paragraphs (33.8%) contain fewer than
250 characters.

The majority of paragraphs (61.8%) are between 250 and
999 characters long.

These observations describe structural extraction output.
They do not directly determine optimal retrieval chunk sizes.

## 5. Preliminary Data-Quality Analysis

### 5.1. Short paragraphs

The shortest extracted paragraphs include several distinct
categories.

**Mathematical continuations**

Examples include isolated words such as:

- `where`
- `and`

These fragments often occur near mathematical expressions
and should not be interpreted as independent QA passages.

**Introductory text preceding structured blocks**

Some short paragraphs introduce lists, equations, or experiments.

Such text may require neighboring blocks to provide sufficient
semantic context.

**LaTeX control and bibliography directives**

Examples include:

- `\newpage`
- `\bibliographystyle`
- `\bibliography`

These constructs do not constitute independent scientific
evidence for QA.

**Short but meaningful scientific statements**

Some paragraphs shorter than 100 characters contain valid
scientific information.

Therefore, paragraph length alone is insufficient for
determining QA suitability.

### 5.2. Source-span boundaries

Only one paragraph contains multiple source spans.

It occurs in article `1706.03762v7` and consists of LaTeX
layout commands spanning `ms.tex` and `visualizations.tex`.

The combined source-span lengths match the extracted paragraph
length.

This example illustrates the distinction between correct
source provenance and downstream semantic usefulness.

### 5.3. Implications for QA selection

The initial QA candidate selection should account for:

- paragraph length;
- semantic self-containment;
- dependence on equations, figures, or tables;
- LaTeX comments and control sequences;
- scientific relevance;
- availability of sufficient supporting evidence.

A simple character-length threshold should be treated as
a preliminary filtering heuristic, not a quality label.

## 6. QA Generation Experiments

### 6.1. Experimental setup

Local QA generation was evaluated using Ollama.

Models:

- `qwen3:8b-q4_K_M`
- `qwen3.5:9b`

Both models were evaluated on the same initial set of eight
source paragraphs.

Qwen3.5 9B was subsequently evaluated on 12 additional,
non-overlapping paragraphs.

The initial evaluation set contained two source paragraphs
from each of the four Research 08.5b publications.
The extended set contained three additional paragraphs
per publication, including passages from experimental
and results sections.

These small, balanced sets were used for exploratory
analysis of generation behavior and error types.
They should not be considered random or representative
samples of scientific text.

The experiments used:

- QA generation prompt v2;
- structured JSON output;
- temperature 0;
- context length 4096;
- thinking disabled;
- strict evidence alignment.

Prompt SHA-256:

`79017558d3b9396acddacfd9cf7d8d102d302fe0351c9d8ad1df85e38655cb98`

### 6.2. Technical validation

A generated QA record is technically successful when its
structured response is valid and the evidence quote can be
uniquely aligned to the source paragraph.

Evidence alignment permits whitespace normalization but
does not permit arbitrary changes to source text, mathematical
notation, or LaTeX commands.

| Experiment | Generated | Technical pass | Pass rate |
|---|---:|---:|---:|
| Qwen3 8B, initial set | 8 | 4 | 50.0% |
| Qwen3.5 9B, initial set | 8 | 7 | 87.5% |
| Qwen3.5 9B, extended set | 12 | 11 | 91.7% |

Across its two non-overlapping sets, Qwen3.5 9B achieved
18 technically valid generations out of 20 (90.0%).

### 6.3. Preliminary scientific review

Technical validity does not guarantee scientific correctness
or sufficient evidence coverage.

Generated QA records were assigned one of three preliminary
review statuses:

- `accepted`: usable without revision;
- `needs_revision`: potentially usable after correction;
- `rejected`: unsuitable in its generated form.

| Experiment | Accepted | Needs revision | Rejected |
|---|---:|---:|---:|
| Qwen3 8B, initial set | 2 | 5 | 1 |
| Qwen3.5 9B, initial set | 3 | 5 | 0 |
| Qwen3.5 9B, extended set | 7 | 4 | 1 |

Across 20 Qwen3.5 generations:

- 10 were accepted without revision (50%);
- 9 required revision (45%);
- 1 was rejected (5%).

All review labels currently have the status
`assistant_preliminary`.

They must not be represented as independently validated
human gold annotations.

### 6.4. Combined QA Candidate Dataset

A combined QA candidate dataset was assembled from three
previously curated groups and a separately reviewed manual top-up:

| QA group | Recorded model | Candidates |
|---|---|---:|
| Research 08.5b | qwen3.5:9b | 19 |
| Initial manual QA pilot | GPT-6 | 11 |
| Additional Qwen pilot | qwen3:8b-q4_K_M | 2 |
| Manual top-up (November 4) | GPT-6 | 3 |
| Total | — | 35 |

The GPT-6 records were prepared through a manually assisted
ChatGPT workflow. They should not be interpreted as outputs
of an automated generation experiment directly comparable
to the local Ollama runs.

The combined dataset contains 35 QA candidates associated
with 30 distinct source paragraphs across seven papers.

Multiple QA candidates may refer to the same source paragraph.
Consequently, source_id is not required to be unique across
QA records. The unique identifier is qa_id.

#### 6.4.1. Distribution across publications

| arXiv ID | QA candidates |
|---|---:|
| 1406.1078v3 | 5 |
| 1409.0473v7 | 5 |
| 1508.07909v5 | 5 |
| 1512.00567v3 | 5 |
| 1607.06450v1 | 5 |
| 1610.02357v3 | 5 |
| 1706.03762v7 | 5 |
| Total | 35 |

Each publication contributes five QA candidates.

The 30 distinct source paragraphs represent approximately
6.2% of the 482 extracted paragraphs.

The selected QA evidence spans 25 of the 136 document
sections containing extracted paragraphs (18.4%).

These ratios describe selection coverage of extracted structures.
Neither paragraph nor section coverage should be interpreted
as a measure of semantic coverage or QA dataset quality.

#### 6.4.2. QA length statistics

| Field | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Question | 65 | 122 | 202 |
| Answer | 30 | 126 | 272 |
| Evidence quote | 81 | 221 | 479 |

Lengths are measured in characters of the stored strings.

Evidence quotes retain their original LaTeX representation.
Their character lengths do not directly correspond to
plain-text lengths or model token counts.

#### 6.4.3. Review and curation history

The initial manually assisted pilot contributed 11 accepted
first-pass QA records. During the second review pass,
seven were revised and four were retained unchanged.

The additional Qwen3:8B pilot contributed two preliminarily
accepted records. One was revised during the second pass,
while the other remained unchanged.

The 19 Research 08.5b candidates have the following
curation origins:

| Curation origin | Candidates |
|---|---:|
| Second-pass processing | 10 |
| Revised from technically successful generation | 7 |
| Reconstructed from technically failed generation | 2 |
| Total | 19 |

The revision_applied flag is true for 14 of the 19
Research 08.5b candidates.

However, this count includes different curation procedures,
including reconstruction after technical generation failure.

It must not be interpreted as the number of scientifically
incorrect model answers or as a direct model error rate.

All 19 Research 08.5b candidates received an accepted
decision during the final checkpoint review.

These decisions remain assistant_preliminary annotations.
No independent expert adjudication has yet been performed.

The final set of 19 Research 08.5b candidates represents
the result of assistant-assisted curation, not the
generation accuracy of Qwen3.5 9B.

Of the original 20 Qwen3.5 generations, 10 were accepted
unchanged, seven technically successful candidates were
revised, two were reconstructed after technical failures,
and one was rejected. Thus, the 19 curated records do not
imply a 95% first-pass model success rate.

The November 4 manual top-up contributed three further QA
records, one each for neural machine translation (bidirectional
encoder annotations), Inception (label-smoothing loss), and
Xception (intermediate activation ablation). All three use
previously unused source paragraphs. Their candidate and review
records are preserved separately, and the review status remains
`assistant_preliminary`. This was an assistant-assisted manual
curation step, not an additional Ollama generation experiment.

### 6.5. Dataset Integrity and Provenance

The combined dataset was constructed using a deterministic
assembly pipeline.

The pipeline validates individual QA groups before merging
them into a single collection.

The final integrity results are summarized below.

| Integrity check | Result |
|---|---:|
| Total QA candidates | 35 |
| Unique QA IDs | 35 |
| Distinct source IDs | 30 |
| Publications | 7 |
| Conflicting paper source hashes | 0 |
| Conflicting metadata for reused source IDs | 0 |
| Missing required QA fields | 0 |
| Exact duplicate questions | 0 |
| Unexpected dataset statuses | 0 |
| JSONL readback verification | Passed |

Exact duplicate questions were checked using paper IDs
and question strings normalized for case and whitespace.

This procedure does not detect semantic paraphrases or
near-duplicate questions.

Each combined QA record preserves the following information:

- QA identifier and source paragraph identifier;
- publication identifier and source archive SHA-256;
- source document and section metadata;
- question, answer, and exact evidence quote;
- evidence coordinates within the raw LaTeX paragraph;
- generation model and available generation provenance;
- applicable review and revision metadata;
- original dataset status and combined QA group.

For the manually assisted pilot, source-file evidence
coordinates were also checked against the corresponding
source paragraph spans.

For the 19 Research 08.5b candidates, the stored checkpoint
review hashes were verified against the reviewed QA content.

The combined dataset uses:

`dataset_status = combined_candidate`

All records retain:

`verification_status = pending`

These statuses distinguish successful dataset assembly
from independent scientific validation.

The versioned output is stored in:

`research/data/qa_pilot/qa_combined_candidates_v2.jsonl`

Its SHA-256 checksum is:

`a380a375bb040ba60a42a89bf6863535a93fada7c600bb9c2038fa138dc800a4`

This checksum was reproduced in the author's local development
environment as well as in the handoff-based working copy.
The earlier v1 dataset and its verified checksum are preserved
unchanged. JSONL line endings are explicitly configured as LF
through `.gitattributes`.

## 7. QA Error Analysis

The following error categories were observed:

Across the 20 Qwen3.5 9B generations, the preliminary
review records contain these issue-code frequencies:

| Issue code | Occurrences |
|---|---:|
| `evidence_insufficient` | 3 |
| `question_context_dependent` | 3 |
| `latex_corrupted` | 2 |
| `non_verbatim_evidence` | 2 |
| `unsupported_claim` | 2 |
| `uncertainty_omitted` | 1 |

One generation may have more than one issue code, so the
counts must not be summed as if they represented distinct
failed generations. The frequencies are descriptive only;
the small, selected sample does not support general error-rate
estimates for Qwen3.5 9B.

### 7.1. LaTeX corruption

Generated evidence quotes sometimes modify LaTeX macros
or contain incorrectly escaped backslashes.

JSON decoding may introduce additional changes when
LaTeX sequences resemble JSON escape sequences.

These failures can occur even when the generated answer
is scientifically correct.

### 7.2. Insufficient evidence

Some generated answers contain information supported by
the source paragraph but not fully covered by the selected
evidence quote.

Exact text matching alone cannot detect this problem.

### 7.3. Unsupported causal claims

Some answers introduce plausible causal explanations that
are not directly established by the cited text.

This is particularly problematic when reporting
experimental results.

### 7.4. Loss of scientific uncertainty

Scientific claims may be presented more categorically
than in the original paper.

Expressions indicating uncertainty or preliminary findings
must be preserved.

### 7.5. Context-dependent questions

Some questions refer to an unspecified model, algorithm,
experiment, or architecture.

Such questions may be unsuitable for cross-document retrieval
without additional context.

### 7.6. Incorrect experimental comparisons

One Xception QA generation incorrectly interpreted a
relative performance improvement on JFT as a comparison
against results obtained on ImageNet.

The evidence quote was an exact source substring, but
it did not support the generated comparison.

This case demonstrates that technical evidence validation
is necessary but insufficient for scientific QA quality.

## 8. Limitations

The present study has several limitations.

1. The corpus contains only seven scientific publications.
   The results are not representative of arbitrary scholarly
   documents or scientific disciplines.

2. Paragraphs are extracted as raw LaTeX structures.
   Their character lengths do not directly correspond to
   normalized plain-text lengths or model token counts.

3. The local QA generation experiments use small,
   deliberately selected samples. They do not support
   statistically reliable comparisons or model rankings.

4. Some extracted paragraphs depend on neighboring
   equations, tables, figures, or lists. Exact evidence
   alignment does not guarantee that a passage contains
   sufficient context to answer a question independently.

5. The combined QA dataset contains only 35 candidates
   derived from 30 distinct source paragraphs.
   Although five candidates are included for each publication,
   semantic coverage, question types, and difficulty levels
   have not yet been systematically evaluated.

6. Review decisions remain preliminary.
   The current annotations were produced through
   assistant-supported review procedures and have not
   undergone independent expert adjudication.

   Consequently, the combined dataset must not be treated
   as a validated gold-standard QA benchmark.

7. Exact duplicate detection uses normalized question
   strings within each publication.
   Semantic duplicates, paraphrases, and questions requiring
   substantially overlapping evidence have not yet been
   systematically reviewed.

8. The dataset has not been divided into independent
   development and evaluation subsets.

   Multiple QA candidates may share a source paragraph.
   Future dataset splitting must account for shared
   source evidence to avoid evaluation leakage.

9. Retrieval and RAG performance have not been evaluated.
   The present study does not report retrieval recall,
   ranking quality, answer faithfulness, or citation
   attribution performance.

10. Dataset assembly is deterministic given the preserved
    input artifacts. However, this does not establish
    end-to-end reproducibility of the original LLM
    generations.

    Original model outputs, generation configurations,
    manually curated records, and downloaded arXiv sources
    represent distinct provenance layers that require
    appropriate preservation and documentation.

These limitations define the scope of the current
research checkpoint and the requirements for subsequent
dataset development.

## 9. Next Steps

### 9.1. Before the November 10 checkpoint

The immediate priority is to finalize and document
the combined QA candidate dataset.

Planned activities include:

1. Document the completed assistant-preliminary review of
   35 QA candidates, including scientific correctness,
   evidence sufficiency, ambiguity, and preservation
   of the original authors' claims.

2. Identify potentially redundant questions and evaluate
   whether individual questions are understandable
   outside the context of their source paragraphs.

3. Document the dataset composition, curation procedures,
   provenance fields, and remaining quality limitations.

4. Preserve the curated QA artifacts and their required
   inputs under version control, while keeping downloaded
   source archives and unrelated intermediate data outside
   the repository.

5. Complete the Research 08.5b exploratory analysis report
   and verify its quantitative results against the
   versioned dataset.

6. Review the implementation and tests of the dataset
   assembly pipeline, removing unnecessary duplication
   where this can be done without changing its behavior.

7. Discuss the current results and the next research
   checkpoint with the project supervisor.

The dataset will retain candidate status until its
scientific annotations have been independently reviewed.

### 9.2. Subsequent research

The following stages will focus on:

- retrieval-oriented text normalization and chunking;
- construction of a scientifically reviewed QA
  evaluation subset;
- development of BM25, dense, and hybrid retrieval
  baselines;
- retrieval evaluation with explicit source evidence;
- citation-aware retrieval across linked publications;
- grounded answer generation and evidence attribution;
- evaluation of answer faithfulness and citation quality;
- agentic components where their contribution can
  be empirically justified.

Dataset splitting and evaluation design must account
for shared evidence passages and publication identities
to reduce the risk of information leakage.

## 10. Reproducibility

### 10.1. Implementation

Corpus and generation experiment analysis is implemented
in:

- `research/spikes/eda_qa_corpus.py`
- `research/spikes/eda_qa_generation.py`

QA curation and dataset assembly use:

- `research/spikes/apply_qa_second_pass.py`
- `research/spikes/assemble_qa_checkpoint.py`
- `research/spikes/assemble_qa_dataset.py`

The assembly pipeline preserves the original candidate
identifiers, source provenance, model metadata, and
available review and revision history.

The combined dataset is constructed from four
separately prepared QA groups:

- 19 Research 08.5b candidates;
- 11 manually assisted pilot candidates;
- 2 additional Qwen3:8B candidates;
- 3 additional manually curated candidates (November 4).

The three manual additions are recorded in
`qa_manual_topup_v1.jsonl` and `manual_review_qa_topup_v1.jsonl`.
The assembly validates their evidence coordinates and review
content hashes against the versioned source paragraphs.

The final assembly performs cross-group consistency
checks and applies a deterministic record ordering.

### 10.2. Combined dataset

The versioned output is:

`research/data/qa_pilot/qa_combined_candidates_v2.jsonl`

Dataset characteristics:

| Property | Value |
|---|---:|
| QA candidates | 35 |
| Unique QA IDs | 35 |
| Distinct source paragraphs | 30 |
| Publications | 7 |
| Exact duplicate questions | 0 |
| Dataset status | `combined_candidate` |
| Verification status | `pending` |

The output SHA-256 is:

`a380a375bb040ba60a42a89bf6863535a93fada7c600bb9c2038fa138dc800a4`

The original 32-candidate `qa_combined_candidates_v1.jsonl`
remains unchanged, with SHA-256
`e942be16fdaef71bdd2320b776de937503e7172df822b25e0c93d645067bca16`.

The checksum of v2 was independently reproduced from
the preserved inputs and compared with the working-tree
file in the author's development environment.

The relevant JSONL artifacts use LF line endings,
enforced through the repository's `.gitattributes`.

### 10.3. Reproducing the assembly

The dataset can be rebuilt from the preserved
source-paragraph exports, curated QA candidates,
generation records, and review files.

From the repository root, execute:

```bash
uv run --locked python -m research.spikes.assemble_qa_dataset \
  --include-manual-topup \
  --data-dir research/data/qa_pilot \
  --output research/data/qa_pilot/qa_combined_candidates_v2_reproduced.jsonl
```

A separate output filename is required because the CLI
uses exclusive file creation and does not overwrite
existing datasets.

The expected output contains:

- 35 QA candidates;
- 30 unique source IDs;
- seven publications;
- no exact duplicate questions;
- a successful JSONL readback verification.

The reproduced file can be checked with:

```bash
sha256sum research/data/qa_pilot/qa_combined_candidates_v2_reproduced.jsonl
```

For the same versioned inputs and assembly code,
the expected SHA-256 is:

`a380a375bb040ba60a42a89bf6863535a93fada7c600bb9c2038fa138dc800a4`

Omitting `--include-manual-topup` continues to reproduce
the unchanged 32-record v1 dataset with its original checksum.

This procedure reproduces the combined dataset
from its curated intermediate artifacts.

It does not rerun the original LLM generation
experiments or independently reproduce their outputs.

### 10.4. Automated validation

The project uses:

- `ruff` for formatting and static linting;
- `mypy` for static type checking;
- `pytest` for automated testing.

Tests cover source extraction, provenance consistency,
QA evidence alignment, review integrity, candidate
revision, content hashing, and dataset combination.

The full test suite passed on November 4, 2026:

```text
329 passed in 3.26s
```

Ruff formatting and linting checks passed.
Mypy checks passed for the modified source and test files.

The validation environment used Python 3.12.14
and the project's locked dependencies.

Automated validation establishes technical and
structural consistency. It does not replace
independent scientific review of QA content.
