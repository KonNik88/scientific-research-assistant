# Research 08.5b — Corpus and QA Dataset Exploratory Analysis

**Project:** Scientific Research Assistant
**Checkpoint:** November 10, 2026
**Report status:** Preliminary, v0.1
**Analysis snapshot:** November 2, 2026

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

## 7. QA Error Analysis

The following error categories were observed:

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
   Results are not representative of arbitrary scholarly
   documents.

2. Paragraphs are extracted as raw LaTeX structures.
   Their lengths do not directly correspond to plain-text
   token counts or retrieval chunk sizes.

3. The QA experiments use small, deliberately selected
   samples. They do not establish statistically reliable
   model rankings.

4. Several paragraphs depend on neighboring equations,
   tables, figures, or lists.

5. Review labels are preliminary and have not undergone
   independent human adjudication.

6. The current QA sets are not a complete retrieval benchmark.
   Neither retrieval recall nor RAG answer quality has been
   evaluated at this stage.

7. Some research data are stored in a Git-ignored directory.
   A versioned experiment manifest and explicit artifact
   preservation procedure are still required for complete
   reproducibility.

## 9. Next Steps

Before the November 10 checkpoint:

1. Extend the manually reviewed QA collection.
2. Preserve original model outputs separately from
   human-revised QA records.
3. Increase coverage across papers and scientific sections.
4. Add figures summarizing corpus characteristics and
   QA quality.
5. Finalize the experiment manifest and dataset documentation.
6. Review and confirm preliminary scientific annotations.

Subsequent research will address:

- retrieval-oriented text normalization and chunking;
- BM25, dense, and hybrid retrieval baselines;
- evidence attribution;
- citation-aware retrieval;
- grounded RAG generation;
- agentic components where empirically justified.

## 10. Reproducibility

The analysis is implemented in:

- `research/spikes/eda_qa_corpus.py`
- `research/spikes/eda_qa_generation.py`

Both modules accept explicit input files and produce
deterministic JSON summaries.

Unit tests cover aggregation logic, boundary conditions,
record uniqueness, provenance consistency, and review
integrity.

At the current repository snapshot, the full test suite
contains 267 passing tests.