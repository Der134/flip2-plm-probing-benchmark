# Direction F: Empirical Reliability Audit of `google-deepmind/science-skills` (v1.4.0)

**Benchmark:** `ScienceBench-Bio50` (50 deterministic multi-database bioinformatics tasks across 5 families)
**Upstream Snapshot:** `google-deepmind/science-skills` `main` (v1.4.0, commit audited `2026-11-24`)
**Reproducible Harness:** `patch_builder.py` + `run_science_bench_50.py` (`results/bench50_results.csv`, `results/bench50_summary.json`)

---

## 1. Executive Summary

We audited all **40 skills / 66 Python wrapper scripts** in `google-deepmind/science-skills` (v1.4.0) via static analysis and a live 50-task A/B evaluation (`ScienceBench-Bio50`) executing every wrapper through `uv run` strictly following each skill's documented `SKILL.md` CLI contract.

- **Upstream v1.4.0 pass rate:** **6 / 50 (12.0%)**
- **Patched pass rate (3 minimal script patches + `POLITE_HTTP_USER_AGENT`):** **45 / 50 (90.0%)** (+78.0 pp; remaining 5 misses are external API latency/404 coverage, zero script crashes or silent data corruption)

| Task Family (10 tasks each) | Upstream v1.4.0 | Patched | Primary Upstream Failure Mode |
|---|---:|---:|---|
| A. `clinvar_coord` (T01–T10) | **0 / 10** | **10 / 10** | **Silent data corruption (10/10)**: returns Gene span + `ref=null, alt=null` |
| B. `gnomad_cli` (T11–T20) | **0 / 10** | **10 / 10** | **Runtime crash (10/10)**: `os.makedirs("")` `FileNotFoundError` on `SKILL.md` relative `--output` |
| C. `cross_db_chain` (T21–T30) | **0 / 10** | **9 / 10** | **Cascading chain failure (10/10)** from ClinVar `null` alleles (1 transient VEP timeout in patched) |
| D. `protein_struct` (T31–T40) | **0 / 10** | **9 / 10** | **HTTP 403 (10/10)** empty User-Agent on AlphaFold DB (1 AFDB 404 on 2,768-aa `P01266` in patched) |
| E. `lit_and_go` (T41–T50) | **6 / 10** | **7 / 10** | **Invalid concatenated JSON (4/5 arXiv)** when `results > 1` (Issue #13) |
| **Total** | **6 / 50 (12%)** | **45 / 50 (90%)** | **40/44 upstream failures eliminated by 3 surgical patches + UA config** |

---

## 2. Verified High-Severity Bugs & Failure Taxonomy

### Bug 1 — Silent Variant Coordinate Corruption in `clinvar_database/scripts/clinvar_api.py` (Severity: Critical)
- **Root Cause:** `_extract_allele_info()` (lines 488–495) scans `root.findall('.//SequenceLocation')` and breaks on the first `Assembly="GRCh38"` match. In ClinVar VCV XML, the first GRCh38 `<SequenceLocation>` belongs to `<Gene>` (full 53–126 kb gene bounding box, no alleles), whereas the variant's coordinates live in `<SimpleAllele>/<Location>/<SequenceLocation>` using `positionVCF`, `referenceAlleleVCF`, and `alternateAlleleVCF`.
- **Impact:** 10/10 tested SNVs and indels silently returned the gene span with `reference_allele: null, alternate_allele: null`, causing 10/10 downstream `ClinVar → Ensembl VEP` workflows to break.
- **Additional findings:** `_extract_structural_variant()` (line 598) has the same gene-precedence bug; `get_interpretation_summary()` (line 254) passes unbounded IDs via `GET` without chunking; `SKILL.md` line 77 contradicts line 135 (`--retmax` default `200` vs `0`).
- **Patch:** `patches/clinvar_api.patch` (0/10 → 10/10 on `clinvar_coord`; 0/10 → 9/10 on `cross_db_chain`).

### Bug 2 — `os.makedirs(os.path.dirname(output_path))` Crash Across 4 Scripts / 2 Skills (Severity: High)
- **Locations:**
  - `skills/gnomad_database/scripts/get_variant_frequency.py:152`
  - `skills/gnomad_database/scripts/get_gene_constraint.py:71`
  - `skills/gnomad_database/scripts/search_variants.py:104` and `:171`
  - `skills/alphagenome_single_variant_analysis/scripts/generate_ontology_mapping.py:142`
- **Impact:** Every example in `gnomad_database/SKILL.md` uses a relative filename (`--output variant_frequency.json`), where `os.path.dirname(...)` returns `""` and crashes **after** spending a rate-limited GraphQL query.
- **Patch:** `os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)` (`patches/get_variant_frequency.patch`, `patches/get_gene_constraint.patch`, `patches/search_variants.patch`; 0/10 → 10/10).

### Bug 3 — Known Upstream Issues & Cross-Skill Friction
- **`literature_search_arxiv/scripts/search_arxiv.py:149-158` (Open Issue #13):** `print(json.dumps(...))` sits inside the `for entry` loop, emitting $N$ concatenated JSON objects whenever $N > 1$ (failed 4/5 queries upstream; 100% valid JSON after dedent patch `patches/search_arxiv.patch`).
- **`alphafold_database_fetch_and_analyze/scripts/fetch_structure.py` (Closed Issue #2):** Intentionally leaves `POLITE_HTTP_USER_AGENT` empty so agents adapt on retry, causing 10/10 first-call HTTP 403s until the agent parses the error hint and sets `POLITE_HTTP_USER_AGENT`.
- **Cross-Skill Coordinate Convention Drift:** `clinvar_database/SKILL.md` mandates `chr:pos:ref>alt`, whereas `ensembl_database/scripts/ensembl_api.py vep` expects 4-colon `chr:pos:ref:alt` (passing `chr:pos:ref>alt` falls through to `/vep/human/hgvs/` and returns HTTP 400).

---

## 3. Upstream Contribution Strategy & Live Issue Links

Because `google-deepmind/science-skills/CONTRIBUTING.md` explicitly states that external PRs are not currently accepted and asks contributors to open reproducible GitHub Issues, we submitted two verified, patch-backed issues directly upstream (avoiding duplicate filing of already-open Issue #13):

- **Issue #15 (ClinVar silent coordinate corruption + `SKILL.md` drift):** https://github.com/google-deepmind/science-skills/issues/15
- **Issue #16 (`gnomad_database` & `alphagenome_single_variant_analysis` `os.makedirs` crash on relative `--output`):** https://github.com/google-deepmind/science-skills/issues/16

Full issue texts are also archived in [`ISSUE_DRAFTS.md`](ISSUE_DRAFTS.md).
