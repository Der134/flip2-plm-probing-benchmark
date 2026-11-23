# Ready-to-File GitHub Issues for `google-deepmind/science-skills`

Per `CONTRIBUTING.md`, `google-deepmind/science-skills` currently does not accept external PRs and instead invites reproducible bug reports via GitHub Issues. Below are two verified, high-severity, zero-ambiguity issues discovered during our 50-task automated audit (`ScienceBench-Bio50`).

---

## Issue Draft 1 — ClinVar `evidence` Silently Returns Gene Bounding Box with `null` Ref/Alt Alleles (100% Failure Across SNVs/Indels)

**Title:**
`clinvar_database: clinvar_api.py evidence returns Gene SequenceLocation (null ref/alt) instead of SimpleAllele VCF coordinates + SKILL.md drift`

**Body:**
### Summary
In `skills/clinvar_database/scripts/clinvar_api.py`, `_extract_allele_info()` iterates over `root.findall('.//SequenceLocation')` and breaks on the first `Assembly="GRCh38"` element. In ClinVar VCV XML (`efetch.fcgi?rettype=vcv`), the **first** `<SequenceLocation Assembly="GRCh38">` belongs to the enclosing `<Gene>` element (e.g., the 53 kb–126 kb gene span) and has no allele attributes, whereas the variant's own coordinates live under `<SimpleAllele><Location><SequenceLocation>` using `positionVCF`, `referenceAlleleVCF`, and `alternateAlleleVCF`.

As a result, `clinvar_api.py evidence` **silently returns the full gene span and `reference_allele: null, alternate_allele: null`** across 10/10 tested ClinVar SNVs and indels, breaking downstream chaining to Ensembl VEP, gnomAD, or AlphaGenome.

### Minimal Reproduction
```bash
uv run skills/clinvar_database/scripts/clinvar_api.py evidence --variant_id 13913 --output ev_13913.json
```
Observed `allele_info` for `13913` (`NM_020975.6(RET):c.1833C>G (p.Cys611Trp)`):
```json
{
  "chromosome": "10",
  "position_start": "43077069",
  "position_stop": "43130351",
  "reference_allele": null,
  "alternate_allele": null,
  "cytogenetic_band": "10q11.21",
  "dbsnp_rsid": "rs80069458"
}
```
Expected (present in `<SimpleAllele><Location><SequenceLocation Assembly="GRCh38" ...>` in the same XML):
```json
{
  "chromosome": "10",
  "position_start": "43113629",
  "position_stop": "43113629",
  "reference_allele": "C",
  "alternate_allele": "G"
}
```
Verified on 10/10 diverse variants (`13913`, `12582`, `17661`, `37610`, `5042465`, `14198`, `12706`, `92689`, `13440`, `428889`): 0/10 succeed upstream; 10/10 return exact GRCh38 VCF coordinates after the patch below.

### Related Findings in `clinvar_database`
1. `_extract_structural_variant()` (line 598) has the same `root.findall('.//SequenceLocation')` gene-span precedence issue.
2. `get_interpretation_summary()` (line 254) joins all `variant_ids` into a single `GET` query without chunking (risks HTTP 414 on large `search` outputs).
3. `skills/clinvar_database/SKILL.md` line 77 states `--retmax` defaults to `200`, contradicting line 135 and `clinvar_api.py:679` where the default is `0` (fetch all).

### Verified Patch
```diff
--- a/skills/clinvar_database/scripts/clinvar_api.py
+++ b/skills/clinvar_database/scripts/clinvar_api.py
@@ -484,25 +484,22 @@
         'dbsnp_rsid': None,
     }
 
-    # Try SequenceLocation nodes (GRCh38 preferred)
-    for loc in root.findall('.//SequenceLocation'):
-      if loc.attrib.get('Assembly') == 'GRCh38':
-        info['chromosome'] = loc.attrib.get('Chr')
-        info['position_start'] = loc.attrib.get('start')
-        info['position_stop'] = loc.attrib.get('stop')
-        info['reference_allele'] = loc.attrib.get('referenceAllele')
-        info['alternate_allele'] = loc.attrib.get('alternateAllele')
-        break
-
-    # Fallback: first SequenceLocation if GRCh38 not found
-    if info['chromosome'] is None:
-      first_loc = root.find('.//SequenceLocation')
-      if first_loc is not None:
-        info['chromosome'] = first_loc.attrib.get('Chr')
-        info['position_start'] = first_loc.attrib.get('start')
-        info['position_stop'] = first_loc.attrib.get('stop')
-        info['reference_allele'] = first_loc.attrib.get('referenceAllele')
-        info['alternate_allele'] = first_loc.attrib.get('alternateAllele')
+    # Target variant-level SequenceLocation (not Gene-level span which appears first in VCV XML)
+    variant_locs = (
+        root.findall('.//SimpleAllele/Location/SequenceLocation')
+        + root.findall('.//Measure/SequenceLocation')
+        + root.findall('.//SequenceLocation')
+    )
+    chosen = next(
+        (loc for loc in variant_locs if loc.attrib.get('Assembly') == 'GRCh38'),
+        variant_locs[0] if variant_locs else None,
+    )
+    if chosen is not None:
+      info['chromosome'] = chosen.attrib.get('Chr')
+      info['position_start'] = chosen.attrib.get('positionVCF') or chosen.attrib.get('start')
+      info['position_stop'] = chosen.attrib.get('stop')
+      info['reference_allele'] = chosen.attrib.get('referenceAlleleVCF') or chosen.attrib.get('referenceAllele')
+      info['alternate_allele'] = chosen.attrib.get('alternateAlleleVCF') or chosen.attrib.get('alternateAllele')
```

---

## Issue Draft 2 — `gnomad_database` (3 Scripts) & `alphagenome_single_variant_analysis` Crash with `FileNotFoundError: ''` on Relative `--output` Filenames

**Title:**
`gnomad_database & alphagenome_single_variant_analysis: os.makedirs(os.path.dirname(output_path)) crashes on relative --output filenames shown in SKILL.md`

**Body:**
### Summary
Every CLI example in `skills/gnomad_database/SKILL.md` passes a bare relative filename to `--output` (e.g., `--output variant_frequency.json`, `--output BRCA1_constraint.json`). However, all three `gnomad_database` scripts—and `generate_ontology_mapping.py` in `alphagenome_single_variant_analysis`—call:
```python
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```
When `output_path` has no directory prefix, `os.path.dirname(output_path)` returns `""`, causing `os.makedirs("", exist_ok=True)` to crash with `FileNotFoundError` **after** already completing the rate-limited network call.

### Affected Locations (5 call sites across 4 scripts)
- `skills/gnomad_database/scripts/get_variant_frequency.py:152`
- `skills/gnomad_database/scripts/get_gene_constraint.py:71`
- `skills/gnomad_database/scripts/search_variants.py:104`
- `skills/gnomad_database/scripts/search_variants.py:171`
- `skills/alphagenome_single_variant_analysis/scripts/generate_ontology_mapping.py:142`

(Note: `string_database/scripts/string_cli.py:75` and `unibind_database/scripts/unibind_api.py:60` already avoid this bug using `os.path.dirname(os.path.abspath(output_file))`.)

### Minimal Reproduction
```bash
uv run skills/gnomad_database/scripts/get_variant_frequency.py --rsid rs1800562 --output variant_frequency.json
# -> FileNotFoundError: [Errno 2] No such file or directory: ''
```

### Verified Fix
```diff
-  os.makedirs(os.path.dirname(output_path), exist_ok=True)
+  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
```
Verified on 10/10 gnomAD variant-frequency and gene-constraint CLI invocations (0/10 pass upstream → 10/10 pass after fix).
