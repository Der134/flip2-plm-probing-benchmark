# -*- coding: utf-8 -*-
"""Build patched copies of buggy science-skills scripts and generate unified .patch diffs."""
import difflib
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
UPSTREAM_SKILLS = (BASE / "science-skills" / "skills").resolve()
PATCHED_SKILLS = (BASE / "patched_skills").resolve()
PATCH_DIR = (BASE / "patches").resolve()

SKILLS_TO_COPY = (
    "clinvar_database",
    "gnomad_database",
    "literature_search_arxiv",
    "literature_search_europepmc",
    "ensembl_database",
    "uniprot_database",
    "alphafold_database_fetch_and_analyze",
    "quickgo_database",
)

PATCH_TARGETS = (
    "clinvar_database/scripts/clinvar_api.py",
    "gnomad_database/scripts/get_variant_frequency.py",
    "gnomad_database/scripts/get_gene_constraint.py",
    "gnomad_database/scripts/search_variants.py",
    "literature_search_arxiv/scripts/search_arxiv.py",
)


def build_patched_skills():
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    if PATCHED_SKILLS.exists():
        shutil.rmtree(PATCHED_SKILLS)
    PATCHED_SKILLS.mkdir(parents=True, exist_ok=True)

    for sk in SKILLS_TO_COPY:
        shutil.copytree(UPSTREAM_SKILLS / sk, PATCHED_SKILLS / sk)

    # 1. ClinVar VCV XML SimpleAllele/Measure SequenceLocation + *VCF attributes + esummary chunking
    cv_rel = "clinvar_database/scripts/clinvar_api.py"
    cv_path = PATCHED_SKILLS / cv_rel
    cv_new = cv_path.read_text(encoding="utf-8")

    old_summary = """    # Ensure all IDs are strings and join with commas
    ids_str = ','.join(map(str, variant_ids))

    params = {'db': 'clinvar', 'id': ids_str, 'retmode': 'json'}

    response = self._request('esummary.fcgi', params)
    data = self._parse_json(response.content)

    result_data = data.get('result', {})
    uids = result_data.get('uids', [])

    summaries = []
    for uid in uids:
      var_data = result_data.get(uid, {})"""

    new_summary = """    # Chunk IDs (<=200 per GET) to avoid HTTP 414 URI Too Long on large searches.
    batched_records = []
    str_ids = [str(v) for v in variant_ids]
    for start in range(0, len(str_ids), 200):
      batch_ids = str_ids[start:start + 200]
      params = {'db': 'clinvar', 'id': ','.join(batch_ids), 'retmode': 'json'}
      response = self._request('esummary.fcgi', params)
      data = self._parse_json(response.content)
      result_data = data.get('result', {})
      for uid in result_data.get('uids', []):
        batched_records.append((uid, result_data.get(uid, {})))

    summaries = []
    for uid, var_data in batched_records:"""

    if old_summary not in cv_new:
        raise RuntimeError("old_summary not found")
    cv_new = cv_new.replace(old_summary, new_summary)

    old_allele = """    # Try SequenceLocation nodes (GRCh38 preferred)
    for loc in root.findall('.//SequenceLocation'):
      if loc.attrib.get('Assembly') == 'GRCh38':
        info['chromosome'] = loc.attrib.get('Chr')
        info['position_start'] = loc.attrib.get('start')
        info['position_stop'] = loc.attrib.get('stop')
        info['reference_allele'] = loc.attrib.get('referenceAllele')
        info['alternate_allele'] = loc.attrib.get('alternateAllele')
        break

    # Fallback: first SequenceLocation if GRCh38 not found
    if info['chromosome'] is None:
      first_loc = root.find('.//SequenceLocation')
      if first_loc is not None:
        info['chromosome'] = first_loc.attrib.get('Chr')
        info['position_start'] = first_loc.attrib.get('start')
        info['position_stop'] = first_loc.attrib.get('stop')
        info['reference_allele'] = first_loc.attrib.get('referenceAllele')
        info['alternate_allele'] = first_loc.attrib.get('alternateAllele')"""

    new_allele = """    # Target variant-level SequenceLocation (not Gene-level span which appears first in VCV XML)
    variant_locs = (
        root.findall('.//SimpleAllele/Location/SequenceLocation')
        + root.findall('.//Measure/SequenceLocation')
        + root.findall('.//SequenceLocation')
    )
    chosen = next(
        (loc for loc in variant_locs if loc.attrib.get('Assembly') == 'GRCh38'),
        variant_locs[0] if variant_locs else None,
    )
    if chosen is not None:
      info['chromosome'] = chosen.attrib.get('Chr')
      info['position_start'] = chosen.attrib.get('positionVCF') or chosen.attrib.get('start')
      info['position_stop'] = chosen.attrib.get('stop')
      info['reference_allele'] = chosen.attrib.get('referenceAlleleVCF') or chosen.attrib.get('referenceAllele')
      info['alternate_allele'] = chosen.attrib.get('alternateAlleleVCF') or chosen.attrib.get('alternateAllele')"""

    if old_allele not in cv_new:
        raise RuntimeError("old_allele not found")
    cv_new = cv_new.replace(old_allele, new_allele)

    old_sv = """    loc = None
    for seq_loc in root.findall('.//SequenceLocation'):
      if seq_loc.attrib.get('Assembly') == 'GRCh38':
        loc = seq_loc
        break
    if loc is None:
      loc = root.find('.//SequenceLocation')"""

    new_sv = """    sv_locs = (
        root.findall('.//SimpleAllele/Location/SequenceLocation')
        + root.findall('.//Measure/SequenceLocation')
        + root.findall('.//SequenceLocation')
    )
    loc = next(
        (seq_loc for seq_loc in sv_locs if seq_loc.attrib.get('Assembly') == 'GRCh38'),
        sv_locs[0] if sv_locs else None,
    )"""

    if old_sv not in cv_new:
        raise RuntimeError("old_sv not found")
    cv_new = cv_new.replace(old_sv, new_sv)
    cv_path.write_text(cv_new, encoding="utf-8")

    # 2. gnomAD relative --output os.makedirs fix
    for rel_str in (
        "gnomad_database/scripts/get_variant_frequency.py",
        "gnomad_database/scripts/get_gene_constraint.py",
        "gnomad_database/scripts/search_variants.py",
    ):
        p = PATCHED_SKILLS / rel_str
        orig = p.read_text(encoding="utf-8")
        patched = orig.replace(
            "os.makedirs(os.path.dirname(output_path), exist_ok=True)",
            "os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)",
        )
        if patched == orig:
            raise RuntimeError(f"Patch did not match in {rel_str}")
        p.write_text(patched, encoding="utf-8")

    # 3. arXiv single-JSON output fix (Issue #13)
    ax_p = PATCHED_SKILLS / "literature_search_arxiv/scripts/search_arxiv.py"
    ax_orig = ax_p.read_text(encoding="utf-8")
    old_ax = """    paper["authors"] = authors
    results.append(paper)

    print(
        json.dumps(
            {
                "status": "success",
                "results_count": len(results),
                "papers": results,
            },
            indent=2,
        )
    )"""
    new_ax = """    paper["authors"] = authors
    results.append(paper)

  print(
      json.dumps(
          {
              "status": "success",
              "results_count": len(results),
              "papers": results,
          },
          indent=2,
      )
  )"""
    if old_ax not in ax_orig:
        raise RuntimeError("old_ax not found")
    ax_p.write_text(ax_orig.replace(old_ax, new_ax), encoding="utf-8")

    # Write unified diffs
    for rel_str in PATCH_TARGETS:
        p_up = UPSTREAM_SKILLS / rel_str
        p_pa = PATCHED_SKILLS / rel_str
        a = p_up.read_text(encoding="utf-8").splitlines(keepends=True)
        b = p_pa.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(a, b, fromfile=f"a/skills/{rel_str}", tofile=f"b/skills/{rel_str}"))
        (PATCH_DIR / f"{Path(rel_str).stem}.patch").write_text(diff, encoding="utf-8")
    print("[OK] Built patched_skills/ and generated 5 .patch files.")


if __name__ == "__main__":
    build_patched_skills()
