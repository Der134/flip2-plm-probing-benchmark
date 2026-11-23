# -*- coding: utf-8 -*-
"""Direction F: ScienceBench-Bio50 — Reproducible A/B evaluation of science-skills v1.4.0 vs patched."""
import csv
import io
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SKILLS_ROOTS = {
    "upstream": (BASE / "science-skills" / "skills").resolve(),
    "patched": (BASE / "patched_skills").resolve(),
}
SANDBOXES = {
    "upstream": (BASE / "sandbox_upstream").resolve(),
    "patched": (BASE / "sandbox_patched").resolve(),
}
for sb in SANDBOXES.values():
    sb.mkdir(parents=True, exist_ok=True)

SCRIPT_WHITELIST = {
    "clinvar": "clinvar_database/scripts/clinvar_api.py",
    "gnomad_var": "gnomad_database/scripts/get_variant_frequency.py",
    "gnomad_con": "gnomad_database/scripts/get_gene_constraint.py",
    "arxiv": "literature_search_arxiv/scripts/search_arxiv.py",
    "epmc": "literature_search_europepmc/scripts/europepmc_api.py",
    "ensembl": "ensembl_database/scripts/ensembl_api.py",
    "uniprot": "uniprot_database/scripts/uniprot_tools.py",
    "af_fetch": "alphafold_database_fetch_and_analyze/scripts/fetch_structure.py",
    "af_plddt": "alphafold_database_fetch_and_analyze/scripts/analyze_plddt.py",
    "quickgo": "quickgo_database/scripts/quickgo_tool.py",
}

CLINVAR_CASES = [
    {"vid": "13913",   "gene": "RET",    "chr": "10", "pos": "43113629",  "ref": "C",     "alt": "G"},
    {"vid": "12582",   "gene": "KRAS",   "chr": "12", "pos": "25245350",  "ref": "C",     "alt": "T"},
    {"vid": "17661",   "gene": "BRCA1",  "chr": "17", "pos": "43106487",  "ref": "A",     "alt": "C"},
    {"vid": "37610",   "gene": "BRCA1",  "chr": "17", "pos": "43071135",  "ref": "TATGT", "alt": "G"},
    {"vid": "5042465", "gene": "BRCA1",  "chr": "17", "pos": "43094819",  "ref": "GT",    "alt": "G"},
    {"vid": "14198",   "gene": "MPZ",    "chr": "1",  "pos": "161306789", "ref": "C",     "alt": "T"},
    {"vid": "12706",   "gene": "TG",     "chr": "8",  "pos": "132873222", "ref": "G",     "alt": "A"},
    {"vid": "92689",   "gene": "MYBPC3", "chr": "11", "pos": "47342138",  "ref": "TAC",   "alt": "T"},
    {"vid": "13440",   "gene": "TTR",    "chr": "18", "pos": "31592974",  "ref": "G",     "alt": "C"},
    {"vid": "428889",  "gene": "TP53",   "chr": "17", "pos": "7673835",   "ref": "C",     "alt": "A"},
]

GNOMAD_VARIANTS = [
    ("rsid", "rs1800562", "6-26092913-G-A"),
    ("rsid", "rs11591147", "1-55039974-G-T"),
    ("rsid", "rs28897672", "17-43106487-A-C"),
    ("variant_id", "1-55039974-G-T", "1-55039974-G-T"),
    ("variant_id", "6-26092913-G-A", "6-26092913-G-A"),
]
GNOMAD_GENES = ["BRCA1", "TP53", "RET", "KRAS", "TTR"]

PROTEINS = [
    ("P04637", "TP53", 393),
    ("P07949", "RET", 1114),
    ("P38398", "BRCA1", 1863),
    ("P01116", "KRAS", 189),
    ("P00520", "ABL1", 1130),
    ("P02766", "TTR", 147),
    ("P25189", "MPZ", 248),
    ("P01266", "TG", 2768),
    ("Q14896", "MYBPC3", 1274),
    ("Q9NQ94", "A1CF", 594),
]

ARXIV_QUERIES = [
    ("ti:ESM-2 AND ti:protein", 3),
    ("ti:AlphaFold AND abs:structure", 3),
    ("ti:ClinVar AND abs:variant", 3),
    ("au:Dallago AND ti:FLIP", 2),
    ("ti:LoRA AND ti:protein", 3),
]
EUROPEPMC_QUERIES = [
    "FLIP protein fitness benchmark",
    "AlphaMissense pathogenicity ClinVar",
    "ESM-2 protein language model",
]
QUICKGO_TERMS = ["GO:0003677", "GO:0005524"]


def safe_fname(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name) or ".." in name:
        raise ValueError(f"unsafe filename: {name!r}")
    return name


def run_uv(mode: str, script_key: str, args: list[str], env: dict, timeout: int = 45):
    skills_root = SKILLS_ROOTS[mode]
    sandbox = SANDBOXES[mode]
    script_p = (skills_root / SCRIPT_WHITELIST[script_key]).resolve()
    if not script_p.is_relative_to(skills_root):
        raise ValueError("script escapes root")
    t0 = time.time()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as f_out, \
         tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as f_err:
        p = subprocess.Popen(
            ["uv", "run", str(script_p), *args],
            cwd=str(sandbox),
            env=env,
            shell=False,
            stdout=f_out,
            stderr=f_err,
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=5)
            except Exception:
                pass
            return -1, "", f"TIMEOUT after {timeout}s", round(time.time() - t0, 2)
        f_out.seek(0)
        f_err.seek(0)
        return rc, f_out.read(), f_err.read(), round(time.time() - t0, 2)


def run_suite(mode: str):
    sandbox = SANDBOXES[mode]
    env = os.environ.copy()
    if mode == "patched":
        env["POLITE_HTTP_USER_AGENT"] = "ScienceBench-Bio50/1.0 (https://github.com/google-deepmind/science-skills)"
    else:
        env.pop("POLITE_HTTP_USER_AGENT", None)

    records = []
    cv_coords = {}

    for idx, case in enumerate(CLINVAR_CASES, 1):
        tid = f"T{idx:02d}"
        out_name = safe_fname(f"{tid}_cv.json")
        out_f = (sandbox / out_name).resolve()
        if not out_f.is_relative_to(sandbox):
            raise ValueError("output escapes sandbox")
        rc, out, err, sec = run_uv(mode, "clinvar", ["evidence", "--variant_id", case["vid"], "--output", out_name], env)
        passed, ftype, detail = False, "", ""
        if rc == 0 and out_f.exists():
            d = json.loads(out_f.read_text(encoding="utf-8"))
            ai = d.get("allele_info") or {}
            cv_coords[case["vid"]] = ai
            got = (str(ai.get("chromosome")), str(ai.get("position_start")), ai.get("reference_allele"), ai.get("alternate_allele"))
            exp = (case["chr"], case["pos"], case["ref"], case["alt"])
            if got == exp:
                passed = True
            else:
                ftype = "silent_data_corruption"
                detail = f"got {got} != expected {exp}"
        else:
            ftype = "runtime_crash"
            detail = (err or out)[-240:].strip()
        records.append({"task_id": tid, "family": "clinvar_coord", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} clinvar/{case['vid']}: pass={passed} {detail}")

    for idx, (kind, qval, exp_vid) in enumerate(GNOMAD_VARIANTS, 11):
        tid = f"T{idx:02d}"
        flag = "--rsid" if kind == "rsid" else "--variant_id"
        rel_out = safe_fname(f"{tid}_gnvar.json")
        out_f = (sandbox / rel_out).resolve()
        if not out_f.is_relative_to(sandbox):
            raise ValueError("output escapes sandbox")
        rc, out, err, sec = run_uv(mode, "gnomad_var", [flag, qval, "--output", rel_out], env)
        passed, ftype, detail = False, "", ""
        if rc == 0 and out_f.exists():
            d = json.loads(out_f.read_text(encoding="utf-8"))
            vnode = (d.get("data") or {}).get("variant")
            if vnode and vnode.get("variant_id") == exp_vid:
                passed = True
            else:
                ftype = "api_empty_or_mismatch"
                detail = f"variant={vnode.get('variant_id') if vnode else None}"
        else:
            ftype = "cli_relative_output_crash" if "FileNotFoundError" in (err + out) else "runtime_crash"
            detail = (err or out).splitlines()[-1][:180] if (err or out) else f"rc={rc}"
        records.append({"task_id": tid, "family": "gnomad_cli", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} gnomad_var/{qval}: pass={passed} {ftype}")

    for idx, gene in enumerate(GNOMAD_GENES, 16):
        tid = f"T{idx:02d}"
        rel_out = safe_fname(f"{tid}_gncon.json")
        out_f = (sandbox / rel_out).resolve()
        if not out_f.is_relative_to(sandbox):
            raise ValueError("output escapes sandbox")
        rc, out, err, sec = run_uv(mode, "gnomad_con", ["--gene", gene, "--output", rel_out], env)
        passed, ftype, detail = False, "", ""
        if rc == 0 and out_f.exists():
            d = json.loads(out_f.read_text(encoding="utf-8"))
            gnode = (d.get("data") or {}).get("gene")
            if gnode and gnode.get("symbol") == gene and gnode.get("gnomad_constraint") is not None:
                passed = True
            else:
                ftype = "api_empty_or_mismatch"
                detail = f"gene={gene}"
        else:
            ftype = "cli_relative_output_crash" if "FileNotFoundError" in (err + out) else "runtime_crash"
            detail = (err or out).splitlines()[-1][:180] if (err or out) else f"rc={rc}"
        records.append({"task_id": tid, "family": "gnomad_cli", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} gnomad_gene/{gene}: pass={passed} {ftype}")

    for idx, case in enumerate(CLINVAR_CASES, 21):
        tid = f"T{idx:02d}"
        ai = cv_coords.get(case["vid"], {})
        c_chr, c_pos, c_ref, c_alt = ai.get("chromosome"), ai.get("position_start"), ai.get("reference_allele"), ai.get("alternate_allele")
        passed, ftype, detail, sec = False, "", "", 0.0
        if not all([c_chr, c_pos, c_ref, c_alt]):
            ftype = "upstream_coord_corruption"
            detail = f"missing ref/alt from ClinVar ({c_chr}:{c_pos} ref={c_ref} alt={c_alt})"
        else:
            if len(c_ref) == 1 and len(c_alt) == 1:
                var_str = f"{c_chr}:{c_pos}:{c_ref}:{c_alt}"
            elif len(c_ref) > 1 and len(c_alt) == 1 and c_ref.startswith(c_alt):
                del_start = int(c_pos) + 1
                del_len = len(c_ref) - 1
                var_str = f"{c_chr}:{del_start}:{'N'*del_len}:-"
            else:
                var_str = f"{c_chr}:{c_pos}:{c_ref}:{c_alt}"
            out_name = safe_fname(f"{tid}_vep.json")
            out_f = (sandbox / out_name).resolve()
            if not out_f.is_relative_to(sandbox):
                raise ValueError("output escapes sandbox")
            rc, out, err, sec = run_uv(mode, "ensembl", ["vep", var_str, "--output", out_name], env)
            if rc == 0 and out_f.exists():
                d = json.loads(out_f.read_text(encoding="utf-8"))
                msq = d[0].get("most_severe_consequence") if isinstance(d, list) and d else None
                if msq:
                    passed = True
                    detail = f"msq={msq}"
                else:
                    ftype = "empty_vep"
            else:
                ftype = "vep_timeout_heavy_plugins" if "TIMEOUT" in err else "vep_api_error"
                detail = (err or out).splitlines()[-1][:180]
        records.append({"task_id": tid, "family": "cross_db_chain", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} chain/{case['vid']}: pass={passed} {ftype} {detail}")

    for idx, (acc, gene, exp_len) in enumerate(PROTEINS, 31):
        tid = f"T{idx:02d}"
        rc1, out1, err1, s1 = run_uv(mode, "uniprot", ["get", acc, "--format", "json"], env)
        af_dirname = safe_fname(f"{tid}_af_{acc}")
        af_dir = (sandbox / af_dirname).resolve()
        if not af_dir.is_relative_to(sandbox):
            raise ValueError("af_dir escapes sandbox")
        rc2, out2, err2, s2 = run_uv(mode, "af_fetch", [acc, "-o", af_dirname], env)
        sec = round(s1 + s2, 2)
        passed, ftype, detail = False, "", ""
        if rc1 != 0:
            ftype = "uniprot_error"
            detail = (err1 or out1)[-180:]
        elif rc2 != 0:
            ftype = "http_403_missing_user_agent" if "403" in (out2 + err2) else "afdb_error"
            detail = "HTTP 403 empty User-Agent" if "403" in (out2 + err2) else (err2 or out2)[-180:]
        else:
            meta_files = list(af_dir.glob("*-metadata.json"))
            if meta_files:
                meta_name = safe_fname(meta_files[0].name)
                rc3, out3, err3, s3 = run_uv(mode, "af_plddt", [f"{af_dirname}/{meta_name}"], env)
                sec = round(sec + s3, 2)
                if rc3 == 0 and "pLDDT" in out3:
                    passed = True
                    detail = f"len={exp_len}"
                else:
                    ftype = "plddt_parse_error"
            else:
                ftype = "afdb_missing_metadata"
        records.append({"task_id": tid, "family": "protein_struct", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} struct/{acc}: pass={passed} {ftype}")

    for idx, (q, k) in enumerate(ARXIV_QUERIES, 41):
        tid = f"T{idx:02d}"
        rc, out, err, sec = run_uv(mode, "arxiv", ["--query", q, "--max_results", str(k)], env)
        passed, ftype, detail = False, "", ""
        if rc == 0:
            try:
                d = json.loads(out)
                if d.get("status") == "success" and d.get("results_count") == len(d.get("papers", [])) and d.get("results_count", 0) > 0:
                    passed = True
                else:
                    ftype = "arxiv_count_mismatch"
            except json.JSONDecodeError as e:
                ftype = "concatenated_invalid_json"
                detail = str(e)
        else:
            ftype = "runtime_crash"
            detail = (err or out)[-180:]
        records.append({"task_id": tid, "family": "lit_and_go", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} arxiv/{q[:18]}: pass={passed} {ftype}")

    for idx, q in enumerate(EUROPEPMC_QUERIES, 46):
        tid = f"T{idx:02d}"
        out_name = safe_fname(f"{tid}_epmc.json")
        out_f = (sandbox / out_name).resolve()
        if not out_f.is_relative_to(sandbox):
            raise ValueError("output escapes sandbox")
        rc, out, err, sec = run_uv(mode, "epmc", ["search", q, "--max_results", "3", "--output", out_name], env)
        passed, ftype, detail = False, "", ""
        if rc == 0 and out_f.exists():
            d = json.loads(out_f.read_text(encoding="utf-8"))
            hits = d.get("results") or d.get("resultList", {}).get("result", [])
            if hits:
                passed = True
            else:
                ftype = "empty_results"
        else:
            ftype = "runtime_crash"
            detail = (err or out)[-180:]
        records.append({"task_id": tid, "family": "lit_and_go", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} epmc/{q[:18]}: pass={passed} {ftype}")

    for idx, go_id in enumerate(QUICKGO_TERMS, 49):
        tid = f"T{idx:02d}"
        out_name = safe_fname(f"{tid}_qgo.json")
        out_f = (sandbox / out_name).resolve()
        if not out_f.is_relative_to(sandbox):
            raise ValueError("output escapes sandbox")
        rc, out, err, sec = run_uv(mode, "quickgo", ["go", "terms", "--ids", go_id, "--output", out_name], env)
        passed, ftype, detail = False, "", ""
        if rc == 0 and out_f.exists():
            d = json.loads(out_f.read_text(encoding="utf-8"))
            res = d.get("results") or []
            if res and res[0].get("id") == go_id:
                passed = True
            else:
                ftype = "empty_or_mismatch"
        else:
            ftype = "runtime_crash"
            detail = (err or out)[-180:]
        records.append({"task_id": tid, "family": "lit_and_go", "mode": mode, "passed": passed, "failure_type": ftype, "seconds": sec, "detail": detail})
        print(f"  [{mode:8s}] {tid} quickgo/{go_id}: pass={passed} {ftype}")

    return records


def main():
    out_dir = (BASE / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Running ScienceBench-Bio50: UPSTREAM v1.4.0 ===")
    up_rec = run_suite("upstream")
    print("\n=== Running ScienceBench-Bio50: PATCHED ===")
    pa_rec = run_suite("patched")

    all_rec = up_rec + pa_rec
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["task_id", "family", "mode", "passed", "failure_type", "seconds", "detail"])
    w.writeheader()
    w.writerows(all_rec)
    csv_path = (out_dir / "bench50_results.csv").resolve()
    if not csv_path.is_relative_to(out_dir):
        raise ValueError("csv_path escapes out_dir")
    csv_path.write_text(buf.getvalue(), encoding="utf-8")

    summary = {}
    for m, recs in (("upstream", up_rec), ("patched", pa_rec)):
        by_fam = {}
        for r in recs:
            fam = r["family"]
            by_fam.setdefault(fam, {"n": 0, "pass": 0, "failures": {}})
            by_fam[fam]["n"] += 1
            if r["passed"]:
                by_fam[fam]["pass"] += 1
            else:
                ft = r["failure_type"] or "unknown"
                by_fam[fam]["failures"][ft] = by_fam[fam]["failures"].get(ft, 0) + 1
        tot_pass = sum(1 for r in recs if r["passed"])
        summary[m] = {"total_tasks": len(recs), "passed": tot_pass, "pass_rate": round(tot_pass / len(recs), 4), "by_family": by_fam}

    sum_path = (out_dir / "bench50_summary.json").resolve()
    if not sum_path.is_relative_to(out_dir):
        raise ValueError("sum_path escapes out_dir")
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
