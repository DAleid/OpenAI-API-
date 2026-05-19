"""
preprocessing_pipeline.py — HR data pipeline: Excel → JSON / CSV

Reads HR assessment Excel files, normalises the data, computes per-employee
alignment, and writes cleaned_hr_data.json and cleaned_hr_data.csv.

Self-contained: no backend imports. All config comes from JSON files.

Usage:
    python scripts/preprocessing_pipeline.py
    python scripts/preprocessing_pipeline.py --input-dir scripts/data/
    python scripts/preprocessing_pipeline.py file1.xlsx file2.xlsx
    python scripts/preprocessing_pipeline.py --input-dir scripts/data/ --output backend/hrData/cleaned_hr_data.json
    python scripts/preprocessing_pipeline.py --input-dir scripts/data/ --stage load
    python scripts/preprocessing_pipeline.py --input-dir scripts/data/ --stage normalize
"""
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# Configuration

_SCRIPTS_DIR  = Path(__file__).parent
_BACKEND_DATA = Path(__file__).parent.parent / "backend" / "data"

COLUMNS      = json.loads((_SCRIPTS_DIR / "columns.json").read_text(encoding="utf-8"))
TRANSLATIONS = json.loads((_SCRIPTS_DIR / "translations.json").read_text(encoding="utf-8"))
ORG          = json.loads((_BACKEND_DATA.parent / "refs" / "org_structure.json").read_text(encoding="utf-8"))

DEFAULT_INPUT  = Path(__file__).parent / "data"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "backend" / "hrData" / "cleaned_hr_data.json"

HEADER_ROW      = 6
SHEET_PREFIX    = "hr template"
VP_SHEET_MARKER = "مكتب نائب"

DEGREE_ORDER = ["PhD", "Master", "Bachelor", "Diploma", "High School", "Primary School", "Literate"]
NOT_IN_ORG   = {"Resigned", "Transferred", "Terminated"}
STOP_WORDS   = {"and", "of", "the", "in", "for", "a", "an", "with", "to", "at", "by", "on"}

SECTOR_FALLBACK: dict[str, list[str]] = {
    "Health":                       ["health", "medical", "clinical", "biological", "genomics"],
    "Sustainability & Environment": ["environment", "climate", "water", "ecology", "sustainability"],
    "Energy & Industry":            ["energy", "materials", "manufacturing", "industrial", "robotics"],
    "Economies of the Future":      ["digital", "data", "technology", "innovation", "artificial"],
    "Enabling Infrastructure":      ["computing", "metrology", "standards", "infrastructure", "calibration"],
}

SECTOR_MAP        = TRANSLATIONS["sectors"]
INSTITUTE_MAP     = TRANSLATIONS["institutes"]
CAREER_PATH_MAP   = TRANSLATIONS["career_paths"]
DEGREE_MAP        = TRANSLATIONS["degrees"]
GENDER_MAP        = TRANSLATIONS["genders"]
STATUS_MAP        = TRANSLATIONS.get("statuses", {})
EXECUTIVE_ROLES   = set(TRANSLATIONS["executive_roles"])
VP_OFFICE_NAMES   = set(TRANSLATIONS.get("vp_office_institutes", []))
INSTITUTE_DOMAINS = TRANSLATIONS["institute_domains"]

TEMPLATE_RENAME  = COLUMNS["template_rename"]
GENERIC_VARIANTS = COLUMNS["generic_variants"]
PRIVACY_DROP     = COLUMNS["privacy_drop"]

REQUIRED_COLUMNS = [
    "sector", "institute", "career_path", "degree", "gender",
    "focus_1_area", "focus_1_pct",
    "focus_2_area", "focus_2_pct",
    "focus_3_area", "focus_3_pct",
]


# Org structure — derived from org_structure.json

def _build_org_lookups(org: dict) -> dict:
    sectors             = []
    sector_colors       = {}
    institutes          = {}
    focus_area_tiers    = {}
    sector_focus_lower  = {}
    institute_to_sector = {}

    for division in org.get("divisions", []):
        if division.get("division_en") != "R&D":
            continue
        for sector in division.get("sectors", []):
            name = sector.get("sector_en", "").strip()
            if not name:
                continue
            sectors.append(name)
            sector_colors[name] = sector.get("color", "#888")
            inst_names = [
                inst["name_en"].strip()
                for inst in sector.get("institutes", [])
                if inst.get("name_en")
            ]
            institutes[name] = inst_names
            for inst_name in inst_names:
                institute_to_sector[inst_name] = name
            focus_entries = sector.get("research_focus", [])
            sector_focus_lower[name] = {f["area"].lower() for f in focus_entries if f.get("area")}
            for f in focus_entries:
                if f.get("area"):
                    focus_area_tiers[f["area"].strip()] = f.get("tier", "Tier3")

    return {
        "sectors":             sectors,
        "sector_colors":       sector_colors,
        "institutes":          institutes,
        "focus_area_tiers":    focus_area_tiers,
        "sector_focus_lower":  sector_focus_lower,
        "focus_areas":         list(focus_area_tiers.keys()),
        "institute_to_sector": institute_to_sector,
    }


_org                = _build_org_lookups(ORG)
SECTORS             = _org["sectors"]
SECTOR_COLORS       = _org["sector_colors"]
INSTITUTES          = _org["institutes"]
FOCUS_AREA_TIERS    = _org["focus_area_tiers"]
SECTOR_FOCUS_LOWER  = _org["sector_focus_lower"]
FOCUS_AREAS         = _org["focus_areas"]
INSTITUTE_TO_SECTOR = _org["institute_to_sector"]


# Stage 1 — Load

def load_data(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    log.info("Stage 1 — Load")

    all_frames  = []
    all_errors  = []
    all_dropped = []
    sheet_count = 0

    for path in files:
        if path.name.startswith("~$"):
            continue
        log.info("  Reading %s", path.name)
        try:
            frames, errors, dropped = _read_file(path)
            all_frames.extend(frames)
            all_errors.extend(errors)
            sheet_count += len(frames)
            if not dropped.empty:
                all_dropped.append(dropped)
            for err in errors:
                log.warning("  Skipped '%s' in %s: %s", err.get("sheet", "?"), path.name, err.get("reason", ""))
        except Exception as e:
            log.error("Failed to read %s: %s", path.name, e)
            all_errors.append({"file": path.name, "reason": str(e)})

    if not all_frames:
        raise ValueError("No valid HR data found in the provided files.")

    df = pd.concat(all_frames, ignore_index=True)
    log.info("  Loaded %d rows from %d sheet(s) across %d file(s)", len(df), sheet_count, len(files))

    meta = {
        "files_loaded":  len(files),
        "sheets_loaded": sheet_count,
        "rows_loaded":   len(df),
        "errors":        all_errors,
        "dropped_rows":  pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame(),
    }
    return df, meta


def _read_file(path: Path) -> tuple[list[pd.DataFrame], list[dict], pd.DataFrame]:
    frames, errors, all_dropped = [], [], []

    xl = _open_workbook(path)
    if xl is None:
        return frames, [{"file": path.name, "sheet": "(open)", "reason": "Could not open with openpyxl or calamine"}], pd.DataFrame()

    hr_sheets = [s for s in xl.sheet_names if s.strip().lower().startswith(SHEET_PREFIX)]

    if not hr_sheets:
        try:
            raw         = pd.read_excel(path)
            df, dropped = _drop_empty_rows(_apply_generic_column_mapping(raw))
            if not dropped.empty:
                dropped["_source_file"]  = path.name
                dropped["_source_sheet"] = "(generic)"
                all_dropped.append(dropped)
            if len(df) > 0:
                frames.append(df)
        except Exception as exc:
            errors.append({"file": path.name, "sheet": "(first)", "reason": str(exc)})
        return frames, errors, pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame()

    for sheet in hr_sheets:
        try:
            raw = _read_sheet(path, sheet)
            df  = _apply_template_column_mapping(raw)
            if VP_SHEET_MARKER in sheet:
                df["is_vp_office"] = True
            df, dropped = _drop_empty_rows(df)
            if not dropped.empty:
                dropped["_source_file"]  = path.name
                dropped["_source_sheet"] = sheet
                all_dropped.append(dropped)
            if len(df) > 0:
                frames.append(df)
        except Exception as exc:
            errors.append({"file": path.name, "sheet": sheet, "reason": str(exc)})

    return frames, errors, pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame()


def _open_workbook(path: Path):
    try:
        return pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        try:
            return pd.ExcelFile(path, engine="calamine")
        except Exception:
            return None


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet, header=HEADER_ROW, engine="openpyxl")
    except Exception:
        return pd.read_excel(path, sheet_name=sheet, header=HEADER_ROW, engine="calamine")


def _apply_template_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in TEMPLATE_RENAME.items() if k in df.columns})
    return df.drop(columns=[c for c in PRIVACY_DROP if c in df.columns], errors="ignore")


def _apply_generic_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for standard_name, variants in GENERIC_VARIANTS.items():
        for col in df.columns:
            if col in variants:
                rename[col] = standard_name
                break
    return df.rename(columns=rename)


def _drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = [c for c in ("sector", "institute", "career_path") if c in df.columns]

    def _is_empty(val):
        return pd.isna(val) or str(val).strip() in ("", "nan")

    all_empty_mask = df[key_cols].apply(lambda row: all(_is_empty(v) for v in row), axis=1)

    dropped = df[all_empty_mask].copy()
    dropped["_drop_reason"] = "all key columns empty"
    df = df[~all_empty_mask].reset_index(drop=True)

    return df, dropped


# Stage 2 — Normalize

def normalize_hr_data(df: pd.DataFrame) -> pd.DataFrame:
    start = len(df)
    print(f"\n{'='*55}")
    print(f"  PIPELINE ROW TRACKER")
    print(f"{'='*55}")
    print(f"  {'After loading Excel':<35} {start:>5} rows")

    df = _ensure_required_columns(df)

    before = len(df)
    df = normalize_sectors(df)
    print(f"  {'After sector normalisation':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = normalize_institutes(df)
    print(f"  {'After institute normalisation':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = normalize_career_paths(df)
    print(f"  {'After career path normalisation':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = normalize_education(df)
    print(f"  {'After education normalisation':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = normalize_focus_areas(df)
    print(f"  {'After focus areas normalisation':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = parse_employment_status(df)
    print(f"  {'After status parsing':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    before = len(df)
    df = detect_vp_office(df)
    print(f"  {'After VP office detection':<35} {len(df):>5} rows  (dropped {before - len(df)})")

    print(f"{'─'*55}")
    print(f"  {'TOTAL DROPPED':<35} {start - len(df):>5} rows")
    print(f"  {'FINAL COUNT':<35} {len(df):>5} rows")
    print(f"{'='*55}\n")

    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if "area" in col else 0.0
    return df


def normalize_sectors(df: pd.DataFrame) -> pd.DataFrame:
    df["sector"] = df["sector"].astype(str).str.strip().apply(_translate_sector)

    unmapped_mask = ~df["sector"].isin(SECTORS)
    if unmapped_mask.any():
        def _recover_sector(row):
            recovered = INSTITUTE_TO_SECTOR.get(row["sector"])
            if recovered:
                return recovered
            recovered = INSTITUTE_TO_SECTOR.get(str(row.get("institute", "")).strip())
            return recovered if recovered else row["sector"]

        df.loc[unmapped_mask, "sector"] = df[unmapped_mask].apply(_recover_sector, axis=1)

        still_unmapped = ~df["sector"].isin(SECTORS)
        if still_unmapped.any():
            bad = df.loc[still_unmapped, "sector"].unique()
            print(f"\n  [!] {still_unmapped.sum()} rows have unrecognised sector values (kept):")
            for val in bad:
                count = (df.loc[still_unmapped, "sector"] == val).sum()
                print(f"      '{val}'  ({count} rows)")
            log.warning("  %d rows have unrecognised sector values (kept): %s",
                        still_unmapped.sum(), list(bad))

    return df


def normalize_institutes(df: pd.DataFrame) -> pd.DataFrame:
    df["institute"] = df["institute"].astype(str).str.strip().apply(_translate_institute)
    return df


def normalize_career_paths(df: pd.DataFrame) -> pd.DataFrame:
    df["role"]         = df["career_path"].fillna("").astype(str).str.strip()
    df["career_path"]  = df["role"].apply(lambda raw: _map_value(raw, CAREER_PATH_MAP))
    df["is_executive"] = df["role"].apply(
        lambda raw: raw in EXECUTIVE_ROLES or raw.lower() in EXECUTIVE_ROLES
    )
    for flag_column in ("General Manager", "Sector VP"):
        if flag_column in df.columns:
            flagged = df[flag_column].astype(str).str.strip().str.lower() == "yes"
            df["is_executive"] = df["is_executive"] | flagged
    return df


def normalize_education(df: pd.DataFrame) -> pd.DataFrame:
    df["degree"] = df["degree"].fillna("").astype(str).str.strip().apply(lambda raw: _map_value(raw, DEGREE_MAP))
    df["gender"] = df["gender"].fillna("").astype(str).str.strip().apply(lambda raw: _map_value(raw, GENDER_MAP))
    return df


def normalize_focus_areas(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("focus_1_pct", "focus_2_pct", "focus_3_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).apply(normalize_percentage_value)

    _FOCUS_JUNK = {"مبتعث", "focus areas"}
    for col in ("focus_1_area", "focus_2_area", "focus_3_area"):
        df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", "")
        df[col] = df[col].apply(lambda x: "" if x in _FOCUS_JUNK else x)

    df = _normalise_focus_pcts(df)

    admin_mask = df["career_path"] == "Administrative"
    for col in ("focus_1_area", "focus_2_area", "focus_3_area"):
        df.loc[admin_mask, col] = ""
    for col in ("focus_1_pct", "focus_2_pct", "focus_3_pct"):
        df.loc[admin_mask, col] = 0.0

    return df


def parse_employment_status(df: pd.DataFrame) -> pd.DataFrame:
    if "research_interest" not in df.columns:
        df["status"] = "Active"
        return df

    def parse(raw) -> str:
        val = str(raw).strip() if not isinstance(raw, str) else raw.strip()
        if val in ("", "nan"):
            return "Active"
        return STATUS_MAP.get(val) or STATUS_MAP.get(val.lower()) or "Active"

    df["status"] = df["research_interest"].fillna("").apply(parse)
    return df


def detect_vp_office(df: pd.DataFrame) -> pd.DataFrame:
    if "is_vp_office" not in df.columns:
        df["is_vp_office"] = False
    else:
        df["is_vp_office"] = df["is_vp_office"].eq(True)

    df["is_vp_office"] = df["is_vp_office"] | df["institute"].isin(VP_OFFICE_NAMES)
    df.loc[df["is_vp_office"], "institute"] = "VP Office"
    return df


# Stage 3 — Alignment (per-employee)

def classify_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df["alignment"]        = df.apply(classify_institute_alignment, axis=1)
    df["sector_alignment"] = df.apply(classify_sector_alignment,    axis=1)
    return df


def classify_institute_alignment(row) -> str:
    if row["career_path"] != "Research":
        return "N/A"

    institute = str(row["institute"])
    domains   = INSTITUTE_DOMAINS.get(institute, SECTOR_FALLBACK.get(str(row["sector"]), []))
    if not domains:
        return "Unclassified"

    f1, f2, f3 = str(row["focus_1_area"]), str(row["focus_2_area"]), str(row["focus_3_area"])
    p1, p2, p3 = float(row["focus_1_pct"]), float(row["focus_2_pct"]), float(row["focus_3_pct"])

    return _weighted_alignment(f1, f2, f3, p1, p2, p3, match_fn=lambda area: _overlaps_domains(area, domains))


def classify_sector_alignment(row) -> str:
    if row["career_path"] != "Research":
        return "N/A"

    sector_areas = SECTOR_FOCUS_LOWER.get(str(row["sector"]), set())
    if not sector_areas:
        return "Unclassified"

    f1, f2, f3 = str(row["focus_1_area"]), str(row["focus_2_area"]), str(row["focus_3_area"])
    p1, p2, p3 = float(row["focus_1_pct"]), float(row["focus_2_pct"]), float(row["focus_3_pct"])

    return _weighted_alignment(f1, f2, f3, p1, p2, p3, match_fn=lambda area: area in sector_areas)


# Shared helpers

def _weighted_alignment(f1, f2, f3, p1, p2, p3, match_fn) -> str:
    if not f1 and not f2 and not f3:
        return "Unclassified"

    total_weight = p1 + p2 + p3
    if total_weight == 0:
        return "Unclassified"

    matched_weight = 0.0
    if f1 and match_fn(f1):
        matched_weight += p1
    if f2 and match_fn(f2):
        matched_weight += p2
    if f3 and match_fn(f3):
        matched_weight += p3

    ratio = matched_weight / total_weight
    if ratio >= 0.55:
        return "Aligned"
    if ratio >= 0.20:
        return "Partial"
    return "Misaligned"


def _overlaps_domains(focus: str, domains: list[str]) -> bool:
    if not focus or focus == "nan":
        return False
    focus_words = {w for w in focus.split() if w not in STOP_WORDS and len(w) > 2}
    for domain in domains:
        domain_words = {w for w in domain.split() if w not in STOP_WORDS and len(w) > 2}
        if focus_words & domain_words or domain in focus or focus in domain:
            return True
    return False


def _translate_sector(raw) -> str:
    raw = str(raw).strip() if not isinstance(raw, str) else raw.strip()
    if raw in ("nan", ""):
        return raw
    mapped = SECTOR_MAP.get(raw) or SECTOR_MAP.get(raw.lower())
    if mapped:
        return mapped
    for sector in SECTORS:
        keywords = [w for w in sector.lower().split() if len(w) > 3]
        if any(kw in raw.lower() for kw in keywords):
            return sector
    return raw


def _translate_institute(raw: str) -> str:
    if raw in ("nan", ""):
        return raw
    return INSTITUTE_MAP.get(raw, raw)


def _map_value(raw: str, mapping: dict) -> str:
    return mapping.get(raw) or mapping.get(raw.lower(), raw)


def percentage(n: int | float, total: int | float) -> float:
    return round(n / total * 100, 1) if total else 0.0


def normalize_percentage_value(val: float) -> float:
    if 0 < val <= 1.0:
        return round(val * 100, 1)
    return max(0.0, min(100.0, val))


def _normalise_focus_pcts(df: pd.DataFrame) -> pd.DataFrame:
    pct_cols  = ["focus_1_pct", "focus_2_pct", "focus_3_pct"]
    area_cols = ["focus_1_area", "focus_2_area", "focus_3_area"]

    def _fix_row(row):
        total = sum(row[c] for c in pct_cols)
        if total == 0 or abs(total - 100.0) < 0.1:
            return row
        for p, a in zip(pct_cols, area_cols):
            if row[a] and row[a] not in ("nan", ""):
                row[p] = round(row[p] / total * 100, 1)
        return row

    return df.apply(_fix_row, axis=1)


# CLI entry point

def _report_dropped_rows(meta: dict, out_dir: Path | None = None):
    dropped = meta.get("dropped_rows", pd.DataFrame())
    lines   = ["", "=" * 60, "  DROPPED ROWS REPORT", "=" * 60]

    if dropped.empty:
        lines.append("  No rows dropped.")
    else:
        lines.append(f"  Total dropped: {len(dropped)}")
        _SHOW = ["_source_file", "_source_sheet", "sector", "institute",
                 "career_path", "degree", "gender", "focus_1_area", "focus_2_area", "focus_3_area"]

        for reason, group in dropped.groupby("_drop_reason"):
            lines.append("")
            lines.append(f"  Reason: {reason}  ({len(group)} rows)")
            lines.append("  " + "-" * 56)
            for _, row in group.iterrows():
                parts = []
                for col in _SHOW:
                    val = str(row.get(col, "")).strip()
                    if val and val.lower() not in ("nan", ""):
                        label = col.replace("_source_", "").replace("_", " ")
                        parts.append(f"{label}: {val}")
                lines.append("  • " + " | ".join(parts))

    lines.append("")
    out = (out_dir or DEFAULT_OUTPUT.parent) / "dropped_rows_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Dropped rows report -> %s", out)


def _report_focus_areas(df: pd.DataFrame, suffix: str = ""):
    from collections import Counter
    lines = []

    def _section(title: str, counter: Counter, known_set: set):
        unknown     = {v: c for v, c in counter.items() if v.lower() not in known_set}
        known_found = {v: c for v, c in counter.items() if v.lower() in known_set}
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}  ({len(counter)} unique, {sum(counter.values())} total)")
        lines.append("=" * 60)
        lines.append(f"  [OK] KNOWN ({len(known_found)}):")
        for v, c in sorted(known_found.items(), key=lambda x: -x[1]):
            lines.append(f"     {c:4d}  |  {v}")
        lines.append(f"  [!!] UNKNOWN ({len(unknown)}) -- needs translation:")
        for v, c in sorted(unknown.items(), key=lambda x: -x[1]):
            lines.append(f"     {c:4d}  |  {v}")

    def _count(cols):
        counter: Counter = Counter()
        for col in cols if isinstance(cols, list) else [cols]:
            if col in df.columns:
                for val in df[col].dropna():
                    v = str(val).strip()
                    if v and v.lower() not in ("nan", ""):
                        counter[v] += 1
        return counter

    known_focus = set()
    for domains in INSTITUTE_DOMAINS.values():
        known_focus.update(d.lower() for d in domains)
    for domains in SECTOR_FALLBACK.values():
        known_focus.update(d.lower() for d in domains)

    _section("FOCUS AREAS",  _count(["focus_1_area", "focus_2_area", "focus_3_area"]), known_focus)
    _section("SECTORS",      _count("sector"),      {v.lower() for v in SECTOR_MAP.values()} | {k.lower() for k in SECTOR_MAP})
    _section("INSTITUTES",   _count("institute"),   {v.lower() for v in INSTITUTE_MAP.values()} | {k.lower() for k in INSTITUTE_MAP})
    _section("CAREER PATHS", _count("career_path"), {v.lower() for v in CAREER_PATH_MAP.values()} | {k.lower() for k in CAREER_PATH_MAP})
    _section("DEGREES",      _count("degree"),      {v.lower() for v in DEGREE_MAP.values()} | {k.lower() for k in DEGREE_MAP})
    _section("GENDERS",      _count("gender"),      {v.lower() for v in GENDER_MAP.values()} | {k.lower() for k in GENDER_MAP})
    _section("STATUSES",     _count("status"),      {v.lower() for v in STATUS_MAP.values()} | {k.lower() for k in STATUS_MAP})

    lines.append("")
    lines.append("=" * 60)
    lines.append("  Fix [!!] UNKNOWN values in translations.json")
    lines.append("=" * 60)

    out = DEFAULT_OUTPUT.parent / f"translation_report{suffix}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Translation report saved to %s", out)


# Employee columns written to both output files
_STAFF_COLS = [
    "sector", "institute", "career_path", "degree", "gender", "status",
    "focus_1_area", "focus_1_pct", "focus_2_area", "focus_2_pct", "focus_3_area", "focus_3_pct",
    "research_interest", "specialization", "alignment", "sector_alignment",
]


def _save_output(df: pd.DataFrame, meta: dict, out_path: Path, csv_mode: bool = False):
    # Compute per-employee alignment fields
    df = classify_alignment(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in _STAFF_COLS if c in df.columns]

    # File 1: cleaned_hr_data.json — employee records
    records = json.loads(df[cols].to_json(orient="records", force_ascii=False))
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Written to %s  (%d employees)", out_path, len(records))

    # File 2: cleaned_hr_data.csv — same data as spreadsheet
    csv_path = out_path.with_suffix(".csv")
    df[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Written to %s  (%d employees)", csv_path, len(df))

    if meta.get("errors"):
        log.warning("%d loading error(s)", len(meta["errors"]))


def main():
    args  = _parse_args()
    files = _resolve_files(args)

    if not files:
        log.error("No Excel files found.")
        sys.exit(1)

    if args.per_file:
        _run_per_file(files, args)
        return

    df, meta = load_data(files)
    if args.stage == "load":
        log.info("Columns: %s", list(df.columns))
        log.info("Sample:\n%s", df.head(3).to_string())
        return

    if args.report_focus_areas:
        _report_focus_areas(df, suffix="_raw")

    n_dropped_load = len(meta.get("dropped_rows", pd.DataFrame()))
    print(f"\n  {'After empty row filter':<35} {len(df):>5} rows  (dropped {n_dropped_load})")
    _report_dropped_rows(meta)
    df = normalize_hr_data(df)
    if args.stage == "normalize":
        log.info("Sample after normalize:\n%s", df.head(3).to_string())
        return

    if args.report_focus_areas:
        _report_focus_areas(df, suffix="")
        return

    _save_output(df, meta, args.output)


def _run_per_file(files: list[Path], args):
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in [f for f in files if not f.name.startswith("~$")]:
        log.info("--- Processing %s ---", file.name)
        try:
            df, meta = load_data([file])
            if args.report_focus_areas:
                _report_focus_areas(df, suffix="_raw")
            _report_dropped_rows(meta, out_dir)
            df = normalize_hr_data(df.copy())
            if args.report_focus_areas:
                _report_focus_areas(df, suffix="")
                continue
            out_path = out_dir / (file.stem + ".json")
            _save_output(df, meta, out_path)
        except Exception as e:
            log.error("Failed to process %s: %s", file.name, e)


def _resolve_files(args) -> list[Path]:
    if args.files:
        return [Path(f) for f in args.files if not Path(f).name.startswith("~$")]
    files = sorted(args.input_dir.glob("*.xlsx")) + sorted(args.input_dir.glob("*.xls"))
    return [f for f in files if not f.name.startswith("~$")]


def _parse_args():
    parser = argparse.ArgumentParser(description="HR pipeline: Excel → JSON / CSV")
    parser.add_argument("files", nargs="*", metavar="FILE", help="Excel file paths (default: scripts/data/)")
    parser.add_argument("--input-dir",          type=Path, default=DEFAULT_INPUT,  metavar="DIR",
                        help=f"Directory containing Excel files (default: {DEFAULT_INPUT})")
    parser.add_argument("--output",             type=Path, default=DEFAULT_OUTPUT, metavar="FILE",
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--stage",              choices=["load", "normalize", "all"], default="all",
                        help="Stop after this stage for debugging (default: all)")
    parser.add_argument("--per-file",           action="store_true",
                        help="Process each file separately and save one JSON+CSV per file")
    parser.add_argument("--report-focus-areas", action="store_true",
                        help="Save a translation audit report")
    return parser.parse_args()


if __name__ == "__main__":
    main()
