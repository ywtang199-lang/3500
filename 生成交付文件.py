from __future__ import annotations

import csv
import gzip
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "核心词库_母数据.json"
SOURCE_DIR = Path(os.environ.get("TEMP", str(ROOT))) / "fr_core3500_sources"
TARGET_TOTAL = 3500
TARGET_LEMMAS = 3200
TARGET_EXPRESSIONS_MAX = 300
PLAN_COUNTS = [234] * 5 + [233] * 10
SOURCE_ALIASES = {"sœur": "soeur", "œuf": "oeuf", "œil": "oeil", "cœur": "coeur"}
FLELEX_FORM_ALIASES = {"une": "un", "les": "le", "ils": "il"}


def pos_code(pos: str) -> str:
    if pos.startswith("n."):
        return "NOM"
    if pos == "v.":
        return "VER"
    if pos == "adj.":
        return "ADJ"
    if pos == "adv.":
        return "ADV"
    if pos == "prep.":
        return "PRE"
    if pos.startswith("art.") or pos.startswith("dét."):
        return "ART"
    if pos.startswith("conj."):
        return "CON"
    if pos.startswith("pron.") or pos.startswith("dét./pron."):
        return "PRO"
    return ""


def parse_lexique() -> dict[str, list[dict]]:
    archive = SOURCE_DIR / "Lexique4.1.zip"
    if not archive.exists():
        return {}
    result: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zf:
        with zf.open("Lexique4.1/Lexique4.1.tsv") as raw:
            reader = csv.DictReader((line.decode("utf-8-sig") for line in raw), delimiter="\t")
            for row in reader:
                lemma = row.get("4_Lemme", "").strip().casefold()
                if lemma:
                    result[lemma].append(row)
    return result


def parse_flelex() -> dict[str, list[dict]]:
    path = SOURCE_DIR / "FleLex_TT_Beacco.tsv"
    if not path.exists():
        return {}
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            result[row["word"].strip().casefold()].append(row)
    return result


def parse_lexik() -> dict[str, list[dict]]:
    path = SOURCE_DIR / "lexik_formes_mots.tsv.gz"
    if not path.exists():
        return {}
    result: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            lemma = row.get("lemme", "").strip().casefold()
            if lemma:
                result[lemma].append(row)
    return result


def parse_lexik_lemmas() -> dict[str, dict]:
    path = SOURCE_DIR / "lexik_lemmes.tsv.gz"
    if not path.exists():
        return {}
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = row.get("id_lemme", "").strip()
            if key:
                result[key] = row
    return result


def normalize_ipa(s: str) -> str:
    # LeXiK annotates contextual variants in square brackets and alternatives in
    # braces (e.g. {ɛ|e}). Use the first isolated-form alternative for comparison.
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\{([^{}|]+)(?:\|[^{}]*)?\}", r"\1", s)
    s = re.sub(r"[/.·\-‿\s]", "", s.strip().casefold())
    return s.replace("g", "ɡ")


def clean_lexik_phone(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", s.strip())
    s = re.sub(r"\{([^{}|]+)(?:\|[^{}]*)?\}", r"\1", s)
    return s.replace("‿", "")


def lexical_choice(rows: list[dict], entry: dict) -> dict | None:
    if not rows:
        return None
    target = pos_code(entry["pos"])

    def rank(row):
        exact_form = row.get("1_Mot", "").strip().casefold() == entry["fr"].casefold()
        pos_match = bool(target and row.get("5_Cgram", "").strip().upper() == target)
        lemma_form = row.get("14_IsLem", "").strip() in {"1", "oui", "TRUE"}
        try:
            freq = float(row.get("12_FreqLemme", "0") or 0)
        except ValueError:
            freq = 0
        return (exact_form, pos_match, lemma_form, freq)

    return max(rows, key=rank)


def enrich(entries: list[dict]) -> tuple[list[dict], dict]:
    lexique = parse_lexique()
    flelex = parse_flelex()
    lexik = parse_lexik()
    lexik_lemmas = parse_lexik_lemmas()
    counts = Counter()
    for number, entry in enumerate(entries, 1):
        entry["id"] = number
        entry["day"] = 1
        lemma = entry["fr"].casefold()
        source_form = lemma if lemma in lexique else SOURCE_ALIASES.get(lemma, lemma)
        candidates = lexique.get(source_form, [])
        lexrow = lexical_choice(candidates, entry)
        if lexrow:
            counts["lexique_lemma_found"] += 1
            if source_form != lemma:
                entry["lexique_alias"] = source_form
                counts["lexique_alias_used"] += 1
            entry["lexique_frequency"] = lexrow.get("12_FreqLemme", "")
            entry["lexique_gender"] = lexrow.get("7_Genre", "")
            lexique_ipa = lexrow.get("3_Phono_IPA", "").strip()
            entry["lexique_ipa"] = lexique_ipa
            entry["source_status"] = "Lexique 4.1 词元匹配"
        else:
            lexique_ipa = ""
            entry["lexique_ipa"] = ""
            counts["lexique_lemma_missing"] += 1

        fle_source_form = lemma if lemma in flelex else FLELEX_FORM_ALIASES.get(lemma, SOURCE_ALIASES.get(lemma, lemma))
        fle_rows = flelex.get(fle_source_form, [])
        fle_target = pos_code(entry["pos"])
        def fle_pos_matches(tag: str, target: str) -> bool:
            tag = tag.upper()
            if target == "ART":
                return tag == "ART" or tag.startswith("DET:ART")
            if target == "PRE":
                return tag.startswith("PRE") or tag == "PRP"
            if target == "CON":
                return tag.startswith("CONJ") or tag == "CON"
            return tag == target or tag.startswith(target + ":")

        fle_candidates = [r for r in fle_rows if fle_pos_matches(r.get("tag", ""), fle_target)]
        fle = fle_candidates[0] if fle_candidates else None
        entry.pop("flelex_alias", None)
        if fle:
            entry["flelex_level"] = fle.get("level", "")
            if fle_source_form != lemma:
                entry["flelex_alias"] = fle_source_form
            counts["flelex_found"] += 1
        else:
            entry["flelex_level"] = ""
            counts["flelex_missing"] += 1

        lx_rows = lexik.get(lemma, [])
        # LeXiK stores inflected forms. Prefer the entry whose spelling is the lemma,
        # then the matching grammatical category; do not replace Lexique IPA silently.
        target = {"NOM": "NOUN", "VER": "VERB", "ADJ": "ADJ", "ADV": "ADV",
                  "PRE": "ADP", "ART": "DET", "CON": "CCONJ", "PRO": "PRON"}.get(pos_code(entry["pos"]), "")
        lx_candidates = [r for r in lx_rows if r.get("ortho", "").strip().casefold() == lemma]
        if not lx_candidates:
            lx_candidates = lx_rows
        wanted_gender = {"n.m.": "m", "n.f.": "f", "n.f./m.": "f"}.get(entry["pos"])

        def has_wanted_gender(row):
            if not wanted_gender:
                return False
            record = lexik_lemmas.get(row.get("id_lemme", ""), {})
            match = re.search(r"(?:^|;)genre:([^;]+)", record.get("proprietes", ""))
            return bool(match and wanted_gender in match.group(1).split(","))

        lx_candidates.sort(key=lambda r: (
            r.get("upos", "").upper() == target if target else False,
            has_wanted_gender(r),
            float(r.get("freq_opensubtitles") or 0) if (r.get("freq_opensubtitles") or "").replace(".", "", 1).isdigit() else 0,
        ), reverse=True)
        lx_ipa = lx_candidates[0].get("phone", "").strip() if lx_candidates else ""
        if lx_ipa:
            counts["lexik_lemma_found"] += 1
            clean_phone = clean_lexik_phone(lx_ipa)
            confidence = lx_candidates[0].get("confiance", "")
            entry["lexik_confidence"] = confidence
            lemma_id = lx_candidates[0].get("id_lemme", "")
            lemma_record = lexik_lemmas.get(lemma_id, {})
            props = lemma_record.get("proprietes", "")
            gender_match = re.search(r"(?:^|;)genre:([^;]+)", props)
            entry["lexik_gender"] = gender_match.group(1) if gender_match else ""
            if entry["pos"] == "n.f./m.":
                genders = set()
                for candidate in lx_candidates:
                    if candidate.get("upos") != "NOUN":
                        continue
                    if normalize_ipa(candidate.get("phone", "")) != normalize_ipa(lx_ipa):
                        continue
                    record = lexik_lemmas.get(candidate.get("id_lemme", ""), {})
                    match = re.search(r"(?:^|;)genre:([^;]+)", record.get("proprietes", ""))
                    if match:
                        genders.update(match.group(1).split(","))
                if {"f", "m"}.issubset(genders):
                    entry["lexik_gender"] = "f,m"
            # LeXiK's G is model-predicted. Prefer a source-provided Lexique IPA
            # in that case; otherwise retain LeXiK's cleaned, isolated-form IPA.
            if confidence == "G" and lexique_ipa:
                primary_ipa = lexique_ipa
                entry["ipa_source_primary"] = "Lexique 4.1（LeXiK为G预测）"
                counts["lexique_ipa_fallback"] += 1
            else:
                primary_ipa = clean_phone
                entry["ipa_source_primary"] = f"LeXiK 1.0（{confidence or '置信度未标'}）"
            entry["ipa"] = f"/{primary_ipa.strip('/')} /".replace(" /", "/")
            entry["source_status"] = "LeXiK 词形/词性核对；Lexique 4.1 词频核对"
            if not lexique_ipa:
                entry["ipa_check"] = "仅LeXiK可核；Lexique该词元无匹配"
                counts["ipa_lexique_missing"] += 1
            elif normalize_ipa(clean_phone) == normalize_ipa(lexique_ipa):
                entry["ipa_check"] = "两源 IPA 一致（忽略音节分隔、连读注记）"
                counts["ipa_crossmatch"] += 1
            else:
                entry["ipa_check"] = f"两源转写有差异；LeXiK={clean_phone}；Lexique={lexique_ipa}"
                counts["ipa_source_difference"] += 1
        else:
            if lexique_ipa:
                entry["ipa"] = f"/{lexique_ipa.strip('/')} /".replace(" /", "/")
                entry["ipa_source_primary"] = "Lexique 4.1（LeXiK未匹配）"
                entry["ipa_check"] = "LeXiK 未匹配；采用 Lexique IPA"
                entry["source_status"] = "Lexique 4.1 词元/IPA匹配；LeXiK未匹配"
                counts["lexique_ipa_fallback"] += 1
                counts["lexik_missing"] += 1
            else:
                entry["ipa_check"] = "LeXiK 与 Lexique 均未匹配"
                counts["lexik_missing"] += 1

        if not entry.get("ipa"):
            counts["ipa_missing"] += 1
        if entry["pos"].startswith("n."):
            declared = entry["pos"][2:].replace(".", "").split("/")
            found_gender = entry.get("lexik_gender", "").strip().casefold().split(",")
            if found_gender and not any(g in found_gender for g in declared):
                counts["noun_gender_difference"] += 1

    report = {
        "source_dir_available": bool(lexique or flelex or lexik),
        "lexique_rows": sum(len(v) for v in lexique.values()),
        "lexique_lemmas_indexed": len(lexique),
        "flelex_rows": sum(len(v) for v in flelex.values()),
        "flelex_words_indexed": len(flelex),
        "lexik_form_rows_indexed_by_lemma": sum(len(v) for v in lexik.values()),
        "lexik_lemmas": len(lexik),
        "lexik_lemma_properties": len(lexik_lemmas),
        "counts": dict(counts),
    }
    return entries, report


def french_sort_key(word: str) -> tuple[str, str]:
    folded = word.casefold().replace("œ", "oe").replace("æ", "ae")
    plain = "".join(c for c in unicodedata.normalize("NFD", folded) if not unicodedata.combining(c))
    return plain, folded


def day_boundaries() -> list[int]:
    boundaries = []
    total = 0
    for count in PLAN_COUNTS:
        total += count
        boundaries.append(total)
    return boundaries


def day_for_id(entry_id: int) -> int:
    edge = 0
    for d, count in enumerate(PLAN_COUNTS, 1):
        edge += count
        if entry_id <= edge:
            return d
    return 15


def ipa_body(entry: dict) -> str:
    return entry.get("ipa", "")


def edition_rows(entries: list[dict], mode: str) -> list[list[str]]:
    result = []
    for e in entries:
        if mode == "study":
            result.append([f"{e['id']:03d}", e["fr"], e["ipa"], e["pos"], e["zh"], e.get("collocation", "")])
        elif mode == "fr_to_zh":
            result.append([f"{e['id']:03d}", e["fr"], e["ipa"], "________________"])
        elif mode == "zh_to_fr":
            result.append([f"{e['id']:03d}", e["zh"], "________________", "________________" if e.get("collocation") else ""])
    return result


def make_markdown(entries: list[dict], mode: str, target: Path, day: int):
    first_id = entries[0]["id"]
    last_id = entries[-1]["id"]
    if mode == "study":
        headers = ["编号", "法语", "IPA", "词性", "中文", "必要搭配"]
        title = f"法语核心3500｜中法学习版｜Day {day}"
    elif mode == "fr_to_zh":
        headers = ["编号", "法语", "IPA", "中文填写"]
        title = f"法语核心3500｜法填中练习版｜Day {day}"
    else:
        headers = ["编号", "中文提示", "法语填写", "必要介词搭配填写"]
        title = f"法语核心3500｜中填法练习版｜Day {day}"
    lines = [f"# {title}", "", f"本日已整理 {len(entries)} 条；编号 {first_id:03d}–{last_id:03d}。", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for e in entries:
        if mode == "study":
            fr = f"**{e['fr']}**" if e["important"] else e["fr"]
            cells = [f"{e['id']:03d}", fr, e["ipa"], e["pos"], e["zh"], e.get("collocation", "")]
        elif mode == "fr_to_zh":
            fr = f"**{e['fr']}**" if e["important"] else e["fr"]
            cells = [f"{e['id']:03d}", fr, e["ipa"], "________________"]
        else:
            cells = [f"{e['id']:03d}", e["zh"], "________________", "________________" if e.get("collocation") else ""]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_workbook(entries: list[dict], lexical_report: dict):
    from openpyxl import Workbook, load_workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    output = ROOT / "法语核心3500_15天打卡.xlsx"
    checkins = {}
    if output.exists():
        previous = load_workbook(output, read_only=True)
        if "打卡计划" in previous:
            for row in previous["打卡计划"].iter_rows(min_row=2, max_col=10, values_only=True):
                note = row[9] if row[9] not in ("已完成", "待续") else None
                checkins[row[0]] = [*row[5:9], note]
        previous.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "总览"
    navy = "17365D"
    blue = "D9EAF7"
    gold = "FCE4B2"
    green = "DDEBF7"
    white = "FFFFFF"
    thin = Side(style="thin", color="D9E1F2")

    ws.merge_cells("A1:F1")
    ws["A1"] = "法语核心3500｜15天打卡｜词库进度"
    ws["A1"].font = Font(name="Microsoft YaHei", size=16, bold=True, color=white)
    ws["A1"].fill = PatternFill("solid", fgColor=navy)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32
    summary = [
        ("目标总量", TARGET_TOTAL), ("目标独立词元", TARGET_LEMMAS),
        ("必要固定表达上限", TARGET_EXPRESSIONS_MAX), ("已制作", len(entries)),
        ("尚待补充", TARGET_TOTAL - len(entries)),
        ("完成率", len(entries) / TARGET_TOTAL),
        ("已制作天数", sum(bool([e for e in entries if e["day"] == d]) for d in range(1, 16))),
        ("数据状态", f"已整理 {len(entries)} 条；其余编号为待补空位"),
    ]
    for idx, (key, val) in enumerate(summary, 3):
        ws.cell(idx, 1, key).font = Font(name="Microsoft YaHei", bold=True, color=navy)
        ws.cell(idx, 2, val).font = Font(name="Microsoft YaHei", size=11)
        ws.cell(idx, 1).fill = PatternFill("solid", fgColor=blue)
        if key == "完成率":
            ws.cell(idx, 2).number_format = "0.0%"
    ws["D3"] = "实际进度"
    ws["D3"].font = Font(name="Microsoft YaHei", bold=True, color=navy)
    ws["D4"] = f"{len(entries)} / {TARGET_TOTAL}"
    ws["D4"].font = Font(name="Microsoft YaHei", size=20, bold=True, color="2F75B5")
    ws.merge_cells("D5:F5")
    ws["D5"] = "按编号显示每日进度；空白编号待补"
    ws["D5"].font = Font(name="Microsoft YaHei", size=11)

    sched = wb.create_sheet("打卡计划")
    headers = ["天数", "计划编号", "目标条数", "已制作", "完成率", "打卡日期", "实际完成", "已会", "待复习", "备注"]
    sched.append(headers)
    start = 1
    for day, count in enumerate(PLAN_COUNTS, 1):
        made = max(0, min(start + count - 1, len(entries)) - start + 1)
        label = f"Day {day}"
        date_checked, completed, known, review, note = checkins.get(label, [None] * 5)
        sched.append([label, f"{start:04d}–{start + count - 1:04d}", count, made,
                      made / count, date_checked, completed, known, review,
                      note if note is not None else ("已完成" if made == count else "待续")])
        start += count
    for c in sched[1]:
        c.font = Font(name="Microsoft YaHei", bold=True, color=white)
        c.fill = PatternFill("solid", fgColor=navy)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sched.iter_rows(min_row=2):
        row[4].number_format = "0%"
        for c in row:
            c.font = Font(name="Microsoft YaHei", size=10)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = Border(bottom=thin)
    for col, width in enumerate([12, 18, 12, 12, 12, 14, 12, 10, 10, 25], 1):
        sched.column_dimensions[get_column_letter(col)].width = width
    sched.freeze_panes = "A2"
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "15 天计划条数与已制作条数"
    chart.y_axis.title = "词条数"
    chart.x_axis.title = "打卡日"
    chart.height = 8
    chart.width = 18
    chart.grouping = "clustered"
    chart.add_data(Reference(sched, min_col=3, max_col=4, min_row=1, max_row=16), titles_from_data=True)
    chart.set_categories(Reference(sched, min_col=1, min_row=2, max_row=16))
    ws.add_chart(chart, "A13")

    master = wb.create_sheet("词库母表")
    master_headers = ["编号", "天数", "单元类型", "法语原形", "IPA", "词性", "中文义项", "必要搭配", "Lexique词频", "FLELex级别（核对用）", "LeXiK名词性别", "IPA核对", "IPA主来源/置信度", "状态"]
    master.append(master_headers)
    by_id = {e["id"]: e for e in entries}
    for entry_id in range(1, TARGET_TOTAL + 1):
        day = day_for_id(entry_id)
        e = by_id.get(entry_id)
        if e:
            master.append([entry_id, day, e["unit_type"], e["fr"], e["ipa"], e["pos"], e["zh"], e.get("collocation", ""), e.get("lexique_frequency", ""), e.get("flelex_level", ""), e.get("lexik_gender", ""), e.get("ipa_check", ""), f"{e.get('ipa_source_primary','')} / {e.get('lexik_confidence','')}", "已整理"])
        else:
            master.append([entry_id, day, *([""] * 11), "待补充"])
    for c in master[1]:
        c.font = Font(name="Microsoft YaHei", bold=True, color=white)
        c.fill = PatternFill("solid", fgColor=navy)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    master.freeze_panes = "D2"
    master.auto_filter.ref = f"A1:N{TARGET_TOTAL + 1}"
    for row in master.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Microsoft YaHei", size=9)
            c.alignment = Alignment(vertical="top", wrap_text=True)
            c.border = Border(bottom=thin)
        if row[13].value == "待补充":
            row[13].fill = PatternFill("solid", fgColor="F2F2F2")
    widths = [10, 9, 13, 24, 22, 14, 25, 28, 16, 18, 18, 43, 28, 12]
    for col, width in enumerate(widths, 1):
        master.column_dimensions[get_column_letter(col)].width = width
    tbl = Table(displayName="Core3500", ref=f"A1:N{TARGET_TOTAL + 1}")
    tbl.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
    master.add_table(tbl)

    edition_specs = [
        ("中法学习版", ["编号", "法语", "IPA", "词性", "中文", "必要搭配"], "study"),
        ("法填中练习", ["编号", "法语", "IPA", "中文填写"], "fr_to_zh"),
        ("中填法练习", ["编号", "中文提示", "法语填写", "必要介词搭配填写"], "zh_to_fr"),
    ]
    for sheet_name, headers, mode in edition_specs:
        sh = wb.create_sheet(sheet_name)
        sh.append(headers)
        for e in entries:
            row = edition_rows([e], mode)[0]
            sh.append(row)
        for c in sh[1]:
            c.font = Font(name="Microsoft YaHei", bold=True, color=white)
            c.fill = PatternFill("solid", fgColor=navy)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sh.freeze_panes = "A2"
        sh.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(entries)+1}"
        for row_idx, e in enumerate(entries, 2):
            for c in sh[row_idx]:
                c.font = Font(name="Microsoft YaHei", size=10, bold=bool(e["important"] and c.column == 2 and mode != "zh_to_fr"))
                c.alignment = Alignment(vertical="top", wrap_text=True)
                c.border = Border(bottom=thin)
        col_widths = {
            "中法学习版": [9, 24, 20, 12, 28, 25],
            "法填中练习": [9, 24, 20, 22],
            "中填法练习": [9, 28, 24, 28],
        }[sheet_name]
        for col, width in enumerate(col_widths, 1):
            sh.column_dimensions[get_column_letter(col)].width = width

    audit = wb.create_sheet("来源与质检")
    audit_rows = [
        ["检查项", "结果"],
        ["实有条目", len(entries)],
        ["编号连续", f"001–{len(entries):03d}；未发现缺号"],
        ["重复法语词元", sum(n - 1 for n in Counter(e["fr"].casefold() for e in entries).values() if n > 1)],
        ["缺失IPA", sum(not e.get("ipa") for e in entries)],
        ["Lexique词元匹配", lexical_report.get("counts", {}).get("lexique_lemma_found", 0)],
        ["Lexique IPA与LeXiK一致", lexical_report.get("counts", {}).get("ipa_crossmatch", 0)],
        ["IPA来源差异", lexical_report.get("counts", {}).get("ipa_source_difference", 0)],
        ["FLELex匹配", lexical_report.get("counts", {}).get("flelex_found", 0)],
        ["FLELex未匹配", lexical_report.get("counts", {}).get("flelex_missing", 0)],
        ["LeXiK未匹配", lexical_report.get("counts", {}).get("lexik_missing", 0)],
        ["IPA置信度分布", json.dumps(dict(Counter(e.get("lexik_confidence", "未标") or "未标" for e in entries)), ensure_ascii=False)],
        ["必要搭配", f"{sum(bool(e.get('collocation')) for e in entries)} 条。抽查来源见《方法与来源》。"],
        ["词性分布", json.dumps(dict(Counter(e["pos"] for e in entries)), ensure_ascii=False)],
        ["独立固定表达", sum(e["unit_type"] == "固定表达" for e in entries)],
        ["数据结构", "母数据 JSON 按 ID 储存；后续补条目后重跑本脚本，可重建三版、母表、打卡页、PDF。"],
    ]
    for row in audit_rows:
        audit.append(row)
    for c in audit[1]:
        c.font = Font(name="Microsoft YaHei", bold=True, color=white)
        c.fill = PatternFill("solid", fgColor=navy)
    audit.column_dimensions["A"].width = 30
    audit.column_dimensions["B"].width = 105
    for row in audit.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Microsoft YaHei", size=10)
            c.alignment = Alignment(vertical="top", wrap_text=True)
            c.border = Border(bottom=thin)
    wb.save(output)


def build_pdf(entries: list[dict], day: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    cnfont = FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties(family="DejaVu Sans")
    ipafont = FontProperties(family="DejaVu Sans")
    output = ROOT / f"法语核心3500_Day{day:02d}_打印版.pdf"
    specs = [
        ("中法学习版", "study"), ("法填中练习版", "fr_to_zh"), ("中填法练习版", "zh_to_fr")
    ]
    rows_per_page = 18
    with PdfPages(output) as pdf:
        for section, mode in specs:
            rows = edition_rows(entries, mode)
            headers = {
                "study": ["编号", "法语", "IPA", "词性", "中文", "必要搭配"],
                "fr_to_zh": ["编号", "法语", "IPA", "中文填写"],
                "zh_to_fr": ["编号", "中文提示", "法语填写", "必要介词搭配填写"],
            }[mode]
            for start in range(0, len(rows), rows_per_page):
                chunk = rows[start:start + rows_per_page]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                ax.text(0.02, 0.965, f"法语核心3500｜{section}｜Day {day}", fontproperties=cnfont, fontsize=15, weight="bold", color="#17365D", va="top")
                ax.text(0.02, 0.925, f"编号 {int(chunk[0][0]):03d}–{int(chunk[-1][0]):03d}    本日共 {len(entries)} 条", fontproperties=cnfont, fontsize=9, color="#44546A", va="top")
                col_widths = {"study": [0.06, 0.17, 0.18, 0.08, 0.28, 0.20], "fr_to_zh": [0.08, 0.28, 0.28, 0.36], "zh_to_fr": [0.07, 0.30, 0.30, 0.30]}[mode]
                table = ax.table(cellText=chunk, colLabels=headers, colWidths=col_widths, cellLoc="left", loc="center", bbox=[0.02, 0.085, 0.96, 0.79])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                for (r, c), cell in table.get_celld().items():
                    cell.set_edgecolor("#B4C7E7")
                    cell.set_linewidth(0.45)
                    if r == 0:
                        cell_font = cnfont
                    elif mode == "study":
                        cell_font = cnfont if c == 4 else ipafont
                    elif mode == "fr_to_zh":
                        cell_font = ipafont
                    else:
                        cell_font = cnfont if c == 1 else ipafont
                    cell.get_text().set_fontproperties(cell_font)
                    cell.get_text().set_wrap(True)
                    cell.get_text().set_color("#172B4D")
                    if r > 0 and mode in {"study", "fr_to_zh"} and c == 1 and entries[start + r - 1].get("important"):
                        cell.get_text().set_weight("bold")
                    if r == 0:
                        cell.set_facecolor("#17365D")
                        cell.get_text().set_color("#FFFFFF")
                        cell.get_text().set_weight("bold")
                    elif r % 2 == 0:
                        cell.set_facecolor("#F3F7FB")
                ax.text(0.98, 0.035, f"{section} · 第 {start // rows_per_page + 1} 页", fontproperties=cnfont, fontsize=8, ha="right", color="#667085")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def build_html(entries: list[dict]):
    plan = []
    start = 1
    actual = len(entries)
    for day, count in enumerate(PLAN_COUNTS, 1):
        made = max(0, min(start + count - 1, actual) - start + 1)
        plan.append((day, count, made, start, start + count - 1))
        start += count
    bar_svg = []
    for day, target, made, _, _ in plan:
        x = 48 + (day - 1) * 49
        h_target = target * 0.52
        h_made = made * 0.52
        y_target = 186 - h_target
        y_made = 186 - h_made
        bar_svg.append(f'<rect x="{x}" y="{y_target:.1f}" width="16" height="{h_target:.1f}" rx="4" fill="#D9EAF7"/><rect x="{x+18}" y="{y_made:.1f}" width="16" height="{h_made:.1f}" rx="4" fill="#2563EB"/><text x="{x+17}" y="206" text-anchor="middle">{day}</text>')
    percent = actual / TARGET_TOTAL * 100
    finished_days = sum(count == made for _, count, made, _, _ in plan)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>法语核心3500｜进度</title>
<style>body{{font:16px/1.55 "Microsoft YaHei",system-ui,sans-serif;background:#f6f8fb;color:#172b4d;margin:0;padding:32px}}main{{max-width:980px;margin:auto}}.card{{background:white;border:1px solid #e3eaf2;border-radius:16px;padding:24px;margin:16px 0;box-shadow:0 4px 16px #14365b0a}}h1{{margin:0 0 8px;font-size:28px}}.muted{{color:#5f6b7a}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:#f3f7fb;border-radius:12px;padding:15px}}.metric strong{{display:block;font-size:23px;color:#17365d}}.progress{{height:14px;border-radius:10px;background:#e8eef5;overflow:hidden;margin:12px 0}}.progress i{{display:block;width:{percent:.3f}%;height:100%;background:#2563eb}}svg{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e5ebf2;text-align:left}}th{{color:#52657a;background:#f6f9fc}}.tag{{display:inline-block;background:#eaf2ff;border-radius:12px;padding:2px 10px;color:#1d4ed8;font-size:13px}}@media(max-width:680px){{body{{padding:14px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style><main>
<div class="card"><span class="tag">当前进度</span><h1>法语核心3500｜15天打卡</h1><div class="muted">词条按共同编号结构建立；图表显示各日已制作数量。</div><div class="progress"><i></i></div><strong>{actual} / {TARGET_TOTAL} 条 · {percent:.1f}%</strong><div class="muted">剩余 {TARGET_TOTAL-actual} 条待制作。每一天计划 233–234 条。</div></div>
<div class="card metrics"><div class="metric"><strong>{actual}</strong>已整理</div><div class="metric"><strong>{TARGET_TOTAL-actual}</strong>待补充</div><div class="metric"><strong>{finished_days} / {len(PLAN_COUNTS)}</strong>已备齐词表的天数</div><div class="metric"><strong>233–234</strong>每天目标</div></div>
<div class="card"><h2>每日计划与已制作条数</h2><div class="muted">浅蓝色为计划条数；深蓝色为当前已制作条数。数字为 Day 编号。</div><svg viewBox="0 0 810 225" role="img" aria-label="15 天每日计划与词表制作进度"><line x1="35" y1="186" x2="790" y2="186" stroke="#94a3b8"/><line x1="35" y1="130" x2="790" y2="130" stroke="#e2e8f0"/><line x1="35" y1="74" x2="790" y2="74" stroke="#e2e8f0"/><text x="12" y="190">0</text><text x="5" y="134">100</text><text x="5" y="78">200</text>{''.join(bar_svg)}<rect x="610" y="20" width="12" height="12" fill="#D9EAF7"/><text x="630" y="31">计划</text><rect x="690" y="20" width="12" height="12" fill="#2563EB"/><text x="710" y="31">已制作</text></svg></div>
<div class="card"><h2>每日编号与打卡目标</h2><table><thead><tr><th>天数</th><th>编号</th><th>计划</th><th>已制作</th><th>状态</th></tr></thead><tbody>{''.join(f'<tr><td>Day {d}</td><td>{a:04d}–{b:04d}</td><td>{t}</td><td>{m}</td><td>{"词表已备齐" if m == t else "待续"}</td></tr>' for d,t,m,a,b in plan)}</tbody></table></div>
<div class="card muted">本页按当前实际词条数生成；没有把待补编号计作已完成。</div></main></html>'''
    (ROOT / "15天打卡_进度概览.html").write_text(html, encoding="utf-8")


def main():
    if DATA_FILE.exists():
        entries = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        if any((SOURCE_DIR / name).exists() for name in ("Lexique4.1.zip", "FleLex_TT_Beacco.tsv", "lexik_formes_mots.tsv.gz")):
            entries, lexical_report = enrich(entries)
        else:
            qa_file = ROOT / "首批质检报告.json"
            previous_qa = json.loads(qa_file.read_text(encoding="utf-8")) if qa_file.exists() else {}
            lexical_report = previous_qa.get("lexical_source_check", {"counts": {}, "source_dir_available": False})
            counts = Counter()
            counts["lexique_lemma_found"] = sum(bool(e.get("lexique_frequency")) for e in entries)
            counts["lexique_lemma_missing"] = sum(not e.get("lexique_frequency") for e in entries)
            counts["flelex_found"] = sum(bool(e.get("flelex_level")) for e in entries)
            counts["flelex_missing"] = sum(not e.get("flelex_level") for e in entries)
            counts["lexik_lemma_found"] = sum("LeXiK" in e.get("source_status", "") for e in entries)
            counts["lexik_missing"] = sum("LeXiK未匹配" in e.get("source_status", "") for e in entries)
            counts["ipa_crossmatch"] = sum("两源 IPA 一致" in e.get("ipa_check", "") for e in entries)
            counts["ipa_source_difference"] = sum("转写有差异" in e.get("ipa_check", "") for e in entries)
            counts["lexique_alias_used"] = sum(bool(e.get("lexique_alias")) for e in entries)
            lexical_report["counts"] = dict(counts)
            lexical_report["source_dir_available"] = False
    else:
        raise SystemExit(f"找不到词库母数据：{DATA_FILE}")

    if len(entries) > TARGET_TOTAL:
        raise SystemExit("词条超过3500条，请先检查编号范围。")

    boundaries = day_boundaries()
    entries = [
        entry
        for start, end in zip([0, *boundaries[:-1]], boundaries)
        for entry in sorted(entries[start:end], key=lambda e: french_sort_key(e["fr"]))
    ]
    for item_id, entry in enumerate(entries, 1):
        entry.pop("topic", None)
        entry["id"] = item_id
        entry["day"] = day_for_id(item_id)

    if len(entries) != 234 and len(entries) < 3500:
        print(f"当前数据 {len(entries)} 条；目标总量 {TARGET_TOTAL} 条。")
    seen = [e["fr"].casefold() for e in entries]
    duplicate_count = len(seen) - len(set(seen))
    missing_ipa = sum(not e.get("ipa") for e in entries)
    if duplicate_count or missing_ipa:
        raise SystemExit(f"质检未通过：重复词元={duplicate_count}；缺失IPA={missing_ipa}。检查数据后再生成交付件。")
    DATA_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for day in range(1, 16):
        day_entries = [e for e in entries if e["day"] == day]
        if not day_entries:
            continue
        make_markdown(day_entries, "study", ROOT / f"中法学习版_Day{day:02d}.md", day)
        make_markdown(day_entries, "fr_to_zh", ROOT / f"法填中练习版_Day{day:02d}.md", day)
        make_markdown(day_entries, "zh_to_fr", ROOT / f"中填法练习版_Day{day:02d}.md", day)
        build_pdf(day_entries, day)
    build_workbook(entries, lexical_report)
    build_html(entries)
    pos_counts = Counter(e["pos"] for e in entries)
    edition_id_orders = [
        [row[0] for row in edition_rows(entries, mode)]
        for mode in ("study", "fr_to_zh", "zh_to_fr")
    ]
    qa = {
        "generated_on": date.today().isoformat(),
        "actual_entries": len(entries),
        "ids_contiguous": [e["id"] for e in entries] == list(range(1, len(entries) + 1)),
        "same_order_across_three_editions": edition_id_orders[0] == edition_id_orders[1] == edition_id_orders[2],
        "duplicate_lemmas": duplicate_count,
        "missing_ipa": missing_ipa,
        "noun_entries_missing_gender": sum(e["pos"].startswith("n.") and not re.search(r"n\.[mf]", e["pos"]) for e in entries),
        "lexique_missing_words": [e["fr"] for e in entries if not e.get("lexique_frequency")],
        "lexique_alias_words": [e["fr"] + " → " + e["lexique_alias"] for e in entries if e.get("lexique_alias")],
        "flelex_missing_words": [e["fr"] for e in entries if not e.get("flelex_level")],
        "flelex_alias_words": [e["fr"] + " → " + e["flelex_alias"] for e in entries if e.get("flelex_alias")],
        "flelex_level_counts": dict(Counter(e.get("flelex_level") or "未匹配" for e in entries)),
        "ipa_source_difference_words": [e["fr"] for e in entries if "转写有差异" in e.get("ipa_check", "")],
        "ipa_source_differences": [
            {"fr": e["fr"], "LeXiK": e.get("ipa", ""), "Lexique": e.get("lexique_ipa", ""), "note": e.get("ipa_check", "")}
            for e in entries if "转写有差异" in e.get("ipa_check", "")
        ],
        "ipa_confidence_distribution": dict(Counter(e.get("lexik_confidence", "未标") or "未标" for e in entries)),
        "unit_type_counts": dict(Counter(e.get("unit_type", "未标") for e in entries)),
        "collocations_included": [e["fr"] + " → " + e["collocation"] for e in entries if e.get("collocation")],
        "lexical_source_check": lexical_report,
        "pos_counts": dict(pos_counts),
        "batches": {
            f"Day {day}": {
                "count": len(batch),
                "range": [batch[0]["id"], batch[-1]["id"]],
                "flelex_levels": dict(Counter(e.get("flelex_level") or "未匹配" for e in batch)),
                "ipa_source_differences": [e["fr"] for e in batch if "转写有差异" in e.get("ipa_check", "")],
            }
            for day in sorted({e["day"] for e in entries})
            if (batch := [e for e in entries if e["day"] == day])
        },
        "daily_targets": PLAN_COUNTS,
        "day1_count": sum(e["day"] == 1 for e in entries),
        "daily_actual_counts": [sum(e["day"] == day for e in entries) for day in range(1, 16)],
        "remaining_to_target": TARGET_TOTAL - len(entries),
    }
    (ROOT / "首批质检报告.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
