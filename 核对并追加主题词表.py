from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
CACHE = Path(os.environ['TEMP']) / 'fr_core3500_sources'
TEMP = CACHE.parent
MAIN = ROOT / '法语核心3500_总表.md'
LOG = ROOT / '法语核心3500_分批复核.md'
TITLE = re.compile(r'^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|', re.M)
POS = {'n.': ('NOUN', 'NOM'), 'v.': ('VERB', 'VER'), 'adj.': ('ADJ', 'ADJ'), 'adv.': ('ADV', 'ADV'), 'conj.': ('CCONJ|SCONJ', 'CON'), 'prép.': ('ADP', 'PRE')}


def key(text):
    return unicodedata.normalize('NFC', text.strip()).casefold().replace('’', "'").replace('œ', 'oe').replace('æ', 'ae').replace('‑', '-')


def lem(text, pos):
    normalized = key(text)
    if pos.startswith('v. pron.'):
        return re.sub(r"^(?:se |s')", '', normalized)
    return normalized


def align_markdown_tables(markdown):
    """Align pipe columns consistently across the topic tables in a document."""
    def display_width(cell):
        return sum(
            2 if unicodedata.east_asian_width(char) in {'W', 'F'} else 1
            for char in cell
            if not unicodedata.category(char).startswith('M')
        )

    lines = markdown.splitlines(keepends=True)
    tables = {
        i: [cell.strip() for cell in line.strip()[1:-1].split('|')]
        for i, line in enumerate(lines)
        if line.startswith('|') and line.rstrip().endswith('|')
    }
    if not tables:
        return markdown

    separator = re.compile(r'^:?-{3,}:?$')
    widths = [0] * len(next(iter(tables.values())))
    for cells in tables.values():
        if not all(separator.fullmatch(cell) for cell in cells):
            widths = [max(width, display_width(cell)) for width, cell in zip(widths, cells, strict=True)]

    for i, cells in tables.items():
        if all(separator.fullmatch(cell) for cell in cells):
            markers = ['-' * (width + 2) for width in widths]
            markers[0] = markers[0][:-1] + ':'
            row = '|' + '|'.join(markers) + '|'
        else:
            padded = [cell + ' ' * (width - display_width(cell)) for width, cell in zip(widths, cells, strict=True)]
            row = '| ' + ' | '.join(padded) + ' |'
        lines[i] = row + ('\n' if lines[i].endswith('\n') else '')

    return ''.join(lines)


def read_batch(ids):
    candidates = []
    for theme in ids:
        path = TEMP / f'fr3500_theme{theme:02}.txt'
        for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split('|')]
            head, pos, zh, collocation = (parts + [''])[:4]
            if not (3 <= len(parts) <= 4 and all((head, pos, zh))):
                raise ValueError(f'{path.name}:{n} bad row')
            kind = pos.removeprefix('loc. ')
            upos, lexpos = next(value for prefix, value in POS.items() if kind.startswith(prefix))
            gender = set(re.findall(r'(?<=[./])[mf](?=[./])', kind)) if kind.startswith('n.') else set()
            if kind.startswith('n.') and not gender:
                raise ValueError(f'{head} missing gender')
            candidates.append(dict(theme=theme, head=head, pos=pos, zh=zh, collocation=collocation, upos=upos.split('|'), lexpos=lexpos, genders=gender, lemma=lem(head, pos), expression=pos.startswith('loc. ')))
    return candidates


def source_indexes(candidates):
    forms = defaultdict(list)
    expression_forms = defaultdict(list)
    props = {}
    lexique = defaultdict(list)
    wanted = {row['lemma'] for row in candidates}
    with gzip.open(CACHE/'lexik_formes_mots.tsv.gz', 'rt', encoding='utf-8', newline='') as data:
        for line in data:
            fields = line.split('\t')
            lemma = key(fields[1])
            if lemma in wanted and key(fields[0]) in wanted:
                forms[key(fields[0])].append((fields[0], lemma, fields[2], fields[3], fields[5], fields[9], fields[-1].strip()))
    form_ids = {row[2] for entries in forms.values() for row in entries}
    with gzip.open(CACHE/'lexik_lemmes.tsv.gz', 'rt', encoding='utf-8', newline='') as data:
        for row in csv.DictReader(data, delimiter='\t', quoting=csv.QUOTE_NONE):
            if row['id_lemme'] in form_ids:
                props[row['id_lemme']] = row['proprietes']
    with gzip.open(CACHE/'lexik_formes_multimots.tsv.gz', 'rt', encoding='utf-8', newline='') as data:
        for row in csv.DictReader(data, delimiter='\t', quoting=csv.QUOTE_NONE):
            if key(row['ortho']) in wanted:
                expression_forms[key(row['ortho'])].append((row['ortho'], key(row['ortho']), row['proprietes'], row['upos'], row['phone'], row['freq_opensubtitles'], row['confiance']))
    with zipfile.ZipFile(CACHE/'Lexique4.1.zip') as archive:
        with archive.open('Lexique4.1/Lexique4.1.tsv') as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''), delimiter='\t'):
                ortho = key(row['1_Mot'])
                if ortho in wanted and key(row['4_Lemme']) == ortho:
                    lexique[ortho].append(row)
    return forms, expression_forms, props, lexique


def check(row, indexes):
    forms, expression_forms, props, lexique = indexes
    stem = row['lemma']
    expression = row['expression']
    exact = [r for r in (expression_forms[stem] + forms[stem] if expression else forms[stem]) if r[1] == stem and r[3] in row['upos'] and r[4]]
    lx = [r for r in lexique[stem] if r['5_Cgram'] == row['lexpos'] and r['3_Phono_IPA']]
    def genders(r):
        source = props.get(r[2], r[2] if expression else '')
        match = re.search(r'(?:^|;)genre:([^;]+)', source)
        return set(match[1].split(',')) if match else set()
    def order(r):
        expected = row['genders']
        return (len(genders(r) & expected), {'A': 4, 'B': 3, 'R': 2, 'C': 1}.get(r[6], 0), float(r[5] or 0))
    exact.sort(key=order, reverse=True)
    lex = sorted(lx, key=lambda r: (r['7_Genre'] in row['genders'], float(r['12_FreqLemme'] or 0)), reverse=True)
    found_genders = set().union(*(genders(r) for r in exact), {r['7_Genre'] for r in lx})
    acceptable = [r for r in exact if r[6] in {'A', 'B', 'R', 'C'} and (not row['genders'] or row['genders'] & genders(r))]
    clean = lambda s: re.sub(r'\{([^{}|]+)(?:\|[^{}]*)?\}', r'\1', re.sub(r'\[[^\]]*\]', '', s)).replace('‿', '').replace('*h*', '').strip(' /')
    normal = lambda s: re.sub(r'[.\-·\s]', '', clean(s)).replace('g', 'ɡ')
    if not acceptable and expression:
        corroborated = {clean(r[4]) for r in forms[stem] if r[1] == stem and r[6] in {'A', 'B', 'R', 'C'}}
        acceptable = [r for r in exact if r[6] == 'S' and clean(r[4]) in corroborated]
    a = clean(acceptable[0][4]) if acceptable else ''
    b = clean(lex[0]['3_Phono_IPA']) if lex else ''
    if row['genders'] - found_genders:
        return None, f"性别待核（源：{','.join(sorted(found_genders))}）"
    if not a and not b:
        return None, '缺同词元同词性的可用 IPA'
    ipa = a or b
    if row['pos'].startswith('v. pron.'):
        ipa = ('sə.' if key(row['head']).startswith('se ') else 's') + ipa
    return dict(ipa=f'/{ipa}/', source='LeXiK' if a else 'Lexique', lexik=a, lexique=b, diff=bool(a and b and normal(a) != normal(b)), freq=float(lex[0]['12_FreqLemme'] or 0) if lex else None), ''


def main():
    args = sys.argv[1:]
    write = '--write' in args
    ids = tuple(int(x) for x in args if x != '--write')
    plan = {int(i): (theme.strip(), int(n)) for i, theme, n in TITLE.findall((ROOT/'词库主题与词数规划.md').read_text(encoding='utf-8-sig'))}
    if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids) or not set(ids) <= plan.keys():
        raise SystemExit('每批使用 1–3 个不同主题编号')
    past = MAIN.read_text(encoding='utf-8') if MAIN.exists() else ''
    if any(re.search(rf'^## {i:02} ', past, flags=re.M) for i in ids):
        raise SystemExit('主题已经在总表，先核对现有内容。')
    old = re.findall(r'^\|\s*(\d{4})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|', past, re.M)
    seen = {lem(head, pos) for _, head, pos in old}
    candidates = read_batch(ids)
    indexes = source_indexes(candidates)
    accepted, excluded = [], []
    for row in candidates:
        if row['lemma'] in seen:
            excluded.append((row['theme'], row['head'], '词元已计入其他条目'))
            continue
        evidence, issue = check(row, indexes)
        if issue:
            excluded.append((row['theme'], row['head'], issue))
            continue
        seen.add(row['lemma'])
        accepted.append((row, evidence))
    print('Counts:', dict(Counter(row['theme'] for row,_ in accepted)), 'total:', len(accepted), 'excluded:', len(excluded))
    for i, head, reason in excluded:
        print(i, head, reason)
    print('Source:', dict(Counter(e['source'] for _,e in accepted)), 'different:', sum(e['diff'] for _,e in accepted))
    print('Low frequency:', ', '.join(r['head'] for r,e in accepted if e['freq'] is not None and e['freq'] < 1))
    print('No Lexique match:', ', '.join(r['head'] for r,e in accepted if e['freq'] is None))
    comparison = [f"| {r['head']} | /{e['lexik']}/ | /{e['lexique']}/ |" for r,e in accepted if e['diff']]
    if not write:
        (TEMP/'fr3500_review.json').write_text(json.dumps([{**{k:v for k,v in r.items() if k!='genders'}, **e} for r,e in accepted], ensure_ascii=False, indent=2), encoding='utf-8')
        return
    header = '# 法语核心3500｜总表\n\n按主题收录词汇。各主题建议词数可随筛选调整。编号用于查找。来源为 [LeXiK 1.0（Lectura）](https://lexique.lectura.world/lexik/telechargement) 与 [Lexique 4.1](https://www.lexique.org/)，均为 CC BY-SA 4.0。\n'
    log = LOG.read_text(encoding='utf-8') if LOG.exists() else '# 法语核心3500｜分批复核\n'
    body = []
    count = len(old)
    for i in ids:
        title, target = plan[i]
        batch = [(row,evidence) for row,evidence in accepted if row['theme']==i]
        body.extend(['', f'## {i:02} {title}', '', '| 编号 | 法语词头 | 词性 | IPA | 中文释义 | 固定介词搭配 |', '|---:|---|---|---|---|---|'])
        for row,evidence in batch:
            count += 1
            body.append(f"| {count:04} | {row['head']} | {row['pos']} | {evidence['ipa']} | {row['zh']} | {row['collocation']} |")
    review = ['','## 主题 ' + '、'.join(f'{i:02}' for i in ids), '', f"本批纳入 {len(accepted)} 条，累计 {count} 条。逐条按词头与词性匹配音标，名词性别与来源对照。逐项编辑常见义与固定搭配，并抽查需要辨析的结构及主题用法。", '', '| 主题 | 规划参考 | 本批收录 |', '|---|---:|---:|']
    for i in ids:
        title,target = plan[i]
        review.append(f'| {title} | {target} | {sum(r["theme"]==i for r,_ in accepted)} |')
    review.extend(['', f"音标来源：LeXiK {sum(e['source']=='LeXiK' for _,e in accepted)} 条，Lexique 补充 {sum(e['source']=='Lexique' for _,e in accepted)} 条。来源不同的转写列在下方，表中采用 LeXiK。仍待全表总查与人工抽样。"])
    if excluded:
        review.extend(['', '未入表的候选：' + '、'.join(f'{head}（{reason}）' for _,head,reason in excluded) + '。'])
    if comparison:
        review.extend(['', '| 词头 | LeXiK | Lexique |','|---|---|---|', *comparison])
    if ids == (1,2,3):
        review.extend(['', '语境与义项抽查：[attendre](https://www.dictionnaire-academie.fr/article/A9A3030)、[disposer](https://www.dictionnaire-academie.fr/article/A9D2723)、[commencer](https://www.dictionnaire-academie.fr/article/A9C3106)、[postuler](https://www.dictionnaire-academie.fr/article/DNP0194)、[emploi](https://www.larousse.fr/dictionnaires/francais/emploi/28956)、[住房租约用语](https://www.service-public.fr/particuliers/vosdroits/N349?lang=fr)、[招聘与应聘用语](https://www.francetravail.fr/employeur/lessentiel-pour-embaucher/le-recrutement-par-les-competenc/recruter-par-les-competences--la.html)。\n\n复核更正了 chômage 的名词性别，并将重复的 entretien 归入一条保留两个常见义。语义和介词搭配采用编辑复核与重点抽查，完整的逐条词典验证留待进一步核查。'])
    master = (past or header).rstrip() + '\n' + '\n'.join(body) + '\n'
    MAIN.write_text(align_markdown_tables(master), encoding='utf-8')
    LOG.write_text(log.rstrip() + '\n' + '\n'.join(review) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
