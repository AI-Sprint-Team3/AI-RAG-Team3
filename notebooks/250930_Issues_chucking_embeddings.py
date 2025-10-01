#!/usr/bin/env python
# coding: utf-8

# # Trouble Shooting Notebook
# 
# 본 문서는 코사인 **유사도 점수가 낮아 사용하지 않기로한 청킹 전략과 임베딩 결과가 포함**된 정리본입니다.
# 
# Trouble Shooting 과정과 issue 해결 방법을 기재했으며,
# 환경설정/설치/디버그/중복 코드는 제거하고, 섹션별로 재배치했습니다.

# # Index
# 
# 1. Parsing 핵심요약
#     - 문서별 통합 텍스트 머지
# 2. Chunking : ``docs_chuks.jsonl``
# 3. Embeddings & Vector Store
# 4. Issue
#     - 1) 코사인 유사도 결과 미진
#     - 2) 병합 문서 merge_key의 text 41개 누락
# 5. Artifacts Merge
#     - 1차 : 1차 청킹 & 임베딩 작업 후 (코사인 유사도 문제 포함)
#     - 2차 : 병합 문석 text 누락 이슈 해결 이후 

# # 1. 문서별 통합 텍스트 머지

# ### 지금까지 확보한 소스 (정리)
# 
# - **PDF 본문/표**
#     - 본문: `/interim/pages_all_merged.jsonl` → 클린: `/interim/pages_pdf.cleaned.jsonl`
#     - 표 스냅샷: 디렉토리 `/interim/table_snapshots` + 매니페스트 `/interim/table_snapshots_manifest.jsonl`
#     - 표 필터링 결과: 디렉토리 `/interim/table_snapshots_filtered` + 매니페스트 `/interim/table_snapshots_manifest.filtered.jsonl`
#     - (본문+표 메타 통합 리프레젠테이션) **`/interim/rep_pdf.jsonl`**
# - **HTML/HWP 계열**
#     - 원본 변환: `/raw/converted/html`, `/raw/converted/docs`(TXT), CSS 동일 경로
#     - 최소 클린 TXT: `/processed/clean_txt`
#     - HTML→Plain: `/processed/html_plain`
# - **XML 본문**
#     - **`/interim/xml_text.jsonl`**
# - **자산 OCR(HTML/XML)**
#     - **`/interim/ocr_text_by_doc.jsonl`** (이미지 텍스트 집계)
# - **조인키/충돌 해소**
#     - **`/interim/joinkey_fixes.json`** (merge_key 매핑)
#     - with-mergekey 버전: `xml_text.with_mergekey.jsonl`, `ocr_text_by_doc.with_mergekey.jsonl`

# #### 1. 앞으로의 플로우
# 
# 1. doc_id 정규화 & 키 고정
#     - **규칙 통일**(zero-width 제거, 괄호류/기호 정리 등)로 `join_key` 재계산.
#     - *`joinkey_fixes.json`*을 “정답 맵”으로 고정해 **`merge_key`*를 전 소스에 부여.
#     - TOP 리스트에서 보이는 **이명(동일 문서 다른 표기)**는 alias 맵에 추가해 충돌/중복 제거.
# 2. 소스별 대표 JSONL로 정리(“rep_*”)
#     - **rep_xml.jsonl**: `xml_text.jsonl`에 `merge_key` 부여 + 통계(문자수/라인수/경로 샘플).
#     - **rep_html.jsonl**: `/processed/html_plain`(+ 필요한 경우 `/processed/clean_txt`)에서 본문 수집, `merge_key` 부여.
#     - **rep_ocr.jsonl**: `ocr_text_by_doc.jsonl`에 `merge_key` 부여(이미지수, 승급단계 분포 포함).
#     - **rep_pdf.jsonl**: (이미 있음) `merge_key` 부여 + 표 매니페스트(filtered)와 조인해 **문서별 표 리스트**(CSV 경로/페이지/격자지표) 포함.
#     - 각 파일은 공통 스키마 일부를 맞춤: `merge_key, doc_id, source_paths_sample, chars, line_count, has_*`.
# 3. 품질 점검 & 커버리지 리포트
#     - 소스별 문서수, **only_in_**분포, 빈/짧은 문서 플래그, 상위/하위 문자수 목록 출력.
#     - **충돌 4건**은 해시 접미사로 분리된 상태 유지하거나, 진짜 동일 문서면 alias로 통합 결정.
# 4. 최종 병합 산출물 생성 (`docs_merged.jsonl`)
#     - **키:** `merge_key` (문서당 1행)
#     - **핵심 필드(예시):**
#         - 식별: `merge_key`, `canonical_doc_id`, `doc_id_variants`
#         - 소스 존재: `has_xml/html/pdf/ocr/tables`
#         - 텍스트: `text_xml`, `text_html`, `text_pdf`, `text_ocr_html`, `text_ocr_xml`, `text_ocr_combined`, `text_all`
#         - 표: `tables`(리스트: `csv_path, page, n_rows, n_cols, area_ratio, ruling_density…`)
#         - 통계: 각 소스 `chars_*`, `image_count`, `page_count_pdf`, `total_chars`
#         - 경로 샘플: `source_paths_sample`
#     - **병합 정책(우선순위 & 중복 제거):**
#         - 우선순위 제안: **XML > HTML > PDF(cleaned) > OCR**
#             
#             (문장/라인 단위 정규화 후 중복 라인은 1회만, 원문 흐름 우선)
#             
#         - 표는 **본문과 분리 유지**(검색/QA에서 강점).
# 5. 검증 리포트(머지 후)
#     - 최종 문서수(유니온), 소스 커버리지 매트릭스, `only_in_*` 잔여 여부, 충돌 해소 요약.
#     - 샘플 스팟 QA 목록(랜덤 N + 상/하위 N).
# 6. 임베딩용 청크 생성 (`docs_chunks.jsonl`)
#     - `text_all` 기준으로 1200~1500자, 200자 오버랩 권장.
#     - 청크 메타: `merge_key, chunk_id, chunk_index, source_breakdown(비율), approx_page_ranges(가능 시)`.
# 7. 아티팩트 정리
#     - `/processed/`에 `docs_merged.jsonl`, `docs_chunks.jsonl`, 리포트(`.txt/*.csv`),
#         
#         표 인덱스(`tables_index.jsonl` or `rep_pdf.jsonl` 내 포함) 정리.

# #### 2. 소스별 대표 JSONL로 정리(“rep_*”)

# - **rep_xml.jsonl**: `xml_text.jsonl`에 `merge_key` 부여 + 통계(문자수/라인수/경로 샘플).
# - **rep_html.jsonl**: `/processed/html_plain`(+ 필요한 경우 `/processed/clean_txt`)에서 본문 수집, `merge_key` 부여.
# - **rep_ocr.jsonl**: `ocr_text_by_doc.jsonl`에 `merge_key` 부여(이미지수, 승급단계 분포 포함).
# - **rep_pdf.jsonl**: (이미 있음) `merge_key` 부여 + 표 매니페스트(filtered)와 조인해 **문서별 표 리스트**(CSV 경로/페이지/격자지표) 포함.
# - 각 파일은 공통 스키마 일부를 맞춤: `merge_key, doc_id, source_paths_sample, chars, line_count, has_*`.

# [표준화된 대표 파일들]
# 
# - 생성물
#     - ``/home/spai0308/data/interim/rep_xml.jsonl``
#     - ``/home/spai0308/data/interim/rep_html.jsonl``
#     - ``/home/spai0308/data/interim/rep_ocr.jsonl``
#     - ``/home/spai0308/data/interim/rep_pdf.with_mergekey.jsonl`` (기존 rep_pdf는 덮지 않고 별도 출력)

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re, hashlib
from collections import defaultdict

BASE        = Path("/home/spai0308/data")
INTERIM     = BASE / "interim"
PROCESSED   = BASE / "processed"

# 입력
XML_JSONL         = INTERIM / "xml_text.jsonl"
OCR_JSONL         = INTERIM / "ocr_text_by_doc.jsonl"
PDF_REP_JSONL     = INTERIM / "rep_pdf.jsonl"          # 이미 만들어 둔 PDF 리프레젠테이션
HTML_PLAIN_DIR    = PROCESSED / "html_plain"           # .txt(HTML→plain) 파일들

# 조인키/충돌해결 맵
FIXES_JSON        = INTERIM / "joinkey_fixes.json"

# 출력
REP_XML_OUT       = INTERIM / "rep_xml.jsonl"
REP_HTML_OUT      = INTERIM / "rep_html.jsonl"
REP_OCR_OUT       = INTERIM / "rep_ocr.jsonl"
REP_PDF_OUT       = INTERIM / "rep_pdf.with_mergekey.jsonl"

# ---------------------------
# helpers
# ---------------------------
def make_join_key(s: str) -> str:
    if not s: return ""
    t = s.replace("\u200b","")                             # zero-width
    t = re.sub(r"[(){}\[\]<>]+"," ", t)                    # 괄호류 → 공백
    t = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+","_",t)  # 한/영/숫자/_만
    t = re.sub(r"_+","_",t).strip("_").lower()
    return t

def short_hash(s: str, n=6) -> str:
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:n]

def load_jsonl(p: Path):
    rows=[]
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def write_jsonl(rows, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_paths_any(rec):
    for k in ("source_paths","paths","xml_paths","image_paths","source_paths_sample"):
        v = rec.get(k)
        if isinstance(v, list) and v: return v
    p = rec.get("source_path")
    if isinstance(p, str) and p: return [p]
    return []

def get_text_any(rec):
    # 가장 유력한 텍스트 필드 후보들 중 첫 번째를 사용
    for k in ("text", "xml_text", "html_text", "combined_text", "text_combined", "content"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

# fixes.json 로드 → (join_key, doc_id) → merge_key
fix_items = load_jsonl(FIXES_JSON) if FIXES_JSON.suffix == ".jsonl" else (json.loads(FIXES_JSON.read_text("utf-8")) if FIXES_JSON.exists() else [])
fix_map = {}
by_jk = defaultdict(set)
for it in fix_items:
    jk = it.get("join_key","")
    did = str(it.get("doc_id","")).strip()
    mk = it.get("merge_key", jk)
    fix_map[(jk, did)] = mk
    by_jk[jk].add(did)

colliding_jks = {jk for jk, ids in by_jk.items() if len(ids) > 1}

def apply_merge_key(doc_id: str):
    """doc_id→join_key 만들고 fixes.json을 우선 적용.
       (jk, did)가 없고 jk 충돌군이면 해시 접미사로 분리, 아니면 jk 그대로."""
    did = (doc_id or "").strip()
    jk  = make_join_key(did)
    mk  = fix_map.get((jk, did))
    if mk:
        return mk
    if jk in colliding_jks:
        return f"{jk}_{short_hash(did)}"
    return jk

# ---------------------------
# 1) rep_xml.jsonl
# ---------------------------
xml_rows = load_jsonl(XML_JSONL)
rep_xml=[]
for r in xml_rows:
    did = str(r.get("doc_id") or r.get("doc") or "").strip()
    mk  = apply_merge_key(did)
    txt = get_text_any(r)
    rep_xml.append({
        **r,
        "merge_key": mk,
        "doc_id": did,
        "source": "xml",
        "chars": len(txt),
        "line_count": (txt.count("\n")+1 if txt else 0),
        "source_paths_sample": get_paths_any(r)[:2],
    })
write_jsonl(rep_xml, REP_XML_OUT)

# ---------------------------
# 2) rep_html.jsonl (HTML→plain txt 디렉토리에서 구성)
# ---------------------------
rep_html=[]
if HTML_PLAIN_DIR.exists():
    for p in sorted(HTML_PLAIN_DIR.glob("*.txt")):
        did = p.stem  # 파일명(확장자 제외)이 곧 doc_id
        mk  = apply_merge_key(did)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        rep_html.append({
            "merge_key": mk,
            "doc_id": did,
            "source": "html",
            "text_html": txt,
            "chars": len(txt),
            "line_count": (txt.count("\n")+1 if txt else 0),
            "source_paths_sample": [str(p)],
        })
write_jsonl(rep_html, REP_HTML_OUT)

# ---------------------------
# 3) rep_ocr.jsonl
# ---------------------------
ocr_rows = load_jsonl(OCR_JSONL)
rep_ocr=[]
for r in ocr_rows:
    did = str(r.get("doc_id") or r.get("doc") or "").strip()
    mk  = apply_merge_key(did)
    # 가급적 원래 필드를 보존하되, 요약 통계만 추가
    combo_txt = r.get("combined_text") or r.get("text_combined") or ""
    rep_ocr.append({
        **r,
        "merge_key": mk,
        "doc_id": did,
        "source": "ocr",
        "chars": len(combo_txt) if isinstance(combo_txt, str) else 0,
        "line_count": (combo_txt.count("\n")+1 if isinstance(combo_txt, str) else 0),
        "source_paths_sample": get_paths_any(r)[:2],
    })
write_jsonl(rep_ocr, REP_OCR_OUT)

# ---------------------------
# 4) rep_pdf.with_mergekey.jsonl (기존 rep_pdf에 merge_key 부여)
# ---------------------------
rep_pdf_in = load_jsonl(PDF_REP_JSONL)
rep_pdf=[]
for r in rep_pdf_in:
    did = str(r.get("doc_id") or r.get("doc") or "").strip()
    mk  = apply_merge_key(did)
    txt = get_text_any(r)
    rep_pdf.append({
        **r,
        "merge_key": mk,
        "doc_id": did,
        "source": "pdf",
        "chars": len(txt),
        "line_count": (txt.count("\n")+1 if txt else 0),
        "source_paths_sample": get_paths_any(r)[:2],
    })
write_jsonl(rep_pdf, REP_PDF_OUT)

# ---------------------------
# 요약 출력
# ---------------------------
def uniq_keys(rows): return len({r["merge_key"] for r in rows if "merge_key" in r})

print("[DONE] rep_* 생성 완료")
print(f"- rep_xml:  rows={len(rep_xml)}, uniq_merge_keys={uniq_keys(rep_xml)}, out={REP_XML_OUT}")
print(f"- rep_html: rows={len(rep_html)}, uniq_merge_keys={uniq_keys(rep_html)}, out={REP_HTML_OUT}")
print(f"- rep_ocr:  rows={len(rep_ocr)}, uniq_merge_keys={uniq_keys(rep_ocr)}, out={REP_OCR_OUT}")
print(f"- rep_pdf:  rows={len(rep_pdf)}, uniq_merge_keys={uniq_keys(rep_pdf)}, out={REP_PDF_OUT}")
print(f"- collision join_keys: {len(colliding_jks)} (해시 접미사 분리 적용됨)")


# #### 3. 품질 점검 & 커버리지 리포트
# 
# 
# - 소스별 문서수, **only_in_**분포, 빈/짧은 문서 플래그, 상위/하위 문자수 목록 출력.
# 
# 
# - **충돌 4건**은 해시 접미사로 분리된 상태 유지하거나, 진짜 동일 문서면 alias로 통합 결정.

# 1. rep_xml / rep_html / rep_ocr / rep_pdf.with_mergekey 를 불러와 커버리지/문자수 매트릭스를 만든 뒤 리포트로 저장하고,
# 2. PDF 쪽의 doc_id/merge_key 를 경로·메타에서 다시 유추해 rep_pdf.fixed.jsonl 로 보정(가능하면)한다.
# 3. 이후 병합 단계에서 이 fixed 파일을 사용할 수 있게 함

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re, hashlib
from collections import defaultdict

BASE        = Path("/home/spai0308/data")
INTERIM     = BASE / "interim"

# 입력
REP_XML_IN   = INTERIM / "rep_xml.jsonl"
REP_HTML_IN  = INTERIM / "rep_html.jsonl"
REP_OCR_IN   = INTERIM / "rep_ocr.jsonl"
REP_PDF_IN   = INTERIM / "rep_pdf.with_mergekey.jsonl"

# 출력
QC_TXT       = INTERIM / "merge_qc_report.txt"
QC_CSV       = INTERIM / "merge_coverage.csv"
REP_PDF_FIX  = INTERIM / "rep_pdf.fixed.jsonl"  # 보정본(가능하면 생성)

# ---------------------------
def load_jsonl(p: Path):
    rows=[]
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

def write_jsonl(rows, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def make_join_key(s: str) -> str:
    if not s: return ""
    t = s.replace("\u200b","")
    t = re.sub(r"[(){}\[\]<>]+"," ", t)
    t = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+","_",t)
    t = re.sub(r"_+","_",t).strip("_").lower()
    return t

def short_hash(s: str, n=6) -> str:
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:n]

def any_field(rec, keys):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            # 리스트면 첫 문자열 경로를 반환
            for item in v:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""

def derive_docid_from_path(path_str: str) -> str:
    """
    경로에서 문서 폴더명 후보를 뽑아 doc_id로 사용.
    예) /.../table_snapshots/<DOC>/page_0001.png → <DOC>
        /.../<DOC>/.../page_12.json → <DOC>
    """
    try:
        p = Path(path_str)
    except Exception:
        return ""
    # 가장 그럴듯한 상위 폴더명 고르기: 파일 바로 위(parent) 또는 그 위
    # 규칙: page, table, snapshot, images, pages 등의 폴더는 건너뛰고 그 위를 후보로
    skip = {"page","pages","images","image","tables","table","snapshots","snapshot"}
    parts = list(p.parts)
    # 파일명 제외하고 역방향으로 검사
    for seg in reversed(parts[:-1]):
        seg_clean = seg.strip()
        if not seg_clean: 
            continue
        low = seg_clean.lower()
        # 확실히 스킵할 폴더명 걸러내기
        if any(key in low for key in skip):
            continue
        # 공백/이상문자 정리 안 된 이름 그대로 doc_id로 사용(조인키는 따로 계산)
        return seg_clean
    # 실패하면 파일명(확장자 제외)라도 사용
    stem = Path(parts[-1]).stem if parts else ""
    return stem

def ensure_merge_key(did: str, fixes=None):
    # fixes는 이번 단계에선 없지만, 충돌 감지 시 해시 접미사
    jk = make_join_key(did)
    if not jk:
        # did가 비면 해시 기반 임시 키
        return f"_tmp_{short_hash(did or 'none')}"
    return jk

# ---------------------------
# 1) 로드
# ---------------------------
rep_xml  = load_jsonl(REP_XML_IN)
rep_html = load_jsonl(REP_HTML_IN)
rep_ocr  = load_jsonl(REP_OCR_IN)
rep_pdf  = load_jsonl(REP_PDF_IN)

def uniq_keys(rows):
    return len({r.get("merge_key","") for r in rows if r.get("merge_key")})

# ---------------------------
# 2) 커버리지 매트릭스 & QC
# ---------------------------
all_keys = set()
for rows in (rep_xml, rep_html, rep_ocr, rep_pdf):
    all_keys |= {r.get("merge_key","") for r in rows if r.get("merge_key")}

stats = []
by_key = {k:{
    "has_xml":0,"has_html":0,"has_ocr":0,"has_pdf":0,
    "chars_xml":0,"chars_html":0,"chars_ocr":0,"chars_pdf":0
} for k in all_keys}

def add_rows(rows, src):
    for r in rows:
        mk = r.get("merge_key","")
        if not mk: continue
        ch = r.get("chars") or 0
        by_key[mk][f"has_{src}"] = 1
        by_key[mk][f"chars_{src}"] += int(ch)

add_rows(rep_xml,  "xml")
add_rows(rep_html, "html")
add_rows(rep_ocr,  "ocr")
add_rows(rep_pdf,  "pdf")

# CSV 저장
QC_CSV.parent.mkdir(parents=True, exist_ok=True)
with QC_CSV.open("w", encoding="utf-8") as f:
    f.write("merge_key,has_xml,has_html,has_ocr,has_pdf,chars_xml,chars_html,chars_ocr,chars_pdf\n")
    for mk in sorted(all_keys):
        row = by_key[mk]
        f.write(f"{mk},{row['has_xml']},{row['has_html']},{row['has_ocr']},{row['has_pdf']},{row['chars_xml']},{row['chars_html']},{row['chars_ocr']},{row['chars_pdf']}\n")

# 텍스트 리포트
lines=[]
lines.append("[QC] rep_* 요약")
lines.append(f"- rep_xml:  rows={len(rep_xml)}, uniq_merge_keys={uniq_keys(rep_xml)}")
lines.append(f"- rep_html: rows={len(rep_html)}, uniq_merge_keys={uniq_keys(rep_html)}")
lines.append(f"- rep_ocr:  rows={len(rep_ocr)}, uniq_merge_keys={uniq_keys(rep_ocr)}")
lines.append(f"- rep_pdf:  rows={len(rep_pdf)}, uniq_merge_keys={uniq_keys(rep_pdf)}")
lines.append("")
# 소스 보유 조합 카운트
combo_counter = defaultdict(int)
for mk, row in by_key.items():
    combo = tuple(int(row[f"has_{s}"]) for s in ("xml","html","ocr","pdf"))
    combo_counter[combo]+=1
lines.append("[QC] 소스 보유 조합( has_xml,has_html,has_ocr,has_pdf ): count")
for combo, cnt in sorted(combo_counter.items(), key=lambda x:(-x[1],x[0])):
    lines.append(f" - {combo}: {cnt}")
lines.append("")

# PDF 키 이상 감지
pdf_uniq = uniq_keys(rep_pdf)
if pdf_uniq <= 3:
    lines.append(f"[경고] rep_pdf.merge_key 가 {pdf_uniq}개뿐입니다. 문서별 키가 제대로 분리되지 않은 상태로 보입니다.")
else:
    lines.append("[OK] rep_pdf.merge_key 가 문서 단위로 분리되어 보입니다.")
QC_TXT.write_text("\n".join(lines), encoding="utf-8")

print("[QC] 보고서 저장:", QC_TXT)
print("[QC] 커버리지 CSV:", QC_CSV)

# ---------------------------
# 3) rep_pdf 보정 (가능한 경우)
# ---------------------------
# uniq_merge_keys가 너무 작으면 개별 레코드의 doc_id/경로에서 다시 유추
rekeyed = []
if pdf_uniq <= 3 and rep_pdf:
    for r in rep_pdf:
        did = (r.get("doc_id") or r.get("doc") or "").strip()
        if not did:
            # 경로/메타에서 doc_id 후보 찾기
            path_hint = any_field(r, ["source_paths_sample","source_path","pdf_path","page_path","image_path"])
            did = derive_docid_from_path(path_hint)
        if not did:
            # 그래도 없으면 page_id 등 다른 힌트라도
            did = any_field(r, ["page_id","pdf_name","pdf_basename"]) or "_unknown"
        mk = ensure_merge_key(did)
        r2 = dict(r)
        r2["doc_id"] = did
        r2["merge_key"] = mk
        rekeyed.append(r2)

    write_jsonl(rekeyed, REP_PDF_FIX)
    print("[FIX] rep_pdf.fixed.jsonl 작성 완료:", REP_PDF_FIX)
    # 보정 후 상태 출력
    print(" - rows:", len(rekeyed), "uniq_merge_keys:", len({x["merge_key"] for x in rekeyed}))
else:
    print("[FIX] rep_pdf 보정 불필요 또는 입력 없음. 기존 파일을 사용하세요.")


# #### 4. 최종 병합 산출물 생성 (docs_merged.jsonl)

# - **키:** `merge_key` (문서당 1행)
# - **핵심 필드(예시):**
#     - 식별: `merge_key`, `canonical_doc_id`, `doc_id_variants`
#     - 소스 존재: `has_xml/html/pdf/ocr/tables`
#     - 텍스트: `text_xml`, `text_html`, `text_pdf`, `text_ocr_html`, `text_ocr_xml`, `text_ocr_combined`, `text_all`
#     - 표: `tables`(리스트: `csv_path, page, n_rows, n_cols, area_ratio, ruling_density…`)
#     - 통계: 각 소스 `chars_*`, `image_count`, `page_count_pdf`, `total_chars`
#     - 경로 샘플: `source_paths_sample`
# - **병합 정책(우선순위 & 중복 제거):**
#     - 우선순위 제안: **XML > HTML > PDF(cleaned) > OCR**
#         
#         (문장/라인 단위 정규화 후 중복 라인은 1회만, 원문 흐름 우선)
#         
#     - 표는 **본문과 분리 유지**(검색/QA에서 강점).

# 아래 스크립트는 `rep_xml.jsonl / rep_html.jsonl / rep_ocr.jsonl / rep_pdf.fixed.jsonl(있으면)`을 읽어 **merge_key** 단위로 통합.
# 
# - 소스별 텍스트(`text_xml / text_html / text_ocr_* / text_pdf`)를 모으고
# - 간단 라인 중복 제거 후 `text_merged`를 만들고
# - 소스/문자수/이미지 수 등의 요약 통계를 포함함
# - 결과: `/home/spai0308/data/processed/docs_merged.jsonl` (+ 요약 리포트)

# In[215]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re
from collections import defaultdict, Counter

BASE        = Path("/home/spai0308/data")
INTERIM     = BASE / "interim"
PROCESSED   = BASE / "processed"

# 입력 rep_* (PDF는 fixed가 있으면 우선 사용)
REP_XML = INTERIM / "rep_xml.jsonl"
REP_HTML = INTERIM / "rep_html.jsonl"
REP_OCR = INTERIM / "rep_ocr.jsonl"
REP_PDF_FIXED = INTERIM / "rep_pdf.fixed.jsonl"
REP_PDF_FALLBACK = INTERIM / "rep_pdf.with_mergekey.jsonl"

# 출력
OUT_JSONL = PROCESSED / "docs_merged.jsonl"
SUMMARY_TXT = INTERIM / "docs_merge_summary.txt"

# ---------------- helpers ----------------
def load_jsonl(p: Path):
    rows=[]
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def pick_first(rec, keys, default=""):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default

def get_paths_any(rec):
    for k in ("source_paths","paths","xml_paths","image_paths","source_paths_sample","pdf_paths","page_paths"):
        v = rec.get(k)
        if isinstance(v, list) and v:
            return v
    p = rec.get("source_path")
    if isinstance(p, str) and p:
        return [p]
    return []

def textlen(s): return len(s) if isinstance(s, str) else 0

def dedup_lines(*texts, min_len=2):
    """여러 텍스트 블록을 순서대로 합치되, 라인 단위로 중복 제거"""
    seen=set()
    out=[]
    for t in texts:
        if not t: continue
        # 줄 통일
        t = t.replace("\r\n","\n").replace("\r","\n")
        for ln in t.split("\n"):
            s = ln.strip()
            if len(s) < min_len: 
                continue
            key = s  # 그대로 키(필요시 소문자/공백정규화 가능)
            if key in seen: 
                continue
            seen.add(key)
            out.append(s)
    return "\n".join(out)

# ---------------- load reps ----------------
rep_xml  = load_jsonl(REP_XML)
rep_html = load_jsonl(REP_HTML)
rep_ocr  = load_jsonl(REP_OCR)
rep_pdf  = load_jsonl(REP_PDF_FIXED if REP_PDF_FIXED.exists() else REP_PDF_FALLBACK)

# 인덱스: merge_key -> row (여러 개면 첫 것/가장 긴 텍스트 우선)
def index_best_by_key(rows, source_tag):
    idx = {}
    for r in rows:
        mk = r.get("merge_key","")
        if not mk: continue
        # 가장 텍스트가 긴 레코드를 대표로 (보수적으로)
        cur = idx.get(mk)
        cand_text = pick_first(r, ["text","plain_text","content","combined_text","text_combined","page_text"], "")
        if (cur is None) or (textlen(cand_text) > textlen(pick_first(cur, ["text","plain_text","content","combined_text","text_combined","page_text"], ""))):
            r["_src_tag"]=source_tag
            idx[mk]=r
    return idx

ix_xml  = index_best_by_key(rep_xml,  "xml")
ix_html = index_best_by_key(rep_html, "html")
# OCR은 doc 단위 집계(이미 합본일 가능성 높음)
ix_ocr  = index_best_by_key(rep_ocr,  "ocr")
# PDF는 페이지/스냅샷 단위가 섞일 수 있으니 merge 시 따로 모음
pdf_by_key = defaultdict(list)
for r in rep_pdf:
    mk = r.get("merge_key","")
    if not mk: continue
    r["_src_tag"]="pdf"
    pdf_by_key[mk].append(r)

all_keys = set(ix_xml) | set(ix_html) | set(ix_ocr) | set(pdf_by_key)

# ---------------- merge per key ----------------
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
wrote = 0
stats_sources = Counter()
chars_total = 0

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for mk in sorted(all_keys):
        # 소스별 레코드
        rx = ix_xml.get(mk)
        rh = ix_html.get(mk)
        ro = ix_ocr.get(mk)
        rp_list = pdf_by_key.get(mk, [])

        # 대표 doc_id: xml > html > ocr > pdf(최장 텍스트 기준)
        doc_id = None
        for r in (rx, rh, ro):
            if r and r.get("doc_id"): 
                doc_id = r["doc_id"]; break
        if not doc_id and rp_list:
            # pdf 쪽에서 가장 길게 보이는 doc_id
            best = max(rp_list, key=lambda r: textlen(pick_first(r, ["text","page_text","content"], "")))
            doc_id = best.get("doc_id") or pick_first(best, ["pdf_name","pdf_basename"], mk)

        # 소스별 텍스트 추출
        text_xml  = pick_first(rx or {}, ["text","xml_text","content"], "")
        text_html = pick_first(rh or {}, ["text","plain_text","content"], "")
        # OCR은 구조적으로 나뉘어 있을 수 있어 폭넓게 시도
        text_ocr_combined = pick_first(ro or {}, ["combined","text_combined","ocr_combined","text"], "")
        if not text_ocr_combined:
            # 각 소스별 OCR 텍스트도 따로 시도
            part_html = pick_first(ro or {}, ["html_text","ocr_html_text"], "")
            part_xml  = pick_first(ro or {}, ["xml_text","ocr_xml_text"], "")
            text_ocr_combined = "\n".join([t for t in [part_html, part_xml] if t])

        # PDF는 목록을 모아 페이지순/파일명순으로 정렬 후 연결
        text_pdf = ""
        if rp_list:
            # 정렬 힌트: page_no, page, idx, 파일명 등
            def page_key(r):
                for k in ("page_no","page","page_index","idx"):
                    v = r.get(k)
                    if isinstance(v, int): return (0, v)
                    if isinstance(v, str) and v.isdigit(): return (0, int(v))
                # 파일명 fallback
                p = pick_first(r, ["page_path","image_path","source_path"], "")
                return (1, p)
            rp_list_sorted = sorted(rp_list, key=page_key)
            chunks=[]
            for r in rp_list_sorted:
                t = pick_first(r, ["text","page_text","content"], "")
                if t: chunks.append(t)
            text_pdf = "\n".join(chunks)

        # 최종 합본(라인 중복 제거)
        text_merged = dedup_lines(text_html, text_xml, text_ocr_combined, text_pdf, min_len=2)

        # 통계
        has_xml  = 1 if rx else 0
        has_html = 1 if rh else 0
        has_ocr  = 1 if ro else 0
        has_pdf  = 1 if rp_list else 0
        stats_sources[(has_xml,has_html,has_ocr,has_pdf)] += 1

        chars = {
            "xml":  textlen(text_xml),
            "html": textlen(text_html),
            "ocr":  textlen(text_ocr_combined),
            "pdf":  textlen(text_pdf),
            "merged": textlen(text_merged),
        }
        chars_total += chars["merged"]

        # OCR 이미지 수 힌트
        n_img_html = (ro or {}).get("n_images_html") or (ro or {}).get("imgs_html") or 0
        n_img_xml  = (ro or {}).get("n_images_xml")  or (ro or {}).get("imgs_xml")  or 0
        n_img_tot  = (ro or {}).get("n_images_total") or (n_img_html or 0) + (n_img_xml or 0)

        # 샘플 경로
        paths = []
        for r in (rx, rh, ro):
            if r: paths += get_paths_any(r)
        if rp_list:
            for r in rp_list[:3]:
                paths += get_paths_any(r)
        paths = list(dict.fromkeys(paths))[:5]  # 최대 5개 샘플

        rec = {
            "merge_key": mk,
            "doc_id": doc_id or mk,
            "sources": {
                "xml": has_xml, "html": has_html, "ocr": has_ocr, "pdf": has_pdf
            },
            "chars": chars,
            "counts": {
                "ocr_images_total": n_img_tot,
                "ocr_images_html": n_img_html,
                "ocr_images_xml": n_img_xml,
                "pdf_records": len(rp_list)
            },
            "texts": {
                "xml": text_xml,
                "html": text_html,
                "ocr": text_ocr_combined,
                "pdf": text_pdf,
                "merged": text_merged
            },
            "source_paths_sample": paths
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        wrote += 1

# 요약 리포트 쓰기
SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
lines=[]
lines.append(f"[MERGE] wrote lines: {wrote} → {OUT_JSONL}")
lines.append(f"[MERGE] total merged chars: {chars_total:,}")
lines.append("[MERGE] source presence combos (has_xml,has_html,has_ocr,has_pdf): count")
for combo,count in sorted(stats_sources.items(), key=lambda x:(-x[1], x[0])):
    lines.append(f" - {combo}: {count}")
SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))


# #### 검증 리포트(머지 후)

# - 최종 문서수(유니온), 소스 커버리지 매트릭스, `only_in_*` 잔여 여부, 충돌 해소 요약.
# - 샘플 스팟 QA 목록(랜덤 N + 상/하위 N).

# [**최종 문서 수(유니온)**]
# 
# - 소스 커버리지 매트릭스( ``has_xml``, ``has_html``, ``has_ocr``, ``has_pdf`` )
# - only_in_* 잔여(= 한 소스만 가진 문서) 통계 + 샘플
# - 조인 충돌 해소 요약(``joinkey_fixes.json`` 기반)
# - 스팟 QA 샘플(랜덤 N + 상/하위 N) 을 ``텍스트``/``CSV``/``JSONL``로 저장

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, random, csv
from collections import Counter, defaultdict

# ==== 경로 ====
BASE = Path("/home/spai0308/data")
INTERIM = BASE / "interim"
PROCESSED = BASE / "processed"

MERGED = PROCESSED / "docs_merged.jsonl"               # 머지 산출물
FIXES  = INTERIM / "joinkey_fixes.json"                # 조인 충돌 해소 기록

OUT_TXT = INTERIM / "verify_after_merge_report.txt"    # 종합 리포트
OUT_CSV = INTERIM / "verify_after_merge_coverage.csv"  # 커버리지 매트릭스 전개
OUT_SAMPLES = INTERIM / "verify_after_merge_spot_samples.jsonl"  # 스팟 QA 샘플

# ==== 유틸 ====
def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def coalesce_bool(x):
    # 일부 파이프라인이 0/1, "true"/"false" 섞일 수 있어 보정
    if isinstance(x, bool): return x
    if isinstance(x, int): return bool(x)
    if isinstance(x, str): return x.strip().lower() in ("1","true","yes","y","t")
    return False

def get_flags(rec):
    """
    문서 레코드에서 has_xml/html/ocr/pdf 플래그를 탄탄하게 추출.
    - 1순위: has_xml / has_html / has_ocr / has_pdf 필드
    - 2순위: sources (list | dict) 내 키
    - 3순위: texts(dict)의 키 존재 여부
    - 4순위: 개별 텍스트 키 존재(xml_text/html_text/ocr_text/pdf_text 등)
    """
    # 1) 직접 필드
    hx = rec.get("has_xml");  hh = rec.get("has_html")
    ho = rec.get("has_ocr");  hp = rec.get("has_pdf")
    # 2) sources
    sources = rec.get("sources")
    if sources:
        if isinstance(sources, dict):
            hx = coalesce_bool(hx) or coalesce_bool(sources.get("xml"))
            hh = coalesce_bool(hh) or coalesce_bool(sources.get("html"))
            ho = coalesce_bool(ho) or coalesce_bool(sources.get("ocr"))
            hp = coalesce_bool(hp) or coalesce_bool(sources.get("pdf"))
        elif isinstance(sources, list):
            bag = set(sources)
            hx = coalesce_bool(hx) or ("xml" in bag)
            hh = coalesce_bool(hh) or ("html" in bag)
            ho = coalesce_bool(ho) or ("ocr" in bag)
            hp = coalesce_bool(hp) or ("pdf" in bag)
    # 3) texts dict
    texts = rec.get("texts")
    if isinstance(texts, dict):
        keys = set(texts.keys())
        hx = coalesce_bool(hx) or ("xml" in keys)
        hh = coalesce_bool(hh) or ("html" in keys)
        ho = coalesce_bool(ho) or ("ocr" in keys)
        hp = coalesce_bool(hp) or ("pdf" in keys or "tables" in keys)
    # 4) 개별 텍스트 키
    for k in rec.keys():
        lk = k.lower()
        if "xml"  in lk: hx = coalesce_bool(hx) or True
        if "html" in lk: hh = coalesce_bool(hh) or True
        if "ocr"  in lk: ho = coalesce_bool(ho) or True
        if "pdf"  in lk or "table" in lk: hp = coalesce_bool(hp) or hp

    return (bool(hx), bool(hh), bool(ho), bool(hp))

def get_char_count(rec):
    # 선호: 미리 계산된 total/merged chars
    for k in ("merged_chars","total_chars","chars"):
        if isinstance(rec.get(k), int): return rec[k]
    # 텍스트 본문에서 계산
    total = 0
    # merged/plain 본문
    for k in ("merged_text","text","plain_text"):
        v = rec.get(k)
        if isinstance(v, str):
            total = max(total, len(v))
    # texts dict 합산(중복 허용 X, 가장 큰 값 사용)
    texts = rec.get("texts")
    if isinstance(texts, dict):
        s = 0
        for v in texts.values():
            if isinstance(v, str): s += len(v)
            elif isinstance(v, dict) and isinstance(v.get("text"), str):
                s += len(v["text"])
        total = max(total, s)
    return total

def safe_docid(rec):
    return rec.get("doc_id") or rec.get("raw_doc_id") or rec.get("title") or ""

def safe_key(rec):
    return rec.get("merge_key") or rec.get("join_key") or ""

# ==== 데이터 로드 ====
docs = load_jsonl(MERGED)
fixes = []
if FIXES.exists():
    with FIXES.open("r", encoding="utf-8") as f:
        fixes = json.load(f)

# ==== 1) 기본 통계 ====
n_docs = len(docs)
flags = [get_flags(r) for r in docs]
chars = [get_char_count(r) for r in docs]
by_combo = Counter(flags)

# only_in_* (정확히 하나만 True)
only_one = defaultdict(list)
for r, flg, ch in zip(docs, flags, chars):
    s = sum(flg)
    if s == 1:
        label = ("xml","html","ocr","pdf")[flg.index(True)]
        only_one[label].append((safe_key(r), safe_docid(r), ch))

# ==== 2) 충돌 해소 요약 ====
# joinkey_fixes.json 에서 changed=True 인 것만 그룹핑
collision_groups = defaultdict(list)  # join_key -> [ {doc_id, merge_key} ... ]
for e in fixes:
    if e.get("changed"):
        collision_groups[e["join_key"]].append({"doc_id": e["doc_id"], "merge_key": e["merge_key"]})

n_collision_groups = len(collision_groups)

# ==== 3) 스팟 QA 샘플(랜덤 N + 상/하위 N) ====
RAND_N = 10
TOP_N  = 5
BOT_N  = 5
rng = random.Random(42)

indexed = []
for r, ch, flg in zip(docs, chars, flags):
    indexed.append({
        "merge_key": safe_key(r),
        "doc_id": safe_docid(r),
        "chars": ch,
        "flags": {"xml":flg[0],"html":flg[1],"ocr":flg[2],"pdf":flg[3]},
    })

# 상/하위
top = sorted(indexed, key=lambda x: x["chars"], reverse=True)[:TOP_N]
bot = sorted(indexed, key=lambda x: x["chars"])[:BOT_N]
# 랜덤
rand_pool = [x for x in indexed if x not in top and x not in bot]
rand = rng.sample(rand_pool, min(RAND_N, len(rand_pool)))

# ==== 4) 저장 ====
# 4-1) 텍스트 리포트
lines = []
lines.append("=== 검증 리포트 (머지 후) ===")
lines.append(f"- 최종 문서 수(유니온): {n_docs}")
lines.append("")
lines.append("[소스 커버리지 매트릭스] (has_xml, has_html, has_ocr, has_pdf): count")
for combo, cnt in sorted(by_combo.items()):
    lines.append(f" - {combo}: {cnt}")
lines.append("")
if only_one:
    lines.append("[only_in_* 잔여] (한 소스만 보유)")
    for k in ("xml","html","ocr","pdf"):
        lst = only_one.get(k, [])
        lines.append(f" - only_in_{k}: {len(lst)}")
        # 샘플 최대 5개
        for mk, did, ch in lst[:5]:
            lines.append(f"    · {mk} | {did} | chars={ch}")
    lines.append("")
else:
    lines.append("[only_in_* 잔여] 없음")
    lines.append("")

lines.append(f"[조인 충돌 해소 요약] groups={n_collision_groups}")
for jk, arr in list(collision_groups.items())[:5]:  # 너무 길면 5개만 미리보기
    lines.append(f" - join_key: {jk}")
    for e in arr[:5]:
        lines.append(f"    · doc_id='{e['doc_id']}' → merge_key='{e['merge_key']}'")
if n_collision_groups > 5:
    lines.append(f"   ... (+{n_collision_groups-5} groups more)")
lines.append("")

lines.append("[스팟 QA 샘플] (랜덤 10)")
for r in rand:
    lines.append(f" - {r['merge_key']} | {r['doc_id']} | chars={r['chars']} | flags={r['flags']}")
lines.append("")
lines.append("[스팟 QA 샘플] (상위 5 by chars)")
for r in top:
    lines.append(f" - {r['merge_key']} | {r['doc_id']} | chars={r['chars']} | flags={r['flags']}")
lines.append("")
lines.append("[스팟 QA 샘플] (하위 5 by chars)")
for r in bot:
    lines.append(f" - {r['merge_key']} | {r['doc_id']} | chars={r['chars']} | flags={r['flags']}")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

# 4-2) 커버리지 CSV 전개
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["merge_key","doc_id","has_xml","has_html","has_ocr","has_pdf","chars"])
    for r, flg, ch in zip(docs, flags, chars):
        w.writerow([safe_key(r), safe_docid(r), int(flg[0]), int(flg[1]), int(flg[2]), int(flg[3]), ch])

# 4-3) 스팟샘플 JSONL(랜덤+상하위)
with OUT_SAMPLES.open("w", encoding="utf-8") as f:
    for x in rand + top + bot:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("[QC] 리포트 저장:", OUT_TXT)
print("[QC] 커버리지 CSV:", OUT_CSV)
print("[QC] 스팟샘플 JSONL:", OUT_SAMPLES)


# # 2. 임베딩용 청크 생성 (``docs_chunks.jsonl``)

# - `text_all` 기준으로 1200~1500자, 200자 오버랩 권장.
# - 청크 메타: `merge_key, chunk_id, chunk_index, source_breakdown(비율), approx_page_ranges(가능 시)`.

# - **섹션 우선 + 800/200 슬라이딩**

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re
from collections import defaultdict, Counter

BASE = Path("/home/spai0308/data")
PROCESSED = BASE / "processed"
INTERIM = BASE / "interim"
PROCESSED.mkdir(parents=True, exist_ok=True)

SRC = PROCESSED / "docs_merged.jsonl"
OUT_JSONL = PROCESSED / "chunks_xmlhtml_800_200.jsonl"
OUT_MANIFEST = INTERIM / "chunks_manifest.csv"
OUT_STATS = INTERIM / "chunks_stats.txt"

MAX_CHARS = 800
OVERLAP   = 200

def load_jsonl(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def coalesce_texts(rec):
    """
    섹션 인지 우선(가능하면 XML/HTML 텍스트 사용), 없으면 merged_text→ocr→pdf 순.
    반환: (merged_basis_text, sources_used:set)
    """
    texts = rec.get("texts") or {}
    xml = texts.get("xml") if isinstance(texts.get("xml"), str) else None
    html = texts.get("html") if isinstance(texts.get("html"), str) else None
    ocr = texts.get("ocr") if isinstance(texts.get("ocr"), str) else None
    pdf = texts.get("pdf") if isinstance(texts.get("pdf"), str) else None
    merged = rec.get("merged_text")

    srcs = []
    buf = []

    # 섹션/구조가 살아있을 가능성이 큰 순서
    if xml and xml.strip():
        buf.append(xml.strip()); srcs.append("xml")
    if html and html.strip():
        buf.append(html.strip()); srcs.append("html")

    base = "\n\n".join(buf).strip()
    if not base:
        if isinstance(merged, str) and merged.strip():
            base = merged.strip(); srcs.append("merged")
        elif ocr and ocr.strip():
            base = ocr.strip(); srcs.append("ocr")
        elif pdf and pdf.strip():
            base = pdf.strip(); srcs.append("pdf")
        else:
            base = ""
    return base, set(srcs)

# 헤딩/섹션 경계 감지 정규식들
HEADING_PATTS = [
    r"(?m)^\s*#{1,6}\s+.+$",                   # Markdown 헤딩
    r"(?m)^\s*={2,}\s*$",                      # ===== 구분선
    r"(?m)^\s*-{3,}\s*$",                      # ----- 구분선
    r"(?m)^\s*\[\s*목차\s*\]\s*$",
    r"(?m)^\s*(제?\s*\d+\s*장)\s+.+$",         # 제1장 / 1장
    r"(?m)^\s*\d+\.\d+(\.\d+)*\s+.+$",         # 1. / 1.1. / 1.1.1.
    r"(?m)^\s*[IVXⅰ-ⅴⅠ-Ⅴ]+\.\s+.+$",          # 로마자 I. II. …
    r"(?m)^\s*【.+?】\s*$",                     # 【섹션】
]

def split_by_sections(text: str):
    if not text: return []
    # 경계 후보 찾기
    idxs = set([0])
    for patt in HEADING_PATTS:
        for m in re.finditer(patt, text):
            idxs.add(m.start())
    idxs = sorted(list(idxs))
    # 큰 덩어리 섹션으로 자르기
    sections = []
    for i, start in enumerate(idxs):
        end = idxs[i+1] if i+1 < len(idxs) else len(text)
        seg = text[start:end].strip()
        if seg: sections.append(seg)
    # 섹션이 너무 적으면(=실패) 통으로 반환
    if len(sections) <= 1:
        return [text.strip()]
    return sections

def slide_chunks(s: str, max_chars=800, overlap=200):
    if len(s) <= max_chars: return [s]
    chunks=[]
    i=0
    while i < len(s):
        chunk = s[i:i+max_chars]
        chunks.append(chunk)
        if i+max_chars >= len(s): break
        i = max(0, i + max_chars - overlap)
    return chunks

def chunk_doc(rec):
    key = rec.get("merge_key") or rec.get("join_key") or ""
    doc = rec.get("doc_id") or ""
    base, used = coalesce_texts(rec)
    if not base: return []

    sections = split_by_sections(base)
    out=[]
    seq=0
    for sec in sections:
        for ch in slide_chunks(sec, MAX_CHARS, OVERLAP):
            out.append({
                "merge_key": key,
                "doc_id": doc,
                "chunk_id": f"{key}__{seq:04d}",
                "chunk_index": seq,
                "text": ch,
                "len": len(ch),
                "sources_used": sorted(list(used)),
            })
            seq+=1
    return out

# ===== run =====
docs = load_jsonl(SRC)
all_chunks=[]
per_doc_counter = Counter()
for r in docs:
    cs = chunk_doc(r)
    all_chunks.extend(cs)
    if cs:
        per_doc_counter[r.get("merge_key") or ""] += len(cs)

# 저장
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSONL.open("w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# 매니페스트 & 통계
with OUT_MANIFEST.open("w", encoding="utf-8") as f:
    f.write("merge_key,doc_id,chunks,total_chars\n")
    stats = defaultdict(lambda: {"doc_id":"", "chunks":0, "chars":0})
    for c in all_chunks:
        k = c["merge_key"]
        stats[k]["doc_id"] = c["doc_id"]
        stats[k]["chunks"] += 1
        stats[k]["chars"]  += c["len"]
    for k, v in stats.items():
        f.write(f"{k},{v['doc_id']},{v['chunks']},{v['chars']}\n")

lines=[]
lines.append(f"[CHUNK] wrote: {len(all_chunks)} → {OUT_JSONL}")
lines.append(f"[CHUNK] docs with chunks: {len(per_doc_counter)} / total_docs: {len(docs)}")
top = per_doc_counter.most_common(10)
if top:
    lines.append("[CHUNK] top docs by #chunks")
    for k,cnt in top:
        lines.append(f" - {k}: {cnt}")
OUT_STATS.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("[CHUNK] manifest:", OUT_MANIFEST)
print("[CHUNK] stats:", OUT_STATS)


# # 3.  임베딩 & 인덱스(Faiss)

# - ``backend="openai"``: ``OpenAI text-embedding-3-small`` (1536차원)
# - ``backend="sbert"`` : 로컬/허깅페이스(ex. ``jhgan/ko-sroberta-multitask``, ``upskyy``/``bge-m3-korean``)

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, time, math
from collections import defaultdict

# ===== 설정 =====
BASE = Path("/home/spai0308/data")
PROCESSED = BASE / "processed"
INDEXDIR = PROCESSED / "index"
INDEXDIR.mkdir(parents=True, exist_ok=True)

CHUNKS = PROCESSED / "chunks_xmlhtml_800_200.jsonl"
META_OUT = INDEXDIR / "chunks_meta.jsonl"
IDX_PATH = INDEXDIR / "faiss.index"
IDX_INFO = INDEXDIR / "faiss_info.txt"

BACKEND = "openai"   # "openai" 또는 "sbert"

# OpenAI 설정
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_BATCH = 128

# SBERT 설정
SBERT_MODEL = "jhgan/ko-sroberta-multitask"  # 또는 "upskyy/bge-m3-korean"
SBERT_BATCH = 64

L2_NORMALIZE = True  # 코사인 유사도용 정규화

def load_jsonl(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def l2_normalize(v):
    import numpy as np
    x = np.asarray(v, dtype="float32")
    n = np.linalg.norm(x)
    return (x / n) if n>0 else x

# ===== 임베딩 백엔드 =====
def embed_openai(texts):
    # pip install openai>=1.0.0
    from openai import OpenAI
    client = OpenAI()
    # 배치 호출
    vecs=[]
    for i in range(0, len(texts), OPENAI_BATCH):
        batch = texts[i:i+OPENAI_BATCH]
        resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
        time.sleep(0.1)
    return vecs

def embed_sbert(texts):
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(SBERT_MODEL)
    vecs=[]
    for i in range(0, len(texts), SBERT_BATCH):
        batch = texts[i:i+SBERT_BATCH]
        v = model.encode(batch, show_progress_bar=False, normalize_embeddings=False)
        vecs.extend(v.tolist())
    return vecs

def get_vectors(texts):
    if BACKEND == "openai":
        vecs = embed_openai(texts)
    else:
        vecs = embed_sbert(texts)
    if L2_NORMALIZE:
        vecs = [l2_normalize(v).tolist() for v in vecs]
    return vecs

# ===== 실행 =====
chunks = load_jsonl(CHUNKS)
texts  = [c["text"] for c in chunks]
ids    = [c["chunk_id"] for c in chunks]

# 1) 벡터화
vecs = get_vectors(texts)
dim = len(vecs[0]) if vecs else 0

# 2) FAISS 인덱스
import numpy as np, faiss
xb = np.array(vecs, dtype="float32")
index = faiss.IndexFlatIP(dim)  # 코사인 = 내적 (L2 정규화 전제)
index.add(xb)
faiss.write_index(index, str(IDX_PATH))

# 3) 메타 저장(검색 결과 매핑용)
with META_OUT.open("w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps({
            "chunk_id": c["chunk_id"],
            "merge_key": c["merge_key"],
            "doc_id": c["doc_id"],
            "len": c["len"],
            "sources_used": c["sources_used"],
        }, ensure_ascii=False) + "\n")

# 4) 정보 파일
lines=[]
lines.append(f"[INDEX] backend={BACKEND}")
lines.append(f"[INDEX] dim={dim}, vectors={len(vecs)}")
lines.append(f"[INDEX] faiss={IDX_PATH}")
lines.append(f"[INDEX] meta={META_OUT}")
IDX_INFO.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))


# **[생성된 아티팩트 핵심]**
# 
# 
# - `processed/chunks_xmlhtml_800_200.jsonl` : 청킹 결과(총 8,271 청크 예상)
# - `processed/index/faiss.index` : FAISS 벡터 인덱스
# - `processed/index/chunks_meta.jsonl` : 청크 메타(merge_key, doc_id 등)
# - `processed/index/faiss_info.txt` : 방금 출력된 요약(backend=openai, dim=1536, vectors=8271)
# 
# ------------------------
# 
# 

# **빠른 QC 체크리스트**:
# 
# 1. 개수 일치: `vectors=8271` ↔ `chunks_xmlhtml_800_200.jsonl` 라인 수 ↔ `chunks_meta.jsonl` 라인 수가 모두 같아야 함.
# 2. 인덱스 크기: `faiss.read_index(...).ntotal == 8271`
# 3. 스모크 검색: “전자조달”, “ISMP”, “지능정보화전략계획” 같은 쿼리로 3~5건 조회해 상식적인 문서가 뜨는지 확인.

# In[ ]:


# 로컬 검색 테스트 스니펫

# 간단 검색기: 쿼리→임베딩→상위 k개
from pathlib import Path
import json
import numpy as np, faiss

INDEXDIR = Path("/home/spai0308/data/processed/index")
IDX_PATH = INDEXDIR / "faiss.index"
META = INDEXDIR / "chunks_meta.jsonl"

BACKEND = "openai"  # "openai" or "sbert"
OPENAI_MODEL = "text-embedding-3-small"
SBERT_MODEL = "jhgan/ko-sroberta-multitask"

def load_meta(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    by_chunk = {r["chunk_id"]: r for r in rows}
    return rows, by_chunk

def embed_query(q: str):
    if BACKEND=="openai":
        from openai import OpenAI
        from numpy.linalg import norm
        client = OpenAI()
        v = client.embeddings.create(model=OPENAI_MODEL, input=[q]).data[0].embedding
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL)
        v = model.encode([q], show_progress_bar=False)[0].tolist()
    v = np.asarray(v, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)  # L2 정규화
    return v

index = faiss.read_index(str(IDX_PATH))
rows, meta_by_id = load_meta(META)

def search(q, k=5):
    v = embed_query(q).reshape(1,-1).astype("float32")
    D, I = index.search(v, k)
    hits=[]
    for d, i in zip(D[0], I[0]):
        cid = rows[i]["chunk_id"] if i < len(rows) else None
        m = meta_by_id.get(cid, {})
        hits.append((float(d), cid, m.get("merge_key"), m.get("doc_id")))
    return hits

print(search("전자조달 시스템 고도화", 5))


# 
# 
# **[이후 점검 방법]**
# 
# - 점수대가 ``0.45~0.55``로 고만고만하면 **Top-k를 10~20으로 넉넉히** 뽑은 뒤 간단한 **재랭킹(키워드 가중/룰·BM25·reranker)** 붙이면 품질이 확 올라가며
# 
# 
# - 더 구체적 질의를 주면(예: “차세대 ERP + G/W + 클라우드 전환”) 점수 분산이 커져 상위가 더 또렷해짐
# 
# 
# - 특정 청크 내용을 확인하고 싶으면 `chunks_meta.jsonl`에서 해당 `chunk_id`를 찾아 원문 스니펫을 보면 됨
# 

# # 4. Issues

# ### 4-1 코사인 유사도 결과가 낮아 개선 작업 시작

# In[ ]:


# top-k 10으로 수정 퀵 테스트

# 로컬 검색 테스트 스니펫

# 간단 검색기: 쿼리→임베딩→상위 k개
from pathlib import Path
import json
import numpy as np, faiss

INDEXDIR = Path("/home/spai0308/data/processed/index")
IDX_PATH = INDEXDIR / "faiss.index"
META = INDEXDIR / "chunks_meta.jsonl"

BACKEND = "openai"  # "openai" or "sbert"
OPENAI_MODEL = "text-embedding-3-small"
SBERT_MODEL = "jhgan/ko-sroberta-multitask"

def load_meta(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    by_chunk = {r["chunk_id"]: r for r in rows}
    return rows, by_chunk

def embed_query(q: str):
    if BACKEND=="openai":
        from openai import OpenAI
        from numpy.linalg import norm
        client = OpenAI()
        v = client.embeddings.create(model=OPENAI_MODEL, input=[q]).data[0].embedding
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL)
        v = model.encode([q], show_progress_bar=False)[0].tolist()
    v = np.asarray(v, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)  # L2 정규화
    return v

index = faiss.read_index(str(IDX_PATH))
rows, meta_by_id = load_meta(META)

def search(q, k=10):
    v = embed_query(q).reshape(1,-1).astype("float32")
    D, I = index.search(v, k)
    hits=[]
    for d, i in zip(D[0], I[0]):
        cid = rows[i]["chunk_id"] if i < len(rows) else None
        m = meta_by_id.get(cid, {})
        hits.append((float(d), cid, m.get("merge_key"), m.get("doc_id")))
    return hits

print(search("평가 기준", 10))


# - 결론
#   - top-k의 문제는 아니다 --> 유사도가 0.5선에 계속 머물러 있음

# #### 확인 1: 청크 진단

# In[40]:


# 청크프리뷰

from pathlib import Path
import json, re
from collections import defaultdict

CHUNKS = Path("/home/spai0308/data/processed/chunks_xmlhtml_800_200.jsonl")

def iter_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

by_merge = defaultdict(list)
by_id = {}
for r in iter_jsonl(CHUNKS):
    cid = r.get("chunk_id") or r.get("id")
    txt = r.get("text") or r.get("content") or ""
    mk  = r.get("merge_key")
    r["_len"] = len(txt)
    by_id[cid] = r
    by_merge[mk].append(r)

def preview_doc(mk, n=5):
    rows = sorted(by_merge.get(mk, []), key=lambda x: x.get("chunk_idx", 0))
    print(f"[{mk}] chunks={len(rows)}  lens=[min={min(r['_len'] for r in rows) if rows else 0}, max={max(r['_len'] for r in rows) if rows else 0}]")
    for r in rows[:n]:
        txt = (r.get("text") or r.get("content") or "").strip().replace("\n"," ")
        head = txt[:200]
        print(f" - {r.get('chunk_id')} len={r['_len']} :: {head}")


# In[ ]:


# === 0) 경로/파일 존재 확인 + 로드 ===
from pathlib import Path
import json, re
from collections import defaultdict, Counter


CHUNKS = Path("/home/spai0308/data/processed/chunks_xmlhtml_800_200.jsonl")


assert CHUNKS.exists(), f"파일 없음: {CHUNKS}"


def iter_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

by_merge = defaultdict(list)
by_id = {}
line_cnt = 0

for r in iter_jsonl(CHUNKS):
    line_cnt += 1
    cid = r.get("chunk_id") or r.get("id")
    txt = r.get("text") or r.get("content") or r.get("body") or ""
    mk  = r.get("merge_key") or r.get("doc_key") or r.get("join_key")

    if not cid or not mk: 
        continue

    r["_len"] = len(txt)
    by_id[cid] = r
    by_merge[mk].append(r)

total_chunks = sum(len(v) for v in by_merge.values())
print(f"[LOAD] lines={line_cnt}  merge_keys={len(by_merge)}  chunks(collected)={total_chunks}")

# === 1) 유틸: 정렬키(청크 인덱스가 없으면 chunk_id의 '__0003' 숫자 사용) ===
def _chunk_order(r):
    if "chunk_idx" in r and isinstance(r["chunk_idx"], int):
        return r["chunk_idx"]
    m = re.search(r"__(\d{3,})$", r.get("chunk_id",""))
    return int(m.group(1)) if m else 0

# === 2) 프리뷰 / 노이즈 / 길이분포 함수 ===
def preview_doc(mk, n=5, head_chars=200):
    rows = sorted(by_merge.get(mk, []), key=_chunk_order)
    if not rows:
        print(f"[preview_doc] no chunks for {mk}")
        return
    lens = [r["_len"] for r in rows]
    print(f"[{mk}] chunks={len(rows)}  lens[min={min(lens)}, p50={sorted(lens)[len(lens)//2]}, max={max(lens)}]")
    for r in rows[:n]:
        txt = (r.get("text") or r.get("content") or r.get("body") or "").strip().replace("\n"," ")
        print(f" - {r.get('chunk_id')} len={r['_len']} :: {txt[:head_chars]}")

def top_repeated_lines(mk, topn=10, min_len=6):
    rows = by_merge.get(mk, [])
    if not rows:
        print(f"[top_repeated_lines] no chunks for {mk}")
        return
    lines=[]
    for r in rows:
        txt = (r.get("text") or r.get("content") or r.get("body") or "")
        for L in (x.strip() for x in txt.splitlines()):
            if len(L) >= min_len:
                lines.append(L)
    for s, n in Counter(lines).most_common(topn):
        print(f"{n}x  {s[:120]}")

def len_hist(mk):
    import numpy as np
    rows = by_merge.get(mk, [])
    if not rows:
        print(f"[len_hist] no chunks for {mk}")
        return
    lens = [r["_len"] for r in rows]
    q = lambda p: float(np.percentile(lens, p))
    print(f"count={len(lens)}  min={min(lens)}  p25={q(25):.0f}  p50={q(50):.0f}  p75={q(75):.0f}  max={max(lens)}")

# === 3) 샘플 키 보여주기 ===
sample_keys = list(by_merge.keys())[:8]
print("[SAMPLE MERGE_KEYS]", *sample_keys, sep="\n - ")

# === 4) 키워드로 키 찾기(부분문자열 매칭) ===
def find_keys(sub, limit=10):
    sub = sub.strip()
    hits = [k for k in by_merge.keys() if sub in k]
    print(f"[find_keys] '{sub}' -> {len(hits)} hits")
    for k in hits[:limit]:
        print(" -", k)
    return hits


# === 5) 바로 한 개 문서 프리뷰 실행 예시 ===
#    - 여기에 실제 보고 싶은 merge_key를 넣어서 호출하세요.
# preview_doc(sample_keys[0], n=5)
# 샘플 첫 문서를 대상으로 앞 5개 청크 내용 미리보기

# len_hist(sample_keys[0])
# 샘플 첫 문서의 청크 길이 분포 요약(최소/사분위/최대)

# top_repeated_lines(sample_keys[0], topn=10)
# 샘플 첫 문서에서 반복 줄 TOP10(헤더/푸터 후보)


# ## 4-2. ``docs_merged.jsonl`` 파일 text 41개 누락

# ### docs_merged.jsonl 구조 분석

# In[ ]:


from pathlib import Path
import json, collections, statistics

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

# 유틸 함수: JSONL 반복자
def iter_jsonl(p, limit=None):
    with p.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if not ln.strip(): 
                continue
            yield json.loads(ln)
            if limit and i+1 >= limit:
                break

# === 1) 파일 형식 및 기본 구조 ===
first_5 = list(iter_jsonl(FILE, limit=5))
print("샘플 5줄:", first_5)
print("="*100)

# === 2) 키(key) 이름, 값 타입 ===
key_counter = collections.Counter()
val_types = collections.defaultdict(set)
for obj in iter_jsonl(FILE, limit=500):   # 앞부분 500개만 스캔
    for k,v in obj.items():
        key_counter[k]+=1
        val_types[k].add(type(v).__name__)
print("키별 등장 횟수:", key_counter)
print("키별 값 타입:", {k:list(v) for k,v in val_types.items()})
print("="*100)

# === 3) 문서 단위 메타데이터 확인 ===
merge_keys, join_keys, doc_ids, reps = set(), set(), set(), collections.Counter()
for obj in iter_jsonl(FILE, limit=2000):  # 앞 2000개 샘플
    if "merge_key" in obj: merge_keys.add(obj["merge_key"])
    if "join_key" in obj: join_keys.add(obj["join_key"])
    if "doc_id" in obj: doc_ids.add(obj["doc_id"])
    for k in obj:
        if k.startswith("rep_") and obj[k]:
            reps[k]+=1
print("merge_key 예시:", list(merge_keys)[:5])
print("join_key 예시:", list(join_keys)[:5])
print("rep_* 분포:", reps)
print("="*100)

# === 4) 텍스트 / OCR 구조 ===
lens = collections.defaultdict(list)
for obj in iter_jsonl(FILE, limit=5000):
    for k in ["doc_text", "ocr_text", "ocr_image", "ocr_concat"]:
        if k in obj and obj[k]:
            lens[k].append(len(obj[k]))
summary = {k: dict(min=min(v), max=max(v), mean=int(statistics.mean(v))) 
           for k,v in lens.items() if v}
print("텍스트 길이 통계:", summary)
print("="*100)

# === 5) 병합 방식 / 충돌 확인 ===
print("merge_key 총 개수:", len(merge_keys))
print("="*100)
print("join_key 총 개수:", len(join_keys))
print("="*100)
print("doc_id 총 개수:", len(doc_ids))


# === 6) 품질 확인 ===
empty_texts = 0
total = 0
for obj in iter_jsonl(FILE):
    total+=1
    if not (obj.get("doc_text") or obj.get("ocr_text") or obj.get("ocr_image")):
        empty_texts+=1
print("총 문서 수:", total)
print("="*100)
print("빈 텍스트 비율:", round(empty_texts/total*100, 2), "%")


# In[ ]:


# docs_merged_jsonl 에서 merge_key, doc_id 개수 확인

from pathlib import Path
import json

# 파일 경로 (원하는 경로로 수정)
FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

merge_keys = set()
doc_ids = set()

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue
        # merge_key와 doc_id가 있으면 set에 추가
        if "merge_key" in obj and obj["merge_key"]:
            merge_keys.add(obj["merge_key"])
        if "doc_id" in obj and obj["doc_id"]:
            doc_ids.add(obj["doc_id"])

print("merge_key 개수:", len(merge_keys))
print("doc_id 개수:", len(doc_ids))


# In[ ]:


# merge_key, doc_id가 불일치 하는 경우 테스트

from pathlib import Path
import json
import re

# 파일 경로 (원하는 경로로 수정)
FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

def normalize(s: str) -> str:
    """
    문자열 정규화: 특수문자, 공백 제거
    """
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s)

problems = []
count = 0

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:ㅇ
            obj = json.loads(line)
        except:
            continue

        mk = obj.get("merge_key", "")
        did = obj.get("doc_id", "")

        norm_mk = normalize(mk)
        norm_did = normalize(did)

        # 이름이 비어있거나, 정규화 후 불일치하는 경우 기록
        if not norm_mk or not norm_did or norm_mk != norm_did:
            problems.append({
                "merge_key": mk,
                "doc_id": did,
                "norm_merge_key": norm_mk,
                "norm_doc_id": norm_did,
            })
        count += 1

print("총 문서 수:", count)
print("문제 발견 문서 수:", len(problems))

# 문제가 있는 문서 몇 개만 샘플로 출력
for p in problems[:5]:
    print(p)


# ### 문제 : 대소문자 차이, 한글 정규화(NFD/NFC)
# - 이 때문에 위 코드에서 불일치로 잡힘
# - 한글 자모(NFD 분해형)이라서 기존 정규식(가-힣)에서 전부 떨어져 나갔고 -> 그래서 빈 문자열이 됨
# - 아래 코드로 ``Unicode NFC 정규화 -> 소문자화 -> 특수문자 제거`` 순으로 처리하면 해결 하능

# In[ ]:


from pathlib import Path
import json, re, unicodedata

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

def normalize(s: str) -> str:
    """
    1) Unicode NFC 정규화 (분해형 한글 → 결합형으로 통일)
    2) 소문자화 (대소문자 무시 비교)
    3) 한글/영문/숫자만 남기기 (공백/특수문자 제거)
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)  # ← 중요!
    s = s.lower().strip()
    return re.sub(r"[^0-9a-z가-힣]", "", s)

total = 0
problems = []
unique_mk = set()
unique_did = set()
unique_mk_norm = set()
unique_did_norm = set()

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue

        mk  = obj.get("merge_key", "")
        did = obj.get("doc_id", "")

        mk_norm  = normalize(mk)
        did_norm = normalize(did)

        unique_mk.add(mk)
        unique_did.add(did)
        unique_mk_norm.add(mk_norm)
        unique_did_norm.add(did_norm)

        # 이름이 비었거나(정규화 후) 서로 다르면 문제로 기록
        if not mk_norm or not did_norm or mk_norm != did_norm:
            problems.append({
                "merge_key": mk,
                "doc_id": did,
                "norm_merge_key": mk_norm,
                "norm_doc_id": did_norm,
                "reason": (
                    "empty_after_normalize" if (not mk_norm or not did_norm)
                    else "mismatch_after_normalize"
                )
            })
        total += 1

print("총 문서 수:", total)
print("원본 기준 고유 merge_key 수:", len(unique_mk))
print("원본 기준 고유 doc_id 수:", len(unique_did))
print("정규화 기준 고유 merge_key 수:", len(unique_mk_norm))
print("정규화 기준 고유 doc_id 수:", len(unique_did_norm))
print("문제 발견 문서 수:", len(problems))

# 샘플 몇 개만 보기
for p in problems[:5]:
    print(p)


# **[실행 이후]**
# 
# - NFC 정규화: 경(분해형) → 경(결합형)으로 통일. 기존 정규식에 정상적으로 매칭됨.
# - 소문자화: bioin vs BioIN 같은 대소문자 차이 무시.
# - 정규식 필터: 비교 시 의미 없는 공백/특수문자 제거.

# In[ ]:


from pathlib import Path
import json, re, unicodedata
from collections import Counter, defaultdict

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

def normalize(s: str) -> str:
    # 한글 분해형을 결합형으로 통일 → 소문자 → 한글/영문/숫자만
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    return re.sub(r"[^0-9a-z가-힣]", "", s)

raw_keys = []
norm_keys = []
origins_by_norm = defaultdict(set)  # 정규화 키 → 원래 merge_key 집합

with FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue
        mk = obj.get("merge_key", "")
        raw_keys.append(mk)
        nk = normalize(mk)
        norm_keys.append(nk)
        origins_by_norm[nk].add(mk)

# 1) 원본 기준 중복
raw_counter = Counter(raw_keys)
raw_dups = {k:c for k,c in raw_counter.items() if c > 1}

# 2) 정규화 기준 중복(동일 의미로 보이는 충돌 포함)
norm_counter = Counter(norm_keys)
norm_dups = {k:c for k,c in norm_counter.items() if c > 1}
norm_empty = [k for k in norm_keys if not k]  # 정규화 후 빈 키

# 2-1) 정규화 키 하나에 서로 다른 원본 키가 매핑되는 충돌 목록
norm_collisions = {nk: sorted(list(origins))
                   for nk, origins in origins_by_norm.items()
                   if nk and len(origins) > 1}

print("총 레코드 수:", len(raw_keys))
print("고유 merge_key 수(원본):", len(raw_counter))
print("중복 merge_key 수(원본):", len(raw_dups))

print("고유 merge_key 수(정규화):", len(norm_counter))
print("중복 merge_key 수(정규화):", len(norm_dups))
print("정규화 후 빈 merge_key 개수:", len(norm_empty))

# 필요 시 상세 출력 (상위 몇 개만)
if raw_dups:
    print("\n[원본 중복 예시 TOP 10]")
    for k,c in raw_counter.most_common(10):
        if c > 1:
            print(f"- {k!r}: {c}회")

if norm_collisions:
    print("\n[정규화 기준 충돌(서로 다른 원본이 같은 정규화 키)]")
    for nk, originals in list(norm_collisions.items())[:10]:
        print(f"- norm='{nk}': originals={originals}")

if not raw_dups and not norm_dups and not norm_empty:
    print("\n✅ 모든 merge_key가 유일하며(원본/정규화 기준 모두), 빈 키도 없습니다.")


# [실행 이후]
# 
# - 원본 중복: 파일에 같은 merge_key가 그대로 여러 번 등장.
# - 정규화 중복: 표기 차이만 있는 키(대소문자/공백/특수문자/분해형 한글 등)가 사실상 같은 키로 취급될 때 충돌.
# - 정규화 충돌 목록: 한 정규화 키에 서로 다른 원본 키들이 매핑된 사례(예: "KOICA 공고" vs "koica-공고").
# - 정규화 후 빈 키: 정규화 규칙을 지나고 나서 내용이 사라진 경우 → 데이터 클린업 필요.

# ### 문제 : text 필드 69개라는 사실검증 필요

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                yield json.loads(ln)
            except:
                continue

total = 0
# 1) texts.merged 가 문자열로 들어있는 케이스 (있을 수도/없을 수도)
texts_merged_nonempty = 0
texts_merged_empty    = 0

# 2) merged_text 필드(다른 파이프라인에서 쓰는 이름)
merged_text_nonempty  = 0
merged_text_empty     = 0

# 3) chars.merged 길이 기준(≥1 이면 실질 텍스트가 병합되어 있다고 볼 수 있음)
chars_merged_gt0      = 0
chars_merged_eq0      = 0
chars_merged_missing  = 0

# 4) texts 하위 소스 중 하나라도 내용이 있는 문서 개수
texts_any_nonempty    = 0
texts_any_empty       = 0

mismatch_examples = {"chars>0_but_no_texts.merged": [],
                     "texts.merged_but_chars0": [],
                     "only_xml_or_html_no_explicit_merged": []}

for rec in read_jsonl(FILE):
    total += 1
    mk = rec.get("merge_key","")
    # --- (1) texts.merged ---
    tm = ((rec.get("texts") or {}).get("merged"))
    if isinstance(tm, str):
        if tm.strip():
            texts_merged_nonempty += 1
        else:
            texts_merged_empty += 1

    # --- (2) merged_text ---
    mt = rec.get("merged_text")
    if mt is not None:  # 존재 자체를 카운트
        if isinstance(mt, str) and mt.strip():
            merged_text_nonempty += 1
        else:
            merged_text_empty += 1

    # --- (3) chars.merged ---
    cm = (rec.get("chars") or {}).get("merged")
    if cm is None:
        chars_merged_missing += 1
    else:
        if isinstance(cm, int) and cm > 0:
            chars_merged_gt0 += 1
        else:
            chars_merged_eq0 += 1

    # --- (4) texts 하위 소스 중 하나라도 비어있지 않나? ---
    t = rec.get("texts") or {}
    has_any = False
    for key in ("xml","html","ocr","pdf","merged"):
        v = t.get(key)
        if isinstance(v, str) and v.strip():
            has_any = True
            break
    if has_any:
        texts_any_nonempty += 1
    else:
        texts_any_empty += 1

    # --- 불일치/상황 샘플 수집 ---
    if isinstance(cm, int) and cm > 0 and not (isinstance(tm, str) and tm.strip()):
        # chars 기준으로는 merged가 있는데, texts.merged 실제 문자열이 없음
        if len(mismatch_examples["chars>0_but_no_texts.merged"]) < 5:
            mismatch_examples["chars>0_but_no_texts.merged"].append(mk)

    if isinstance(tm, str) and tm.strip() and (isinstance(cm, int) and cm == 0):
        if len(mismatch_examples["texts.merged_but_chars0"]) < 5:
            mismatch_examples["texts.merged_but_chars0"].append(mk)

    if not (isinstance(tm, str) and tm.strip()):
        # 명시적인 texts.merged는 없지만, xml/html 중 하나만 채워져 있는 케이스
        if ((isinstance(t.get("xml"), str) and t["xml"].strip()) or
            (isinstance(t.get("html"), str) and t["html"].strip())):
            if len(mismatch_examples["only_xml_or_html_no_explicit_merged"]) < 5:
                mismatch_examples["only_xml_or_html_no_explicit_merged"].append(mk)

print("===== docs_merged.jsonl 점검 =====")
print(f"총 문서 수: {total}")

print("\n[텍스트 존재 여부(명시적 필드 기준)]")
print(f"- texts.merged   → 내용 있음: {texts_merged_nonempty}, 내용 없음(빈문자 포함): {texts_merged_empty}")
print(f"- merged_text    → 내용 있음: {merged_text_nonempty}, 내용 없음/비문자열: {merged_text_empty}")

print("\n[길이 지표 기준(chars.merged)]")
print(f"- chars.merged > 0: {chars_merged_gt0}")
print(f"- chars.merged = 0: {chars_merged_eq0}")
print(f"- chars.merged 없음: {chars_merged_missing}")

print("\n[texts 하위 소스 중 하나라도 내용 있음? (xml/html/ocr/pdf/merged)]")
print(f"- 있음: {texts_any_nonempty} / 없음: {texts_any_empty}")

print("\n[불일치/상황 샘플]")
for k, v in mismatch_examples.items():
    if v:
        print(f"- {k}: {len(v)}개 (예시 5개) → {v}")


# 
# #### 41건이 누락이유 추측
# 
# - 원문 파싱 실패(HTML/XML 추출 실패, OCR 미수행/실패)
# - 원본이 스캔 이미지인데 OCR 단계 누락
# - 소스 파일 경로나 접근권한 문제
# - 전처리에서 필터(예: 극단적 짧은 텍스트)로 제거
# 
# 

# --------------------------
# 
# #### 빠른 체크리스트
# 
# - 빈 41건의 `merge_key` 목록 추출해 원본 경로(`source_paths_sample`) 확인
# - `sources`/`counts`/`chars`를 같이 찍어 **어느 단계에서 끊겼는지** 구분(예: `pdf_records>0`인데 `ocr_images_total=0`이면 OCR 누락)
# - 가능하면 OCR 재실행 또는 HTML/plain 추출 재시도

# ### solution 1: 본문 없는 1건 찾아내고, 빈 이유 원인별로 분류

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except:
                continue

empty_docs = []
for rec in read_jsonl(FILE):
    mk = rec.get("merge_key", "")
    doc_id = rec.get("doc_id", "")
    texts = rec.get("texts") or {}
    chars = rec.get("chars") or {}
    counts = rec.get("counts") or {}
    src_paths = rec.get("source_paths_sample") or []

    # 실제 텍스트 여부 판단
    has_text = False
    for k in ("xml","html","ocr","pdf","merged"):
        if isinstance(texts.get(k), str) and texts[k].strip():
            has_text = True
            break

    if not has_text and not (chars.get("merged", 0) > 0):
        empty_docs.append({
            "merge_key": mk,
            "doc_id": doc_id,
            "sources": rec.get("sources"),
            "chars": chars,
            "counts": counts,
            "sample_path": src_paths[0] if src_paths else None
        })

print("본문 없는 문서 수:", len(empty_docs))

# 원인별 분류
reason_groups = {"no_source": [], "ocr_needed": [], "other": []}
for r in empty_docs:
    src = r["sources"] or {}
    counts = r["counts"] or {}

    if sum(src.values()) == 0:
        reason_groups["no_source"].append(r)
    elif counts.get("ocr_images_total", 0) > 0 and counts.get("ocr_images_html", 0) == 0:
        reason_groups["ocr_needed"].append(r)
    else:
        reason_groups["other"].append(r)

print("\n[원인별 분류]")
for reason, docs in reason_groups.items():
    print(f"- {reason}: {len(docs)}건")
    for d in docs[:3]:  # 샘플 3개만 표시
        print("   •", d["merge_key"], "→", d["sample_path"])


# #### text 41건 비어있는 원인 분석

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, os, csv

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")
CSV_OUT = Path("/home/spai0308/data/interim/empty_docs_diagnosis.csv")
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

def rows(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                yield json.loads(ln)
            except:
                continue

def s(x):  # 문자열 길이(안전)
    return len(x.strip()) if isinstance(x, str) else 0

def reason_for(rec):
    """
    비어있는 문서에 대해 보다 세밀한 원인 라벨링
    """
    texts  = rec.get("texts")  or {}
    chars  = rec.get("chars")  or {}
    src    = rec.get("sources") or {}
    paths  = rec.get("source_paths_sample") or []
    path   = paths[0] if paths else None

    # 0) 샘플 경로 상태
    exists = (Path(path).exists() if path else False)
    size   = (os.path.getsize(path) if exists else -1)

    # 1) 텍스트/카운트 관찰
    lens = {k: s(texts.get(k)) for k in ("xml","html","ocr","pdf","merged")}
    chrs = {k: int(chars.get(k,0) or 0) for k in ("xml","html","ocr","pdf","merged")}
    flags = {k: int(src.get(k,0) or 0) for k in ("xml","html","ocr","pdf")}

    any_text = any(lens.values())
    any_char = any(chrs.values())
    any_flag = any(flags.values())

    # 2) 세부 원인 규칙
    if not any_flag and not path:
        return "no_source_trace"   # 소스 흔적 자체가 없음
    if path and not exists:
        return "source_path_missing"
    if exists and size == 0:
        return "source_file_empty"
    if any_flag and not any_char and not any_text:
        return "source_flag_only"  # 소스 플래그는 있는데 실제 본문/글자수 없음
    if any_char and not any_text:
        return "has_chars_but_text_missing"  # 글자수 집계는 있는데 texts.*가 비어있음(병합 누락)
    if any_text and chrs.get("merged",0)==0:
        return "text_present_but_merged_zero" # 텍스트는 있는데 merged 길이 0으로 집계
    # 최종 기타
    return "other"

# === 수집: '본문 없음' 케이스만 타겟 ===
empty = []
for rec in rows(FILE):
    texts = rec.get("texts") or {}
    chars = rec.get("chars") or {}
    has_text = any(isinstance(texts.get(k), str) and texts[k].strip() for k in ("xml","html","ocr","pdf","merged"))
    if has_text:
        continue
    if chars.get("merged",0) > 0:
        continue
    empty.append(rec)

print("본문 없는 문서 수:", len(empty))

# === 라벨링 & 카운트 ===
buckets = {}
for rec in empty:
    r = reason_for(rec)
    buckets[r] = buckets.get(r,0)+1

print("\n[세부 원인 카운트]")
for k,v in sorted(buckets.items(), key=lambda x: (-x[1], x[0])):
    print(f"- {k}: {v}")

# === CSV 저장 ===
cols = [
    "reason","merge_key","doc_id",
    "flag_xml","flag_html","flag_ocr","flag_pdf",
    "chars_xml","chars_html","chars_ocr","chars_pdf","chars_merged",
    "len_xml","len_html","len_ocr","len_pdf","len_merged",
    "sample_path","path_exists","path_size"
]
with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for rec in empty:
        texts  = rec.get("texts")  or {}
        chars  = rec.get("chars")  or {}
        src    = rec.get("sources") or {}
        paths  = rec.get("source_paths_sample") or []
        path   = paths[0] if paths else None
        exists = (Path(path).exists() if path else False)
        size   = (os.path.getsize(path) if exists else -1)
        w.writerow([
            reason_for(rec),
            rec.get("merge_key",""),
            rec.get("doc_id",""),
            int(src.get("xml",0) or 0), int(src.get("html",0) or 0),
            int(src.get("ocr",0) or 0), int(src.get("pdf",0) or 0),
            int(chars.get("xml",0) or 0), int(chars.get("html",0) or 0),
            int(chars.get("ocr",0) or 0), int(chars.get("pdf",0) or 0),
            int(chars.get("merged",0) or 0),
            s(texts.get("xml")), s(texts.get("html")), s(texts.get("ocr")),
            s(texts.get("pdf")), s(texts.get("merged")),
            path or "", int(exists), int(size)
        ])

print("\nCSV 저장:", CSV_OUT)


# **[해결 방법]**
# 
# 
# - source_flag_only가 41건이면
# - “소스 플래그는 있는데 ``texts/*``랑 ``chars/*``가 비어 있음” 케이스
# - 따라서, 샘플 경로에서 원문을 다시 읽어 채워 넣고(texts.*, chars.*, merged) 복구
# 

# #### 샘플경로 복구 작업

# - ``source_paths_sample[0]``에서 텍스트 파일을 읽고
# - 경로/플래그로 **소스 타입(html/xml/ocr/pdf)**를 추정한다.
# - ``texts[src_type]``, ``texts["merged"]``를 채우고,
# - ``chars[src_type]``, ``chars["merged"]``를 각 길이로 채운다.
# - 원본은 .bak로 백업, 수정본은 같은 경로에 덮어쓰기(옵션) 또는 별도 파일로 저장.

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, os, re

# ===== 경로 설정 =====
IN_FILE  = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl")
OUT_FILE = IN_FILE.with_name("docs_merged.fixed.jsonl")   # 수정본을 별도 파일로 저장
BACKUP   = IN_FILE.with_suffix(IN_FILE.suffix + ".bak")   # 혹시 덮어쓸 때 쓰는 백업 경로
WRITE_INPLACE = False  # True면 원본 덮어쓰기(+백업), False면 OUT_FILE에 저장

# ===== 유틸 =====
def read_text(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists() or p.is_dir(): 
            return ""
        # 기본 UTF-8, 실패 시 latin-1 등 시도 (필요하면 추가)
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return p.read_text(encoding="cp949", errors="ignore")
    except Exception:
        return ""

def detect_source_type(rec, sample_path: str) -> str:
    """
    sources 플래그와 경로 패턴으로 src_type(html/xml/ocr/pdf)을 추정
    """
    src = (rec.get("sources") or {})
    # 1) 플래그 우선
    for k in ("html","xml","ocr","pdf"):
        if int(src.get(k,0) or 0) == 1:
            guessed = k
            break
    else:
        guessed = None

    # 2) 경로 힌트 보정
    if sample_path:
        s = sample_path.lower()
        # 경로 명시적 키워드 우선
        if   "html_plain" in s or "/html/" in s or s.endswith(".html") or "htm" in s:
            return "html"
        elif "xml_plain" in s or "/xml/" in s or s.endswith(".xml"):
            return "xml"
        elif "ocr" in s:
            return "ocr"
        elif "pdf" in s:
            return "pdf"
    return guessed or "html"  # 최종 기본값(html)

def lens(texts):
    """각 필드 문자열 길이(공백 제거)"""
    out = {}
    for k in ("xml","html","ocr","pdf","merged"):
        v = texts.get(k)
        out[k] = len(v.strip()) if isinstance(v, str) else 0
    return out

# ===== 실행 =====
total = 0
fixed = 0
skipped = 0
couldnt = []

records = []

with IN_FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            records.append(ln)   # 깨진 라인은 그대로 보존
            continue

        total += 1
        texts  = rec.get("texts") or {}
        chars  = rec.get("chars") or {}
        srcs   = rec.get("sources") or {}
        paths  = rec.get("source_paths_sample") or []
        sample = paths[0] if paths else ""

        has_any_text = any(isinstance(texts.get(k), str) and texts[k].strip()
                           for k in ("xml","html","ocr","pdf","merged"))
        merged_len = int(chars.get("merged", 0) or 0)

        # 수리 대상: source_flag_only 패턴
        flag_on = any(int(srcs.get(k,0) or 0) for k in ("xml","html","ocr","pdf"))
        if flag_on and (not has_any_text) and merged_len == 0:
            src_type = detect_source_type(rec, sample)
            raw = read_text(sample)
            if raw.strip():
                # 딕셔너리 보장
                if not isinstance(rec.get("texts"), dict):
                    rec["texts"] = {}
                if not isinstance(rec.get("chars"), dict):
                    rec["chars"] = {}

                # 채우기
                rec["texts"][src_type] = raw
                rec["texts"]["merged"] = raw
                rec["chars"][src_type] = len(raw)
                rec["chars"]["merged"] = len(raw)

                # sanity: 다른 소스 길이는 없으면 0으로
                for k in ("xml","html","ocr","pdf"):
                    if k != src_type and k not in rec["chars"]:
                        rec["chars"][k] = 0

                fixed += 1
            else:
                couldnt.append({
                    "merge_key": rec.get("merge_key",""),
                    "doc_id": rec.get("doc_id",""),
                    "path": sample,
                    "reason": "read_fail_or_empty"
                })
                skipped += 1
        else:
            skipped += 1

        records.append(rec)

# 저장
if WRITE_INPLACE:
    # 백업 후 덮어쓰기
    if not BACKUP.exists():
        IN_FILE.replace(BACKUP)
    out_path = IN_FILE
else:
    out_path = OUT_FILE

with out_path.open("w", encoding="utf-8") as f:
    for r in records:
        if isinstance(r, str):
            f.write(r + "\n")
        else:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[DONE] total: {total}, fixed: {fixed}, skipped: {skipped}")
print("output:", out_path)
if couldnt:
    print("\n[FAILED TO RECOVER] count =", len(couldnt))
    for r in couldnt[:10]:
        print(" -", r)


# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, os, csv

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.jsonl")
CSV_OUT = Path("/home/spai0308/data/interim/empty_docs_diagnosis.csv")
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

def rows(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                yield json.loads(ln)
            except:
                continue

def s(x):  # 문자열 길이(안전)
    return len(x.strip()) if isinstance(x, str) else 0

def reason_for(rec):
    """
    비어있는 문서에 대해 보다 세밀한 원인 라벨링
    """
    texts  = rec.get("texts")  or {}
    chars  = rec.get("chars")  or {}
    src    = rec.get("sources") or {}
    paths  = rec.get("source_paths_sample") or []
    path   = paths[0] if paths else None

    # 0) 샘플 경로 상태
    exists = (Path(path).exists() if path else False)
    size   = (os.path.getsize(path) if exists else -1)

    # 1) 텍스트/카운트 관찰
    lens = {k: s(texts.get(k)) for k in ("xml","html","ocr","pdf","merged")}
    chrs = {k: int(chars.get(k,0) or 0) for k in ("xml","html","ocr","pdf","merged")}
    flags = {k: int(src.get(k,0) or 0) for k in ("xml","html","ocr","pdf")}

    any_text = any(lens.values())
    any_char = any(chrs.values())
    any_flag = any(flags.values())

    # 2) 세부 원인 규칙
    if not any_flag and not path:
        return "no_source_trace"   # 소스 흔적 자체가 없음
    if path and not exists:
        return "source_path_missing"
    if exists and size == 0:
        return "source_file_empty"
    if any_flag and not any_char and not any_text:
        return "source_flag_only"  # 소스 플래그는 있는데 실제 본문/글자수 없음
    if any_char and not any_text:
        return "has_chars_but_text_missing"  # 글자수 집계는 있는데 texts.*가 비어있음(병합 누락)
    if any_text and chrs.get("merged",0)==0:
        return "text_present_but_merged_zero" # 텍스트는 있는데 merged 길이 0으로 집계
    # 최종 기타
    return "other"

# === 수집: '본문 없음' 케이스만 타겟 ===
empty = []
for rec in rows(FILE):
    texts = rec.get("texts") or {}
    chars = rec.get("chars") or {}
    has_text = any(isinstance(texts.get(k), str) and texts[k].strip() for k in ("xml","html","ocr","pdf","merged"))
    if has_text:
        continue
    if chars.get("merged",0) > 0:
        continue
    empty.append(rec)

print("본문 없는 문서 수:", len(empty))

# === 라벨링 & 카운트 ===
buckets = {}
for rec in empty:
    r = reason_for(rec)
    buckets[r] = buckets.get(r,0)+1

print("\n[세부 원인 카운트]")
for k,v in sorted(buckets.items(), key=lambda x: (-x[1], x[0])):
    print(f"- {k}: {v}")


# - 문제의 4건은 ``source_paths_sample``이 비어 있었으니, 파일 시스템에서 후보 텍스트 파일을 탐색해 매칭하는 방식으로 복구시도
# 
# - 아래 스크립트는 ``/home/spai0308/data/processed/`` 하위의 ``html_plain/*.txt``, ``xml_plain/*.txt``, ``ocr_plain/*.txt``, ``pdf_text/*.txt``를 인덱싱한 뒤,
#     - ``merge_key``/``doc_id``를 정규화(한글/영문/숫자만, 소문자, 공백 제거)해서
#     - 파일명 스템(확장자 제외) 정규화와 토큰 겹침 점수(Jaccard) 로 가장 그럴듯한 후보를 선택함
#     - 해당 파일을 읽어서 ``texts[*]``, ``texts.merged``, ``chars[*]`` 채움

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re

IN_FILE  = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.jsonl")
OUT_FILE = IN_FILE.with_name("docs_merged.fixed.v2.jsonl")

# 파일 후보를 찾을 기본 루트들
ROOT = Path("/home/spai0308/data/processed")
SEARCH_DIRS = [
    ROOT/"html_plain",
    ROOT/"xml_plain",
    ROOT/"ocr_plain",
    ROOT/"pdf_text",
]

def normalize(s: str) -> str:
    if not isinstance(s,str): return ""
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣]+", "", s)  # 한글/영문/숫자만 남김
    return s

def tokenize(s: str):
    # 숫자/영문/한글 단위로 간단 토큰화
    return set(re.findall(r"[0-9a-z가-힣]+", s.lower()))

def jaccard(a:set, b:set):
    if not a or not b: return 0.0
    i = len(a & b); u = len(a | b)
    return i / u if u else 0.0

def detect_src_from_path(p: Path) -> str:
    s = str(p).lower()
    if "html_plain" in s: return "html"
    if "xml_plain"  in s: return "xml"
    if "ocr_plain"  in s: return "ocr"
    if "pdf_text"   in s: return "pdf"
    return "html"

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949", errors="ignore")
    except Exception:
        return ""

# 1) 파일 인덱스 만들기
candidates = []  # (norm_stem, tokens, path)
for d in SEARCH_DIRS:
    if not d.exists(): continue
    for p in d.glob("*.txt"):
        stem = p.stem
        norm = normalize(stem)
        toks = tokenize(stem)
        candidates.append((norm, toks, p))

print(f"[index] 후보 파일 개수: {len(candidates)}")

# 2) 레코드 읽어 2차 복구 대상 찾기
records=[]
total = 0
fixed  = 0
skipped= 0
nohit  = []

with IN_FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln=ln.strip()
        if not ln: continue
        try:
            rec=json.loads(ln)
        except:
            records.append(ln)
            continue

        total += 1
        texts=rec.get("texts") or {}
        chars=rec.get("chars") or {}
        has_any = any(isinstance(texts.get(k), str) and texts[k].strip() for k in ("xml","html","ocr","pdf","merged"))
        if has_any:
            records.append(rec); skipped+=1; continue

        # 매칭 대상 키 두 개 준비
        key = normalize(rec.get("merge_key",""))
        did = normalize(rec.get("doc_id",""))
        keytok = tokenize(rec.get("merge_key",""))
        didtok = tokenize(rec.get("doc_id",""))

        # 후보 중 최고 점수 선택 (merge_key 우선, 동률이면 doc_id 점수 보조)
        best = (0.0, None, None)  # (score, path, src)
        for norm, toks, p in candidates:
            s1 = jaccard(keytok, toks)
            s2 = jaccard(didtok, toks)
            score = max(s1, s2) + 0.05 * min(s1, s2)  # 주점수 + 보조 가중치
            if score > best[0]:
                best = (score, p, detect_src_from_path(p))

        if best[0] >= 0.35 and best[1] is not None:   # 임계치(상황 맞춰 조정 가능)
            raw = read_text(best[1])
            if raw.strip():
                if not isinstance(rec.get("texts"), dict): rec["texts"]={}
                if not isinstance(rec.get("chars"), dict): rec["chars"]={}
                src = best[2]
                rec["texts"][src] = raw
                rec["texts"]["merged"] = raw
                rec["chars"][src] = len(raw)
                rec["chars"]["merged"] = len(raw)
                fixed += 1
            else:
                nohit.append({"merge_key": rec.get("merge_key",""), "doc_id": rec.get("doc_id",""), "reason":"empty_file", "path": str(best[1])})
        else:
            nohit.append({"merge_key": rec.get("merge_key",""), "doc_id": rec.get("doc_id",""), "reason":"no_candidate"})

        records.append(rec)

with OUT_FILE.open("w", encoding="utf-8") as fo:
    for r in records:
        if isinstance(r, str):
            fo.write(r+"\n")
        else:
            fo.write(json.dumps(r, ensure_ascii=False)+"\n")

print(f"[v2] total: {total}, fixed: {fixed}, skipped: {skipped}, unresolved: {len(nohit)}")
if nohit:
    print("[unresolved top 10]")
    for x in nohit[:10]:
        print(" -", x)
print("output:", OUT_FILE)


# #### **other 4개 마저 복구**

# 
# - 검색 루트에 ``/raw/converted_personal_test/html``, ``/raw/converted_personal_test/xml`` 포함
# - ``.txt``뿐 아니라 ``.html``/``.xml`` 파일 인덱싱
# - ``merge_key`` 끝의 해시(예: ``_859d37``)를 떼고 디렉터리 직접 매칭 시도
# - HTML/XML을 읽어 간단 태그 제거(평문 변환) 후 ``texts.html`` 또는 ``texts.xml`` 및 ``texts.merged`` 채움

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re

# 입력/출력
IN_FILE  = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.jsonl")
OUT_FILE = IN_FILE.with_name("docs_merged.fixed.v3.jsonl")

# 1) 후보 탐색 루트 확장: processed + raw/converted_personal_test
ROOT_PROC = Path("/home/spai0308/data/processed")
ROOT_RAW  = Path("/home/spai0308/data/raw/converted_personal_test")

SEARCH_DIRS = [
    ROOT_PROC / "html_plain",    # *.txt
    ROOT_PROC / "xml_plain",     # *.txt
    ROOT_PROC / "ocr_plain",     # *.txt
    ROOT_PROC / "pdf_text",      # *.txt
    ROOT_RAW  / "html",          # */*.html
    ROOT_RAW  / "xml",           # */*.xml
]

# ---------- 유틸 ----------
TAG_RE = re.compile(r"<[^>]+>")
HASH_TAIL_RE = re.compile(r"_[0-9a-f]{6}$", re.IGNORECASE)

def normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣]+", "", s)  # 한/영/숫자만
    return s

def strip_tail_hash(name: str) -> str:
    # merge_key의 마지막 _abcdef 형태 해시 제거
    return HASH_TAIL_RE.sub("", name)

def html_xml_to_text(raw: str) -> str:
    # 아주 단순한 평문 변환(필요시 개선 가능)
    if not isinstance(raw, str): return ""
    # 스크립트/스타일 제거(라이트)
    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
    # 태그 제거
    txt = TAG_RE.sub(" ", raw)
    # 공백 정리
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def detect_src_from_path(p: Path) -> str:
    s = str(p).lower()
    if "html_plain" in s or p.suffix.lower() == ".html": return "html"
    if "xml_plain"  in s or p.suffix.lower() == ".xml":  return "xml"
    if "ocr_plain"  in s: return "ocr"
    if "pdf_text"   in s: return "pdf"
    return "html"

def read_text_file(path: Path, as_plain=True) -> str:
    # as_plain=True면 html/xml도 태그 제거하여 평문 반환
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="cp949", errors="ignore")
    except Exception:
        return ""
    if as_plain and path.suffix.lower() in (".html", ".xml"):
        return html_xml_to_text(raw)
    return raw

# ---------- 0) raw 디렉터리 직접 매핑을 위한 DIR 인덱스 ----------
#   폴더명으로 1:1 매칭 시도(merge_key의 tail hash 제거 후 비교)
dir_index = {
    "html": {},
    "xml": {},
}
for kind in ("html", "xml"):
    base = ROOT_RAW / kind
    if base.exists():
        for d in base.iterdir():
            if d.is_dir():
                key_norm = normalize(d.name)
                dir_index[kind][key_norm] = d

# ---------- 1) 파일 인덱스(후보 풀) ----------
# processed의 *.txt + raw의 */*.html|*.xml
candidates = []  # (norm_stem, path)
for d in SEARCH_DIRS:
    if not d.exists(): continue
    if d.name in ("html", "xml"):  # raw
        # 디렉터리/하위 폴더 내 *.html|*.xml
        exts = ("*.html", "*.xml")
        for sub in d.iterdir():
            if not sub.is_dir(): continue
            for pat in exts:
                for p in sub.glob(pat):
                    candidates.append((normalize(p.stem), p))
    else:
        # processed의 평문 txt
        for p in d.glob("*.txt"):
            candidates.append((normalize(p.stem), p))

print(f"[index] 후보 파일 개수: {len(candidates)}")
print(f"[index] raw/html 폴더 수: {len(dir_index['html'])}, raw/xml 폴더 수: {len(dir_index['xml'])}")

# 빠른 조회용: norm_stem -> paths
from collections import defaultdict
cand_map = defaultdict(list)
for n, p in candidates:
    cand_map[n].append(p)

# ---------- 2) 레코드 읽어 보강 ----------
records=[]
total=fixed=skipped=0
unresolved=[]

def try_direct_dir_match(rec):
    """
    raw/converted_personal_test/{html,xml}/<dir> 에서
    merge_key의 tail hash 떼고 폴더명 직접 매칭 후 파일 읽기
    """
    mk = rec.get("merge_key","")
    base_name = strip_tail_hash(mk)
    norm = normalize(base_name)

    for kind in ("html","xml"):
        d = dir_index[kind].get(norm)
        if not d: 
            # 폴더명이 공백/특수문자 차이로 다를 수도 있으니 느슨한 포함 매칭 보조
            # (너무 과하면 오탐, 여기서는 안전하게 패스)
            continue
        # 우선순위: 같은 이름의 파일 -> 그 외 첫 파일
        cand_files = []
        stem_target = strip_tail_hash(mk)  # 원형 기반
        # 1) 정확/유사 스템 우선
        for p in list(d.glob("*.html")) + list(d.glob("*.xml")):
            if normalize(p.stem) == normalize(stem_target):
                cand_files.insert(0, p)
            else:
                cand_files.append(p)
        for p in cand_files:
            txt = read_text_file(p, as_plain=True)
            if txt.strip():
                src = detect_src_from_path(p)
                return src, txt, str(p)
    return None, None, None

def has_any_text(rec):
    t = rec.get("texts") or {}
    if not isinstance(t, dict): return False
    for k in ("xml","html","ocr","pdf","merged"):
        v = t.get(k)
        if isinstance(v, str) and v.strip():
            return True
    return False

with IN_FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln=ln.strip()
        if not ln: 
            continue
        try:
            rec=json.loads(ln)
        except:
            records.append(ln); 
            continue

        total += 1
        if has_any_text(rec):
            records.append(rec); skipped += 1; 
            continue

        # (A) raw 디렉터리 직접 매칭 우선 시도 (tail hash 제거)
        src, txt, path = try_direct_dir_match(rec)
        if not txt:
            # (B) 후보 인덱스 매칭(노멀라이즈된 파일 스템과 일치)
            mk_norm = normalize(strip_tail_hash(rec.get("merge_key","")))
            did_norm = normalize(strip_tail_hash(rec.get("doc_id","")))
            hit_paths = cand_map.get(mk_norm, []) or cand_map.get(did_norm, [])
            for p in hit_paths:
                tmp = read_text_file(p, as_plain=True)
                if tmp.strip():
                    src = detect_src_from_path(p)
                    txt = tmp
                    path = str(p)
                    break

        if txt:
            if not isinstance(rec.get("texts"), dict): rec["texts"]={}
            if not isinstance(rec.get("chars"), dict): rec["chars"]={}
            rec["texts"][src] = txt
            rec["texts"]["merged"] = txt
            rec["chars"][src] = len(txt)
            rec["chars"]["merged"] = len(txt)
            fixed += 1
            rec.setdefault("source_paths_sample", [])
            if path:
                rec["source_paths_sample"] = list(set(rec["source_paths_sample"] + [path]))
        else:
            unresolved.append({
                "merge_key": rec.get("merge_key",""),
                "doc_id": rec.get("doc_id",""),
                "reason": "no_candidate_v3"
            })

        records.append(rec)

with OUT_FILE.open("w", encoding="utf-8") as fo:
    for r in records:
        if isinstance(r, str):
            fo.write(r + "\n")
        else:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[v3] total: {total}, fixed: {fixed}, skipped: {skipped}, unresolved: {len(unresolved)}")
if unresolved:
    print("[unresolved up to 10]")
    for x in unresolved[:10]:
        print(" -", x)
print("output:", OUT_FILE)


# #### v3 요약 점검(요약 통계)

# In[ ]:


from pathlib import Path
import json

FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.v3.jsonl")

total = 0
texts_merged = 0
chars_merged_pos = 0
chars_merged_zero = 0

per_source_has = {"xml":0,"html":0,"ocr":0,"pdf":0}
unresolved = []  # texts.*가 전부 빈 경우

def has_text(s): return isinstance(s, str) and s.strip()

with FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        obj = json.loads(ln)
        total += 1

        t = obj.get("texts") or {}
        c = obj.get("chars") or {}
        if has_text(t.get("merged")):
            texts_merged += 1

        cm = c.get("merged")
        if isinstance(cm, int):
            if cm > 0: chars_merged_pos += 1
            else:      chars_merged_zero += 1

        any_src = False
        for k in ("xml","html","ocr","pdf"):
            if has_text(t.get(k)):
                per_source_has[k] += 1
                any_src = True
        if not any_src:
            unresolved.append({"merge_key": obj.get("merge_key",""),
                               "doc_id": obj.get("doc_id","")})

print("===== docs_merged.fixed.v3.jsonl 점검 =====")
print(f"총 문서 수: {total}")
print(f"- texts.merged 내용 있음: {texts_merged}")
print(f"- chars.merged > 0: {chars_merged_pos}")
print(f"- chars.merged = 0: {chars_merged_zero}")
print("[소스별 텍스트 존재 수]", per_source_has)
print(f"본문 추출 실패(모든 소스 비어있음): {len(unresolved)}")
if unresolved[:5]:
    print("예시 5건:", unresolved[:5])


# #### “전(before) vs 후(v3)” 변화 비교(고쳐진 문서 확인)

# In[ ]:


import json
from pathlib import Path

OLD = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.jsonl")
NEW = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.v3.jsonl")

def load_map(p):
    m={}
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o=json.loads(ln)
            k=o.get("merge_key","")
            m[k]=o
    return m

old = load_map(OLD)
new = load_map(NEW)

fixed_keys=[]
still_empty=[]
for k, rec in new.items():
    t = (rec.get("texts") or {})
    ok = any((isinstance(t.get(x), str) and t.get(x).strip()) for x in ("xml","html","ocr","pdf","merged"))
    if ok and k in old:
        told = (old[k].get("texts") or {})
        old_ok = any((isinstance(told.get(x), str) and told.get(x).strip()) for x in ("xml","html","ocr","pdf","merged"))
        if not old_ok:
            fixed_keys.append(k)
    if not ok:
        still_empty.append(k)

print(f"[변화] old→new에서 새로 복구된 문서 수: {len(fixed_keys)}")
print("복구 예시 5건:", fixed_keys[:5])
print(f"여전히 비어있는 문서 수: {len(still_empty)}")
print("비어있는 예시 5건:", still_empty[:5])


# #### 문제 남은 문서 CSV로 덤프

# #### docs_merged.jsonl 구조/품질 재점검

# In[81]:


# === docs_merged.jsonl 구조/품질 빠른 점검 스크립트 ===
# * 목적: JSONL 파일의 필드(키) 구조, 타입, 결측 비율, 길이 통계, 대표 값 분포 등을 한 번에 확인
# * 사용 대상: /home/spai0308/data/processed/docs_merged.jsonl
# * 출력: 콘솔 요약 + 샘플 2건 출력

from pathlib import Path
import json
from collections import Counter, defaultdict
import math

# [설정] 점검할 파일 경로를 변수로 빼두면 재사용이 편리합니다.
JSONL_PATH = Path("/home/spai0308/data/processed/docs_merged.jsonl")

# [함수] JSONL 한 줄씩 제너레이터로 읽기 (대용량 파일에도 안전)
def iter_jsonl(p):
    # (설명) encoding='utf-8'로 한글 파일도 안전하게 읽습니다.
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                # (설명) 각 줄을 JSON으로 파싱해서 dict로 반환합니다.
                yield json.loads(ln)

# [방어] 파일 존재 여부 확인
assert JSONL_PATH.exists(), f"파일이 존재하지 않습니다: {JSONL_PATH}"

# [수집용 컨테이너]
all_keys = set()                              # (설명) 전체 문서에서 나타난 모든 키 모음
key_counts = Counter()                        # (설명) 각 키가 등장한 문서 수
key_type_counts = defaultdict(Counter)        # (설명) 키별 값 타입 분포 (str/int/list 등)
key_len_stats = defaultdict(list)             # (설명) 키가 문자열일 때 길이 리스트(텍스트 길이 통계용)
rep_type_counter = Counter()                  # (설명) 대표 필드(예: 'rep_type'/'source_type') 분포를 볼 때 사용
merge_key_list = []                           # (설명) merge_key 중복 체크용
sample_records = []                           # (설명) 샘플 몇 건 저장해 훑어보기
n_total = 0                                   # (설명) 전체 레코드 수

# [선택] 프로젝트 표준에서 자주 쓰이는 키 후보들 (있으면 잡아 보여줍니다)
LIKELY_KEYS = ["merge_key", "join_key", "source_id", "rep_type", "rep", "text", "doc_text", "ocr_text", "title", "url"]

# === 1) 1차 패스: 구조/타입/기본 통계 수집 ===
for rec in iter_jsonl(JSONL_PATH):
    n_total += 1
    # (설명) 처음 2건은 샘플로 보관
    if len(sample_records) < 2:
        sample_records.append(rec)

    # (설명) 레코드의 키를 모으고, 키별 등장 횟수 누적
    keys = list(rec.keys())
    all_keys.update(keys)
    key_counts.update(keys)

    # (설명) 타입/길이 통계 수집
    for k, v in rec.items():
        # (설명) None은 'NoneType'으로 기록
        vtype = type(v).__name__
        key_type_counts[k][vtype] += 1

        # (설명) 문자열이면 길이 통계에 추가
        if isinstance(v, str):
            key_len_stats[k].append(len(v))

    # (설명) 흔히 쓰는 대표 필드 분포를 카운트 (있을 경우에만)
    if "rep_type" in rec:
        rep_type_counter[rec["rep_type"]] += 1
    elif "rep" in rec and isinstance(rec["rep"], str):
        rep_type_counter[rec["rep"]] += 1

    # (설명) merge_key 중복 점검을 위해 수집 (키가 있을 때만)
    if "merge_key" in rec:
        merge_key_list.append(rec["merge_key"])

# === 2) 요약 출력 ===
print("# === 파일 기본 정보 ===")
print(f"- 경로: {JSONL_PATH}")
print(f"- 총 레코드 수: {n_total:,}")

print("\n# === 전체 키(필드) 목록 ===")
print(sorted(all_keys))

print("\n# === 키별 등장 비율(=결측 비율 역추정) ===")
# (설명) 각 키가 전체 레코드 중 몇 %에서 등장했는지 출력 → 낮으면 결측이 많다는 뜻
for k in sorted(all_keys):
    cnt = key_counts[k]
    ratio = (cnt / n_total * 100) if n_total else 0.0
    print(f"- {k:20s} : {cnt:6d}개  ({ratio:5.1f}%)")

print("\n# === 키별 값 타입 분포(데이터 타입 일관성 점검) ===")
# (설명) 같은 키에서 타입이 여러 개면 스키마가 흔들린다는 신호
for k in sorted(all_keys):
    tdist = ", ".join(f"{t}={c}" for t, c in key_type_counts[k].most_common())
    print(f"- {k:20s} : {tdist}")

print("\n# === 문자열 필드 길이 요약(텍스트 크기 감 잡기) ===")
# (설명) 대표적으로 text/doc_text/ocr_text/title 등 문자열 길이의 최소/중간/평균/최대 출력
def quick_stats(vals):
    # (설명) 기본적인 길이 통계를 계산하는 작은 함수
    if not vals:
        return None
    vs = sorted(vals)
    n = len(vs)
    avg = sum(vs) / n
    p50 = vs[n//2]
    return {
        "n": n,
        "min": vs[0],
        "p50": p50,
        "avg": avg,
        "max": vs[-1],
    }

for k in sorted(key_len_stats.keys()):
    st = quick_stats(key_len_stats[k])
    if st:
        print(f"- {k:20s} : n={st['n']}, min={st['min']}, p50={st['p50']}, avg={st['avg']:.1f}, max={st['max']}")

print("\n# === 대표 타입 분포(rep_type/rep 등) ===")
# (설명) 문서 대표 타입(예: xml/html/ocr/pdf 등)이 어떻게 섞였는지 확인
if rep_type_counter:
    for val, c in rep_type_counter.most_common():
        print(f"- {val}: {c}")
else:
    print("- (rep_type/rep 유사 필드를 찾지 못했습니다)")

print("\n# === merge_key 중복 점검 ===")
# (설명) merge_key가 전체 레코드마다 1개여야 하는데 중복이면 합치기/머지 로직 점검 필요
if merge_key_list:
    dup = [k for k, c in Counter(merge_key_list).items() if c > 1]
    print(f"- merge_key 존재 레코드 수: {len(merge_key_list)}")
    print(f"- 중복된 merge_key 수: {len(dup)}")
    if dup[:5]:
        print(f"- (예시) 중복 merge_key 상위 5개: {dup[:5]}")
else:
    print("- merge_key 필드를 가진 레코드가 없습니다.")

print("\n# === 샘플 레코드 (앞 2건) ===")
# (설명) 실제 레코드를 두 건만 예쁘게 출력해 대략 구조를 눈으로도 확인
def pretty(d, indent=2, maxlen=300):
    # (설명) 너무 긴 문자열 필드는 일부만 잘라 보여주기 (가독성)
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > maxlen:
            out[k] = v[:maxlen] + "...[TRUNC]"
        else:
            out[k] = v
    return json.dumps(out, ensure_ascii=False, indent=indent)

for i, rec in enumerate(sample_records, 1):
    print(f"\n[샘플 #{i}] ===========================")
    print(pretty(rec))

# === (선택) 특정 키 존재여부 빠른 확인 ===
print("\n# === 흔히 쓰는 키(LIKELY_KEYS) 존재 여부 ===")
for k in LIKELY_KEYS:
    print(f"- {k:10s}: {'존재' if k in all_keys else '없음'}")


# In[ ]:


# v3 생성 이후 재점검

# 재점검 스크립트

from pathlib import Path
import json, csv

IN_FILE = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.v3.jsonl")
CSV_OUT = IN_FILE.with_name("diagnosis_after_fix.csv")

total=0
has_any=0
per_field={"xml":0,"html":0,"ocr":0,"pdf":0,"merged":0}
empties=[]
rows=[]

def L(x): return len(x.strip()) if isinstance(x,str) else 0

with IN_FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln=ln.strip()
        if not ln: continue
        try:
            rec=json.loads(ln)
        except: 
            continue
        total+=1
        texts=rec.get("texts") or {}
        chars=rec.get("chars") or {}
        lens={k: (chars.get(k) if isinstance(chars.get(k), int) else L(texts.get(k))) for k in ("xml","html","ocr","pdf","merged")}
        if any(lens[k]>0 for k in lens): has_any+=1
        for k in per_field: 
            if lens[k]>0: per_field[k]+=1
        if not any(lens[k]>0 for k in lens):
            empties.append({"merge_key":rec.get("merge_key",""),"doc_id":rec.get("doc_id","")})
        rows.append({
            "merge_key": rec.get("merge_key",""),
            "doc_id": rec.get("doc_id",""),
            **{f"len_{k}": lens[k] for k in lens}
        })

print("===== 재점검 결과 =====")
print(f"총 문서 수: {total}")
print(f"본문(어느 필드든) 있는 문서: {has_any}")
print(f"본문 없는 문서: {total-has_any}")
print("[소스별 내용 있음(>0) 문서 수]")
for k,v in per_field.items():
    print(f" - {k:6s}: {v}")

if empties:
    print("\n[본문 없는 문서 목록(상위 20개)]")
    for e in empties[:20]:
        print(" -", e)

# CSV 저장
with CSV_OUT.open("w", newline="", encoding="utf-8") as fo:
    w=csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nCSV 저장:", CSV_OUT)


# # 5. 아티팩트 정리

# ## 5-1. 1차: 2025.09.25 (``chuck_meta.json``, ``faiss.index`` 포함 버전 - 코사인 유사도 이슈)

# - `/processed/`에 `docs_merged.jsonl`, `docs_chunks.jsonl`, 리포트(`.txt/*.csv`),
#     
#     표 인덱스(`tables_index.jsonl` or `rep_pdf.jsonl` 내 포함) 정리.

# In[223]:


# artifact_pack.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os, json, hashlib, tarfile, time
from datetime import datetime

BASE = Path("/home/spai0308/data")
INTERIM = BASE / "interim"
PROCESSED = BASE / "processed"

# ===== 설정 =====
BUILD_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = BASE / "release" / BUILD_ID           # 산출물 루트
LINK_MODE = "symlink"                             # "symlink" | "copy" | "hardlink"
MAKE_TAR = True                                   # True면 tar.gz 생성

# 임베딩/인덱스 메타(가능한 값 자동 추정·기본값 세팅)
INDEX_DIR  = PROCESSED / "index"
INDEX_META = INDEX_DIR / "chunks_meta.jsonl"
FAISS_FILE = INDEX_DIR / "faiss.index"
INDEX_CONF = {
    "backend": "openai",
    "model": "text-embedding-3-small",
    "dim": 1536,
    "index_type": "faiss.IndexFlatIP",
}

# ===== 유틸 =====
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def line_count(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def safe_link(src: Path, dst: Path, mode: str = "symlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"[SKIP] missing: {src}")
        return False
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        # 작은 파일 위주라면 shutil.copy2 사용
        import shutil
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        # 기본: 심볼릭 링크
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return True

def collect(files_map: list[tuple[Path, Path]]) -> list[dict]:
    """(src, rel_dest) 목록을 OUT_DIR에 모으고 메타 리턴"""
    manifest = []
    for src, rel in files_map:
        dst = OUT_DIR / rel
        ok = safe_link(src, dst, LINK_MODE)
        if not ok:
            continue
        meta = {
            "src": str(src),
            "dest": str(rel),
            "bytes": src.stat().st_size if src.exists() else 0,
            "lines": line_count(src) if src.suffix == ".jsonl" else None,
            "sha256": sha256_file(src) if src.exists() else None,
        }
        manifest.append(meta)
    return manifest

# ===== 수집 대상 정의 =====
files_to_collect = [
    # 1) 최종 병합 산출물
    (PROCESSED / "docs_merged.jsonl",                 Path("data/docs_merged.jsonl")),
    (INTERIM / "merge_qc_report.txt",                 Path("qc/merge_qc_report.txt")),
    (INTERIM / "merge_coverage.csv",                  Path("qc/merge_coverage.csv")),

    # 2) 청크/인덱스 산출물
    (INDEX_META,                                      Path("index/chunks_meta.jsonl")),
    (FAISS_FILE,                                      Path("index/faiss.index")),

    # 3) rep_* (소스별 대표본)
    (INTERIM / "rep_xml.jsonl",                       Path("sources/rep_xml.jsonl")),
    (INTERIM / "rep_html.jsonl",                      Path("sources/rep_html.jsonl")),
    (INTERIM / "rep_ocr.jsonl",                       Path("sources/rep_ocr.jsonl")),
    (INTERIM / "rep_pdf.fixed.jsonl",                 Path("sources/rep_pdf.jsonl")),

    # 4) 조인키/감사 자료
    (INTERIM / "joinkey_fixes.json",                  Path("audit/joinkey_fixes.json")),
    (INTERIM / "join_audit.txt",                      Path("audit/join_audit.txt")),
    (INTERIM / "join_audit.csv",                      Path("audit/join_audit.csv")),

    # 5) OCR 원본/집계
    (INTERIM / "assets_html_ocr.jsonl",               Path("ocr/assets_html_ocr.jsonl")),
    (INTERIM / "assets_xml_ocr.jsonl",                Path("ocr/assets_xml_ocr.jsonl")),
    (INTERIM / "ocr_text_by_doc.jsonl",               Path("ocr/ocr_text_by_doc.jsonl")),

    # 6) XML/HTML 텍스트 원본(merge_key 적용본이 있으면 우선)
    (INTERIM / "xml_text.with_mergekey.jsonl",        Path("raw_text/xml_text.jsonl")),
    (INTERIM / "xml_text.jsonl",                      Path("raw_text/xml_text.original.jsonl")),
    (INTERIM / "html_text.jsonl",                     Path("raw_text/html_text.jsonl")),   # 있으면 수집

    # 7) PDF 전처리/표 스냅샷 메타
    (INTERIM / "pages_pdf.cleaned.jsonl",             Path("pdf/pages_pdf.cleaned.jsonl")),
    (INTERIM / "table_snapshots_manifest.jsonl",      Path("pdf/table_snapshots_manifest.jsonl")),
    (INTERIM / "table_snapshots_manifest.filtered.jsonl", Path("pdf/table_snapshots_manifest.filtered.jsonl")),
    # 스냅샷 원본 디렉토리는 용량이 크면 링크만:
    # (INTERIM / "table_snapshots",                  Path("pdf/table_snapshots")),
    # (INTERIM / "table_snapshots_filtered",         Path("pdf/table_snapshots_filtered")),

    # 8) 변환 로그
    (INTERIM / "hwp_convert_errors.log",              Path("logs/hwp_convert_errors.log")),
]

# ===== 실행 =====
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = collect(files_to_collect)

# 인덱스 설정 파일 기록(자동 추정 포함)
vectors = line_count(INDEX_META)
index_config = {
    **INDEX_CONF,
    "vectors": vectors,
    "faiss_path": str(Path("index/faiss.index")),
    "meta_path":  str(Path("index/chunks_meta.jsonl")),
}
(OUT_DIR / "index/index_config.json").write_text(
    json.dumps(index_config, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# README 작성
readme = f"""# Release Bundle — {BUILD_ID}

## What’s inside
- **data/docs_merged.jsonl**: 최종 병합 문서(HTML 본문 + XML 본문 + OCR 텍스트, 중복 제거/출처 포함)
- **index/**: FAISS 인덱스와 청크 메타
- **sources/**: 소스별 대표본(rep_*)
- **raw_text/**: XML/HTML 추출 원본(jsonl)
- **ocr/**: OCR 원시/집계
- **pdf/**: PDF 전처리 산출물 & 표 스냅샷 메타
- **qc/**: 병합 품질/커버리지 리포트
- **audit/**: join_key/merge_key 충돌 해소 근거 자료
- **logs/**: 변환 로그

## Index
- backend: {index_config["backend"]}
- model:   {index_config["model"]}
- dim:     {index_config["dim"]}
- vectors: {index_config["vectors"]}
- faiss:   {index_config["faiss_path"]}

## Notes
- 이 번들은 기본적으로 **{LINK_MODE}** 방식으로 수집됨.
- 대용량 이미지 스냅샷 디렉토리는 필요 시 주석 해제하여 포함 가능.
"""
(OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

# 아티팩트 매니페스트(JSON)
(OUT_DIR / "ARTIFACTS_MANIFEST.json").write_text(
    json.dumps({
        "build_id": BUILD_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "link_mode": LINK_MODE,
        "items": manifest
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# tar.gz 패키징(심볼릭 링크는 링크 그대로 보존)
if MAKE_TAR:
    tar_path = OUT_DIR.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(OUT_DIR, arcname=OUT_DIR.name, recursive=True)
    print(f"[TAR] created: {tar_path}")

# 간단 요약 출력
total_bytes = sum(m["bytes"] for m in manifest if m.get("bytes"))
print("[ARTIFACT] build:", BUILD_ID)
print("[ARTIFACT] out dir:", OUT_DIR)
print("[ARTIFACT] files:", len(manifest), " total bytes:", total_bytes)
print("[ARTIFACT] index vectors:", index_config["vectors"])
print("[ARTIFACT] README:", OUT_DIR / "README.md")
print("[ARTIFACT] MANIFEST:", OUT_DIR / "ARTIFACTS_MANIFEST.json")


# ### 실행 후 생기는 것
# 
# - `/home/spai0308/data/release/<타임스탬프>/` 아래에
#     - `data/docs_merged.jsonl`, `index/faiss.index`, `index/chunks_meta.jsonl`, `index/index_config.json`
#     - `sources/rep_*.jsonl`, `raw_text/*.jsonl`, `ocr/*.jsonl`
#     - `pdf/*`, `qc/*`, `audit/*`, `logs/*`
#     - `README.md`, `ARTIFACTS_MANIFEST.json`
# - 그리고 동일 경로에 `<타임스탬프>.tar.gz` 패키지(옵션)까지.

# ## 5-2. 2차: 2025.09.29 (``docs_merged.jsonl`` 구조/품질 재구성(누락된 text 41개 포함) 후)

# - 오늘자 release 아래 디렉터리 생성
# - 필요한 파일 복사/정규화
# - 점검 리포트/체크섬/매니페스트 생성

# **[무엇이 정리되나요?]**
# 
# - ``/home/spai0308/data/release/20250929/``
# - ``data/docs_merged.jsonl`` : v3를 표준 파일명으로 복사(다운스트림 호환)
# - ``data/diagnosis_after_fix.csv`` : v3 재점검 CSV(있을 때)
# - ``reports/release_diagnosis.txt`` : 합계/필드별 보유 수 등 요약
# - ``reports/release_diagnosis_detail.csv`` : 문서별 필드 길이 상세
# - ``processed/…, interim/…`` : 과거 산출물 있으면 함께 복사
# - ``MANIFEST.csv`` : 모든 파일 사이즈·SHA256 체크섬
# - ``README.md`` : 릴리즈 메모

# **[결과 확인 포인트]**
# 
# - ``reports/release_diagnosis.txt``
#     → ``empties: 0`` 이면, v3에서 확인한 상태와 일치(정상).
# 
# - ``MANIFEST.csv``
#     → 어떤 파일이 어디에 복사되었는지 일람 확인.
# 
# - ``data/docs_merged.jsonl``
#     → 이번 릴리스의 “정본(CANON)”이 이 위치에 안전하게 들어간 것(해시 동일) 확인.
# 
# 

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import shutil, hashlib, sys, csv, json
from datetime import datetime

# ===== 설정 =====
RELEASE_ROOT = Path("/home/spai0308/data/release/20250929")
DATA_DIR     = RELEASE_ROOT / "data"
CANON        = DATA_DIR / "docs_merged.jsonl"     # 이번 릴리스의 '정본' 파일
LEGACY_DIR   = DATA_DIR / "legacy"
REPORTS_DIR  = RELEASE_ROOT / "reports"

# v3에서 검증이 끝난 소스(정상 110, empties=0) — 너가 방금 프리플라이트했던 그 파일
IN_JSONL = Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.v3.jsonl")

# 함께 담아두고 싶은(선택) 아티팩트들 — 필요에 맞게 추가/수정
OPTIONAL_SOURCES = [
    Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.fixed.jsonl"),
    Path("/home/spai0308/data/release/20250925_051143/data/docs_merged.jsonl"),
    Path("/home/spai0308/data/release/20250925_051143/data/diagnosis_after_fix.csv"),
    Path("/home/spai0308/data/interim/empty_docs_diagnosis.csv"),
    # Path("/home/spai0308/data/processed/chunks_xmlhtml_800_200.jsonl"),
    # Path("/home/spai0308/data/interim/chunks_manifest.csv"),
    # Path("/home/spai0308/data/interim/chunks_stats.txt"),
]

MANIFEST_CSV = RELEASE_ROOT / "MANIFEST.csv"
REPORT_TXT   = REPORTS_DIR / "release_diagnosis.txt"

# ===== 유틸 =====
def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_len(x):
    return len(x.strip()) if isinstance(x, str) else 0

def scan_text_presence(jsonl_path: Path):
    """각 문서에서 texts.{xml,html,ocr,pdf,merged}와 chars.*를 보고
       본문 존재 여부/카운트를 계산."""
    total = 0
    empties = 0
    per_field = {"xml":0,"html":0,"ocr":0,"pdf":0,"merged":0}
    empty_list = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                rec = json.loads(ln)
            except:
                continue
            total += 1
            texts = rec.get("texts") or {}
            chars  = rec.get("chars") or {}
            lens = {k: (chars.get(k) if isinstance(chars.get(k), int) else safe_len(texts.get(k)))
                    for k in ("xml","html","ocr","pdf","merged")}
            has_any = any(lens[k] > 0 for k in lens)
            if not has_any:
                empties += 1
                empty_list.append({
                    "merge_key": rec.get("merge_key", ""),
                    "doc_id": rec.get("doc_id", "")
                })
            for k in per_field:
                if lens[k] > 0:
                    per_field[k] += 1

    return {
        "total": total,
        "empties": empties,
        "per_field": per_field,
        "empty_list": empty_list,
    }

# ===== 실행 =====
# 디렉토리 준비
for d in (DATA_DIR, LEGACY_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 1) CANON 먼저 생성
if not IN_JSONL.exists():
    print("[ERR] source not found:", IN_JSONL)
    sys.exit(2)

src_hash = sha256_of_file(IN_JSONL)
shutil.copy2(IN_JSONL, CANON)

# 복사 직후 쌍방 해시 확인
canon_hash = sha256_of_file(CANON)
if canon_hash != src_hash:
    print("[ABORT] CANON write mismatch right after copy")
    print("        src :", src_hash)
    print("        canon:", canon_hash)
    sys.exit(3)

# 2) 옵션 아티팩트 복사 (CANON 보호)
copied = []
for src in OPTIONAL_SOURCES:
    if not src.exists():
        continue

    s = str(src)
    # release 트리 배치 규칙
    if "/processed/" in s:
        dst = RELEASE_ROOT / "processed" / src.name
    elif "/interim/" in s:
        dst = RELEASE_ROOT / "interim" / src.name
    else:
        # data/에 들어오는데 docs_merged.jsonl과 충돌하면 legacy로 회피
        if src.name == "docs_merged.jsonl":
            dst = LEGACY_DIR / src.name
        else:
            dst = DATA_DIR / src.name

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst)

    # 매번 CANON 무결성 확인 (혹시라도 덮어쓰기 감지)
    canon_hash_after_each = sha256_of_file(CANON)
    if canon_hash_after_each != src_hash:
        print("[ABORT] CANON was modified during optional copies!")
        print("        expected:", src_hash)
        print("        actual  :", canon_hash_after_each)
        print("        last copied:", dst)
        sys.exit(4)

# 3) 최종 무결성 재확인
final_hash = sha256_of_file(CANON)
if final_hash != src_hash:
    print("[ABORT] Final CANON hash mismatch")
    print("        expected:", src_hash)
    print("        actual  :", final_hash)
    sys.exit(5)

# 4) 진단 리포트/매니페스트 생성 (CANON 기준)
scan = scan_text_presence(CANON)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 리포트
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
with REPORT_TXT.open("w", encoding="utf-8") as rf:
    rf.write("=== RELEASE DIAGNOSIS ===\n")
    rf.write(f"time   : {timestamp}\n")
    rf.write(f"root   : {RELEASE_ROOT}\n")
    rf.write(f"canon  : {CANON}\n")
    rf.write(f"sha256 : {final_hash}\n")
    rf.write("\n[scan]\n")
    rf.write(f"total  : {scan['total']}\n")
    rf.write(f"empties: {scan['empties']}\n")
    rf.write(f"per field(has>0): {scan['per_field']}\n")
    if scan["empty_list"]:
        rf.write("\n[empty docs - first 10]\n")
        for e in scan["empty_list"][:10]:
            rf.write(f" - {e}\n")

# 매니페스트 (간단 버전)
with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as mf:
    w = csv.writer(mf)
    w.writerow(["path","type","note"])
    w.writerow([str(CANON), "canon", f"sha256={final_hash}"])
    for p in copied:
        w.writerow([str(p), "optional", ""])
    w.writerow([str(REPORT_TXT), "report", "text_presence"])

# 콘솔 출력
print("=== NEW RELEASE READY ===")
print("root  :", RELEASE_ROOT)
print("canon :", CANON)
print("report:", REPORT_TXT)
print("mani  :", MANIFEST_CSV)
print("sha256:", final_hash)
print("scan  :", scan)


# ### 새 docs_merged.jsonl 점검 코드

# In[110]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json

FILE = Path("/home/spai0308/data/release/20250929/data/docs_merged.jsonl")

total = 0
texts_merged_yes = 0
texts_merged_no = 0
merged_text_yes = 0
merged_text_no = 0
chars_gt0 = 0
chars_eq0 = 0
chars_none = 0
any_text_yes = 0
any_text_no = 0

def L(x): return len(x.strip()) if isinstance(x,str) else 0

with FILE.open("r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        try:
            rec = json.loads(ln)
        except:
            continue
        total += 1
        texts = rec.get("texts") or {}
        chars = rec.get("chars") or {}

        # texts.merged
        if L(texts.get("merged","")) > 0:
            texts_merged_yes += 1
        else:
            texts_merged_no += 1

        # merged_text (혹시 별도 필드 있는 경우)
        if L(rec.get("merged_text","")) > 0:
            merged_text_yes += 1
        else:
            merged_text_no += 1

        # chars.merged
        if "merged" in chars:
            if chars["merged"] > 0:
                chars_gt0 += 1
            else:
                chars_eq0 += 1
        else:
            chars_none += 1

        # texts 하위(xml/html/ocr/pdf/merged) 중 하나라도 본문 존재 여부
        lens = {k:L(texts.get(k,"")) for k in ("xml","html","ocr","pdf","merged")}
        if any(v>0 for v in lens.values()):
            any_text_yes += 1
        else:
            any_text_no += 1

print("===== docs_merged.jsonl 점검 =====")
print(f"총 문서 수: {total}\n")
print("[텍스트 존재 여부(명시적 필드 기준)]")
print(f"- texts.merged   → 내용 있음: {texts_merged_yes}, 내용 없음(빈문자 포함): {texts_merged_no}")
print(f"- merged_text    → 내용 있음: {merged_text_yes}, 내용 없음/비문자열: {merged_text_no}\n")

print("[길이 지표 기준(chars.merged)]")
print(f"- chars.merged > 0: {chars_gt0}")
print(f"- chars.merged = 0: {chars_eq0}")
print(f"- chars.merged 없음: {chars_none}\n")

print("[texts 하위 소스 중 하나라도 내용 있음? (xml/html/ocr/pdf/merged)]")
print(f"- 있음: {any_text_yes} / 없음: {any_text_no}\n")


# In[ ]:




