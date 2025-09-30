#!/usr/bin/env python
# coding: utf-8

# # Baseline Notebook
# 
# 본 문서는 팀 공유용 **Baseline 정리본**입니다. 
# 
# 
# 코드 실행은 하지 않으며, 기존 출력값은 삭제했습니다.
# 
# 
# 환경설정/설치/디버그/중복 코드는 제거하고, 섹션별로 재배치했습니다.

# # Index
# 1. Background Setting
# 2. DataLoad & EDA
# 3. 통합 Jsonl 생성
#    - 공통 준비 & CSV 매핑 (정규화 키)
#     - PDF
#         - PDF 텍스트 파싱
#         - PDF 이미지 스냅샷
#         - PDF 클린 JSONL, 테이블 스냅샷 정합성 체크
#         - 기존 클린 JSONL + 표 CSV 병합 
#     - HWP → HTML/XML
#         - HTML 텍스트 추출
#         - 포맷 수동 변환 후, HTML 내장 이미지 추출
#         - HTML, XML 자산 OCR
#         - OCR 결과 ``doc_id`` 단위로 집계
#         - XML 텍스트 추출
# 
# 4. Issue
#     - 2) 병합 문서 merge_key의 text 41개 누락
# 8. Artifacts Merge
#     - 1차 : 1차 청킹 & 임베딩 작업 후 (코사인 유사도 문제 포함)
#     - 2차 : 병합 문석 text 누락 이슈 해결 이후 

# # 1. Background Setting

# ## 가상환경 활성화

# ## 깃세팅

# ## 백업  파일 생성

# # 2. DataLoad & EDA

# In[ ]:


# gdown 설치(현재  커널)
import sys
get_ipython().system('{sys.executable} -m pip install -U gdown')

# sys.excutable : 지금 노트북이 사용 중인 파이썬 실행기 경로(현재 가상환경)
# -U: 최신 버전으로 업그레이드 설치


# In[ ]:


# 폴더 다운로드 + 평탄화 함수
# gdown.download_folder 사용 + 한겹 폴더 자동 평탄화 + files/대소문자 보정

from pathlib import Path
import shutil, os
import gdown

RAW = Path.home() / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
# -> 최종 목표 위치: rag-project/data/raw

def download_drive_folder(folder_link: str, out_dir: Path = RAW):
    """
    구글드라이브 '폴더' 공유링크를 받아서 out_dir로 통째 다운로드.
    다운로드 후 최상위 폴더가 한 겹 생기면 내용을 out_dir로 평탄화.
    """
    # 1) 폴더 통째 다운로드
    print("[1/3] 폴더 다운로드 중...")
    gdown.download_folder(url=folder_link, output=str(out_dir), quiet=False, use_cookies=False)

    # 2) 평탄화: out_dir 안에 폴더가 한 겹 생겼다면, 그 안의 내용물을 꺼내기
    # 예: raw/SomeFolder/{data_list.csv, files/}  ->  raw/{data_list.csv, files/}
    entries = [p for p in out_dir.iterdir() if p.name not in (".ipynb_checkpoints",)]
    # 폴더 한 겹만 있는 경우 감지
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        print(f"[2/3] 평탄화: 최상위 '{inner.name}' 제거하고 내용만 올리기")
        for item in inner.iterdir():
            dest = out_dir / item.name
            if dest.exists():
                # 같은 이름 있으면 먼저 지움
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        # 껍데기 폴더 삭제
        shutil.rmtree(inner, ignore_errors=True)

    # 3) 최종 구조 확인 및 안내
    print("[3/3] 최종 검증:")
    print(" - data_list.csv 존재:", (out_dir / "data_list.csv").exists())
    print(" - files/ 폴더 존재:", (out_dir / "files").exists())
    if not (out_dir / "data_list.csv").exists():
        print("'data_list.csv'가 보이지 않아요. 폴더 링크 안의 파일명을 확인하세요(철자가 'csv'인지).")
    if not (out_dir / "files").exists():
        # 일부 자료는 'Files', 'FILES' 등으로 올려둔 경우가 있음 → 자동 보정
        for cand in out_dir.iterdir():
            if cand.is_dir() and cand.name.lower() == "files":
                cand.rename(out_dir / "files")
                print("  폴더명을 'files'로 정규화했습니다.")
                break

    print("\n예시 나열:")
    for p in sorted(out_dir.iterdir())[:10]:
        print(" -", p.name)


# In[ ]:


# pandas 설치
import sys
get_ipython().system('{sys.executable} -m pip install -U pandas')


# In[ ]:


# files.zip -> files/로 rubust 추출

from pathlib import Path
import zipfile, shutil, tempfile

RAW = Path.home() / "data" / "raw"               # 쓰고 있는 경로와 동일하게 맞춤
zip_path = RAW / "files.zip"                     # 방금 받은 zip
files_dir = RAW / "files"                        # 최종 목표 폴더

files_dir.mkdir(parents=True, exist_ok=True)        # 없으면 생성

with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()                            # zip 안 파일 목록
    has_top_files_dir = any(n.lower().startswith("files/") for n in names)
    # zip 내부에 이미 'files/' 최상위 폴더가 포함되어 있는지 검사

    if has_top_files_dir:
        # zip 안에 files/가 이미 있으면 그대로 raw/ 아래로 풀기
        z.extractall(RAW)
    else:
        # 최상위에 바로 pdf/hwp들이 있으면 임시폴더에 풀고 → raw/files/로 이동
        tmp = Path(tempfile.mkdtemp())
        z.extractall(tmp)
        for p in tmp.rglob("*"):
            if p.is_file():
                dest = files_dir / p.name          # 같은 이름이면 덮어씀
                shutil.move(str(p), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)

print("✓ 압축 해제 완료:", files_dir)


# ### files/ 안 내용 빠른 점검 + CSV 교차검증

# In[ ]:


# 디렉토리 요약 ( 파일개수, 용량, 확장자 분포, 예시)

from pathlib import Path
RAW = Path.home() / "data" / "raw"
FILES_DIR = RAW / "files"

all_files = [p for p in FILES_DIR.glob("**/*") if p.is_file()]    # 하위 모든 파일만 수집(폴더 제외)
total_count = len(all_files)    # 전체 파일 개수
total_bytes = sum(p.stat().st_size for p in all_files)    # 전체 용량(바이트)
ext_counts = {}    # 확장자별 개수 집계용 딕셔너리
for p in all_files:
    ext = p.suffix.lower()    # 확장자를 소문자로 정규화
    ext_counts[ext] = ext_counts.get(ext, 0) + 1     # 확장자별 개수 누적

print("files 경로:", FILES_DIR)   # 현재 검사 중인 경로 표시
print("총 파일 개수:", total_count)    # 전체 파일 개수 출력
print(f"총 용량(GB): {total_bytes/1024/1024/1024:.3f}")    # 전체 용량을 GB로 환산 출력
print("확장자 분포:", ext_counts)    # 확장자별 개수 분포 출력

print("예시 5개:")   
for p in all_files[:5]:   # 앞부터 5개만
    print(" -", p.name, f"({p.stat().st_size/1024:.1f} KB)")   #파일명과 용량(KB) 표시


# In[ ]:


# CSV에 있는 파일들이 실제로 모두 존재하는지 교차검증

import pandas as pd
import os   # 경로에서 파일명만 뽑아내기 위해 사용

CSV_PATH = RAW / "data_list.csv"  # 메타데이터 CSV 경로
assert CSV_PATH.exists(), "data_list.csv가 없습니다 .경로를 확인하세요"
# csv 존재 여부 확인

df = pd.read_csv(CSV_PATH)
print("CSV 컬럼:", list(df.columns))   # csv의 열 이름들 출력(파일명 컬럼 확인용))


# In[ ]:


# 이름 정규화 헬퍼 (NFC 통일 + 공백/특수문자 보정)

from pathlib import Path              # 경로 안전 처리
import pandas as pd                  # CSV 처리
import os, re, unicodedata           # 경로/정규식/유니코드 정규화

RAW = Path.home() / "data" / "raw"   # 네가 사용 중인 경로 그대로
FILES_DIR = RAW / "files"            # 문서 폴더
CSV_PATH = RAW / "data_list.csv"     # 메타 CSV

# 유니코드 정규화 + 공백/특수문자 정리 함수
def normalize_name(s: str) -> str:
    s = str(s)

    # 1) 유니코드 정규화: NFD/NFC 섞임을 NFC로 통일
    s = unicodedata.normalize("NFC", s)

    # 2) 경로/역슬래시/여러 구분자 정리
    s = s.replace("\\", "/").split("/")[-1]      # 경로 제거
    s = s.replace("\u200b", "")                  # zero width space 제거
    s = s.replace("\ufeff", "")                  # BOM 제거

    # 3) 비슷한 특수문자 통일 (필요시 추가)
    s = s.replace("·", ".")                      # 가운데점 → 점
    s = s.replace("ㆍ", ".")                     # 한글 가운뎃점 → 점
    s = s.replace("–", "-").replace("—", "-")    # 대시 통일
    s = s.replace("“", '"').replace("”", '"')    # 따옴표 통일
    s = s.replace("’", "'")

    # 4) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()           # 연속 공백 → 한 칸, 양끝 공백 제거
    s = re.sub(r"\s+(\.[A-Za-z0-9]+)$", r"\1", s)  # 확장자 앞 공백 제거: "이름 .hwp" → "이름.hwp"

    # 5) 소문자화(대소문자 구분 없애기)
    s = s.lower()
    return s

# CSV의 파일명 컬럼 → 확장자 보정 포함
def normalize_csv_row(row, filename_col="파일명", filetype_col="파일형식"):
    base = normalize_name(row[filename_col])
    # 확장자가 없으면 파일형식으로 추정
    if not base.endswith((".pdf", ".hwp", ".hwpx")):
        ft = str(row.get(filetype_col, "")).lower()
        if "pdf" in ft and not base.endswith(".pdf"):
            base += ".pdf"
        elif "hwp" in ft and not base.endswith((".hwp", ".hwpx")):
            # HWPX도 있을 수 있음
            if "x" in ft:
                base += ".hwpx"
            else:
                base += ".hwp"
    return base


# ### 문서 길이, 문단 단위 분포, 헤더 패턴, OCR 비율 확인

# # 3. 통합 JSONL 만들기

# ## 공통 준비 & CSV 매핑 (정규화 키)

# #### 코드의 목적
# 
# - **CSV의 파일명**과 **실제 디스크에 있는 파일명**을 **완전히 동일한 규칙**으로 ‘정규화’해서, 키 하나(정규화 파일명)로 **빠르게 매칭(lookup)** 하기 위함.
#     
#     → 이후 파이프라인에서 “이 파일의 메타데이터(CSV 행)”를 헷갈림 없이 즉시 가져올 수 있음
# 

# In[ ]:


# 공통 준비 + CSV 매핑(정규화 키)

from pathlib import Path
import os, json, re, unicodedata
import pandas as pd

RAW = Path.home() / "data" / "raw"           # 팀 규칙상 raw는 로컬 관리
FILES_DIR = RAW / "files"
CSV_PATH = RAW / "data_list.csv"
INTERIM = Path.home() / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

def normalize_name(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFC", s)       # 한글 정규화(NFC) 통일
    s = s.replace("\\", "/").split("/")[-1]   # 경로 제거
    s = s.replace("\u200b","").replace("\ufeff","")  # 눈에 안 보이는 공백 제거
    s = (s.replace("·",".").replace("ㆍ",".")         # 특수문자 통일
           .replace("–","-").replace("—","-")
           .replace("“",'"').replace("”",'"').replace("’","'"))
    s = re.sub(r"\s+", " ", s).strip()        # 연속 공백 정리
    s = re.sub(r"\s+(\.[A-Za-z0-9]+)$", r"\1", s)   # 확장자 앞 공백 제거: "이름 .hwp" → "이름.hwp"
    return s.lower()                          # 대소문자 무시

def normalize_csv_row(row, filename_col="파일명", filetype_col="파일형식"):
    base = normalize_name(row[filename_col])  # 파일명 정규화
    if not base.endswith((".pdf",".hwp",".hwpx")):          # 확장자 없으면
        ft = str(row.get(filetype_col,"")).lower()          # '파일형식' 보고 추정
        if "pdf" in ft and not base.endswith(".pdf"):
            base += ".pdf"
        elif "hwp" in ft and not base.endswith((".hwp",".hwpx")):
            base += ".hwp"                                  # hwpx는 필요 시 분기
    return base

df = pd.read_csv(CSV_PATH)
assert "파일명" in df.columns, "'파일명' 컬럼이 필요합니다."
csv_map = { normalize_csv_row(r): r.to_dict() for _, r in df.iterrows() }  # 정규화키 → 메타행


# ## 3-1. PDF 파일

# ### 1) PDF 텍스트 파싱

# In[ ]:


# 사전점검 : 경로/파일 개수

from pathlib import Path

RAW = Path.home() / "data" / "raw"
FILES_DIR = RAW / "files"
INTERIM = Path.home() / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

print("RAW:", RAW)
print("FILES_DIR exists:", FILES_DIR.exists())
print("PDF/HWP counts:",
      len(list(FILES_DIR.rglob("*.pdf"))),
      len(list(FILES_DIR.rglob("*.hwp"))),
      len(list(FILES_DIR.rglob("*.hwpx"))))


# In[ ]:


# ---> 공통 유틸 : 파일명 정규화 + csv 매핑 --> row 확인용

import pandas as pd, re, unicodedata, json

CSV_PATH = RAW / "data_list.csv"

def normalize_name(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\\", "/").split("/")[-1]
    s = s.replace("\u200b","").replace("\ufeff","")
    s = (s.replace("·",".").replace("ㆍ",".")
         .replace("–","-").replace("—","-")
         .replace("“",'"').replace("”",'"').replace("’","'"))
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+(\.[A-Za-z0-9]+)$", r"\1", s)
    return s.lower()

def normalize_csv_row(row, filename_col="파일명", filetype_col="파일형식"):
    base = normalize_name(row[filename_col])
    if not base.endswith((".pdf",".hwp",".hwpx")):
        ft = str(row.get(filetype_col,"")).lower()
        if "pdf" in ft and not base.endswith(".pdf"):
            base += ".pdf"
        elif "hwp" in ft and not base.endswith((".hwp",".hwpx")):
            base += ".hwp"
    return base

df = pd.read_csv(CSV_PATH)
assert "파일명" in df.columns, "'파일명' 컬럼이 필요합니다."
csv_map = { normalize_csv_row(r): r.to_dict() for _, r in df.iterrows() }
print("CSV rows:", len(csv_map))


# In[ ]:


# 교체용 : 경로 캡처 & pdf 파싱 (에러 원인 수정 버전)


# 이 셀은 normalize_name, csv_map, FILES_DIR, INTERIM 이 앞서 정의되어 있다고 가정
# 없다면 '공통 준비 + CSV 매핑' 셀을 먼저 실행필요!!!

import os, sys, contextlib, tempfile, json
import fitz
import pandas as pd
from pathlib import Path

@contextlib.contextmanager
def capture_stderr():
    """stderr를 임시파일로 리다이렉트해 캡처한다."""
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as tf:
        try:
            os.dup2(tf.fileno(), fd)
            yield tf
        finally:
            os.dup2(saved_fd, fd)
            os.close(saved_fd)

pdf_rows, pdf_fail, warn_log = [], [], []
pdf_list = list(FILES_DIR.rglob("*.pdf"))
print("PDF files found:", len(pdf_list))

for p in pdf_list:
    key = normalize_name(p.name)
    meta = csv_map.get(key, {})

    try:
        with capture_stderr() as errbuf:
            doc = fitz.open(p)
            for i, page in enumerate(doc):
                text = page.get_text("text") or page.get_text() or ""
                pdf_rows.append({
                    "doc_id": p.stem,
                    "page": i + 1,
                    "text": text,
                    "source": str(p),
                    "format": "pdf",
                    "meta": meta
                })
            # 여기서 errbuf 내용을 '문자열'로 먼저 꺼내둔다 (블록을 나가면 errbuf는 닫힘)
            errbuf.flush(); errbuf.seek(0)
            msg = errbuf.read().decode("utf-8", "ignore").strip()
        # 블록을 벗어난 뒤에는 '문자열'만 사용
        if msg:
            warn_log.append({"file": str(p), "stderr": msg[:2000]})

        # 자원 정리(선택)
        try:
            doc.close()
        except Exception:
            pass

    except Exception as e:
        pdf_fail.append({"file": str(p), "error": str(e)})

# 저장
pdf_out = INTERIM / "pages_pdf.jsonl"
with open(pdf_out, "w", encoding="utf-8") as f:
    for r in pdf_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[PDF] rows={len(pdf_rows)} | fails={len(pdf_fail)} | saved→ {pdf_out}")

if pdf_fail:
    fail_csv = INTERIM / "pdf_parse_failures.csv"
    pd.DataFrame(pdf_fail).to_csv(fail_csv, index=False)
    print("PDF 실패 목록:", fail_csv)

if warn_log:
    warn_csv = INTERIM / "pdf_parse_warnings.csv"
    pd.DataFrame(warn_log).to_csv(warn_csv, index=False)
    print("경고 로그 저장:", warn_csv)


# In[ ]:


# 결과 파일 줄수로 pdf 파싱 결과 확인

from pathlib import Path

INTERIM = Path.home() / "data" / "interim"

def nlines(p): 
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0

pdf_path  = INTERIM/"pages_pdf.jsonl"
hwp_path  = INTERIM/"pages_hwp.jsonl"
all_path  = INTERIM/"pages_all_merged.jsonl"

print("lines(pdf) =", nlines(pdf_path))
print("lines(hwp) =", nlines(hwp_path))
print("lines(all) =", nlines(all_path))


# In[ ]:


# 다시 파일에서 경고난 거 있는지 확인

import pandas as pd
from pathlib import Path

INTERIM = Path.home() / "data" / "interim"
warn_csv = INTERIM/"pdf_parse_warnings.csv"

if warn_csv.exists():
    warn = pd.read_csv(warn_csv)
    warn["file"] = warn["file"].str.replace(r".*/", "", regex=True)
    warn["kind"] = warn["stderr"].str.extract(r"(syntax error:[^\n]+)", expand=False)
    print("경고 총 건수:", len(warn))
    print("\n파일별 경고 TOP 10")
    display(warn.groupby("file").size().sort_values(ascending=False).head(10))
    print("\n경고 유형 TOP 5")
    display(warn["kind"].value_counts().head(5))
else:
    print("경고 로그 없음(깨끗)")


# In[ ]:


# 머지 파일 갱신 : pdf만!

from pathlib import Path
import shutil

INTERIM = Path.home() / "data" / "interim"
pdf_path = INTERIM/"pages_pdf.jsonl"
hwp_path = INTERIM/"pages_hwp.jsonl"
all_path = INTERIM/"pages_all_merged.jsonl"

with open(all_path, "w", encoding="utf-8") as out:
    if pdf_path.exists():
        with open(pdf_path, encoding="utf-8") as f:
            out.writelines(f.readlines())
    if hwp_path.exists():  # 나중에 hwp5txt 설치 후 다시 합치면 됨
        with open(hwp_path, encoding="utf-8") as f:
            out.writelines(f.readlines())

print("머지 완료 →", all_path)


# In[ ]:


# 머지 산출물 내용 확인 : 품질 확인 (pdf만) 

import json, random, textwrap
from pathlib import Path

INTERIM = Path.home() / "data" / "interim"
rows = [json.loads(l) for l in open(INTERIM/"pages_all_merged.jsonl", encoding="utf-8")]
print("총 레코드:", len(rows))

for r in random.sample(rows, min(3, len(rows))):
    preview = textwrap.shorten((r["text"] or "").replace("\n"," "), width=300, placeholder=" …")
    print(f"\n[{r['format']}] {r['source']} | p.{r['page']}")
    print(preview or "(빈 텍스트)")


# In[ ]:


# 경로, 라인, 용량 전부 확인 용 + 첫줄 프리뷰


from pathlib import Path
import json

INTERIM = Path.home() / "data" / "interim"
pdf_path  = INTERIM / "pages_pdf.jsonl"
hwp_path  = INTERIM / "pages_hwp.jsonl"
all_path  = INTERIM / "pages_all_merged.jsonl"

def info(p: Path):
    print("·", p)
    print("  - exists:", p.exists())
    if p.exists():
        try:
            print("  - size(bytes):", p.stat().st_size)
        except Exception as e:
            print("  - size(?) err:", e)
        try:
            n = sum(1 for _ in open(p, encoding="utf-8"))
        except UnicodeDecodeError:
            # 혹시 인코딩 이슈면 바이너리로 라인 카운트
            n = sum(1 for _ in open(p, "rb"))
            print("  - (binary line count)")
        print("  - lines:", n)
        if n > 0:
            print("  - first line preview:")
            with open(p, encoding="utf-8") as f:
                print("   ", f.readline().strip()[:200])
    print()


# In[ ]:


# 1) 문서별 페이지 수/문자수 요약 (청킹 크기 판단용)
import json, pandas as pd
from pathlib import Path
INTERIM = Path.home() / "data" / "interim"

rows = [json.loads(l) for l in open(INTERIM/"pages_pdf.jsonl", encoding="utf-8")]
df = pd.DataFrame(rows)

# (설명) 문서별 총/평균 길이 → 평균이 800자 근처면 800/200 매우 적절
doc_stats = df.groupby("doc_id").agg(
    pages=("page", "count"),
    total_chars=("text", lambda s: sum(len(t or "") for t in s)),
    avg_chars=("text", lambda s: sum(len(t or "") for t in s) / max(1, len(s))),
).reset_index()
print(doc_stats.sort_values("pages", ascending=False).head(10))


# In[ ]:


# 2) 빈 페이지/짧은 페이지 비율 (파싱 누락/이미지 위주 페이지 점검)
import numpy as np

df["char_len"] = df["text"].fillna("").map(len)
# (설명) 임계값 50자는 “거의 빈 페이지”로 간주(표지, 목차, 이미지 위주 등)
empty_ratio = (df["char_len"] < 50).mean()
print("빈/매우짧은 페이지 비율:", round(empty_ratio*100, 2), "%")


# In[ ]:


# 3) (설명) 섹션 헤더가 줄 첫머리가 아닐 수도 있어, 앞쪽 공백/기호 허용
# - 예: " Ⅰ. 사업 개요", "2) 추진 일정", "제1장", "제 2 조", "1.2.3 소제목"
import re
hdr = df["text"].fillna("").str.contains(
    r"(^|\n)\s*(제?\s*\d+\s*(장|절|조)?|[Ⅰ-Ⅹ]+\s*[.)]|[0-9]+\s*[.)]|[0-9]+(\.[0-9]+){1,}\s+)",
    regex=True
)
print("헤더(섹션) 패턴 비율(보강):", round(hdr.mean()*100, 2), "%")


# In[ ]:


# 4) 중복 의심 페이지 탐지 (잡음/반복 템플릿 확인)
dup = df.groupby(["doc_id","text"]).size().reset_index(name="n").query("n>=2")
print("문서 내부 동일 텍스트 페이지 수:", len(dup))


# #### PDF 클린 + JSONL 저장

# In[ ]:


# 5) 페이지 번호/머리말·꼬리말 제거 전처리 스니펫 (노이즈 감소)
import re, json

def clean_page_text(t: str) -> str:
    if not t: return ""
    # (설명) '-123-', ' - 45 - ' 같은 단독 페이지표기 제거
    t = re.sub(r"^\s*-?\s*\d+\s*-\s*$", "", t, flags=re.M)
    # (설명) 너무 긴 공백/개행 정리
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# (옵션) 필요한 경우, 청킹 전에 클린 버전으로 별도 파일 생성
out = INTERIM / "pages_pdf.cleaned.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        r["text"] = clean_page_text(r.get("text",""))
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("클린 저장 →", out)


# In[ ]:


# 클린 전/후 파일의 기본 현황 비교 (라인 수/용량/문자수 합)

# 목적: 원본 vs 클린 파일의 라인 수(=페이지 수), 총 문자수, 평균 문자수 비교
import json
from pathlib import Path
import pandas as pd

INTERIM = Path.home() / "data" / "interim"
src = INTERIM / "pages_pdf.jsonl"           # 원본
dst = INTERIM / "pages_pdf.cleaned.jsonl"   # 클린본

def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

rows_src = load_jsonl(src)
rows_dst = load_jsonl(dst)

print("라인 수(페이지 수) — 원본 vs 클린:", len(rows_src), len(rows_dst))

# DataFrame으로 비교 요약
df_src = pd.DataFrame(rows_src)
df_dst = pd.DataFrame(rows_dst)
df_src["char_len"] = df_src["text"].fillna("").map(len)
df_dst["char_len"] = df_dst["text"].fillna("").map(len)

summary = pd.DataFrame({
    "total_chars":[df_src["char_len"].sum(), df_dst["char_len"].sum()],
    "avg_chars":[df_src["char_len"].mean(), df_dst["char_len"].mean()],
    "max_chars":[df_src["char_len"].max(), df_dst["char_len"].max()],
    "min_chars":[df_src["char_len"].min(), df_dst["char_len"].min()],
}, index=["before","after"])
print("\n[문자수 요약] 원본 vs 클린")
print(summary)


# In[ ]:


# "페이지 번호 줄: 제거 효과 있었는지 확인

# 목적: 제거 대상인 '페이지번호만 있는 줄'이 실제로 사라졌는지 샘플 확인
import re, random

def page_number_only_lines(text: str):
    if not text: return []
    hits = []
    for i, ln in enumerate(text.splitlines(), start=1):
        if re.fullmatch(r"-?\s*\d+\s*-?", ln.strip()):
            hits.append((i, ln))
    return hits

# 원본에서 페이지번호줄이 있던 페이지를 찾아본다
samples = []
for r0, r1 in zip(rows_src, rows_dst):
    hits0 = page_number_only_lines(r0.get("text",""))
    hits1 = page_number_only_lines(r1.get("text",""))
    if hits0 and not hits1:
        samples.append({
            "doc_id": r0["doc_id"],
            "page": r0["page"],
            "removed_count": len(hits0),
            "before_sample": hits0[:3],  # 앞 몇 줄만
        })
# 샘플 몇 개 출력
print("페이지번호만 있는 줄이 제거된 페이지 수:", len(samples))
for s in samples[:5]:
    print(f"- {s['doc_id']} p.{s['page']} / 제거줄:{s['removed_count']} / 예시:{s['before_sample']}")



# In[ ]:


# 헤더(섹션) 패턴 비율 - 전/후 비교

# 목적: 클린 후 헤더 탐지 비율 변화(대개 큰 변화는 없으나, 소폭 개선될 수 있음)
import re

pattern = re.compile(
    r"(?m)(^|\n)\s*(?:제?\s*\d+\s*(?:장|절|조)?|[Ⅰ-Ⅹ]+\s*[.)]|\d+\s*[.)]|\d+(?:\.\d+){1,}\s+)"
)

def header_ratio(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    hdr = df["text"].fillna("").str.contains(pattern)
    return round(hdr.mean()*100, 2)

print("헤더 비율 — 원본:", header_ratio(rows_src), "%")
print("헤더 비율 — 클린:", header_ratio(rows_dst), "%")


# In[ ]:


# 너무 짧은 페이지 비율: 전후/ 비교 (노이즈 감소 확인)

# 목적: 공백 정리로 인해 '매우 짧은 페이지'가 더 명확해졌는지 확인
import pandas as pd

def short_ratio(rows, threshold=50):
    df = pd.DataFrame(rows)
    clen = df["text"].fillna("").map(len)
    return round((clen < threshold).mean()*100, 2)

print("빈/매우짧은 페이지 비율(원본):", short_ratio(rows_src), "%")
print("빈/매우짧은 페이지 비율(클린):", short_ratio(rows_dst), "%")


# In[ ]:


# 5) 전/후 차이가 있는 페이지 샘플 프리뷰 (눈으로 최종 확인)

# 목적: 같은 페이지의 before/after를 2~3개 정도 비교 프리뷰
import textwrap, random

pairs = list(zip(rows_src, rows_dst))
random.shuffle(pairs)

print("== Before/After 샘플 ==")
for r0, r1 in pairs[:3]:
    b = textwrap.shorten((r0["text"] or "").replace("\n"," "), width=220, placeholder=" …")
    a = textwrap.shorten((r1["text"] or "").replace("\n"," "), width=220, placeholder=" …")
    print(f"\n[{r0['doc_id']}] p.{r0['page']}")
    print(" before:", b or "(빈 텍스트)")
    print(" after :", a or "(빈 텍스트)")


# ### 2) PDF 테이블 관련 이미지 스냅샷 뽑기

# In[ ]:


# 설치 : 터미널
# pdfpulber: pdf 텍스트/선/테이블 감지 라이브러리
# pillow : 이미지 저장용
# tqdm : 진행률 바

# pip install pdfplumber pillow tqdm


# In[ ]:


# -*- coding: utf-8 -*-
"""
PDF에서 '테이블 영역'을 자동 탐지하여 고해상도 PNG 스냅샷으로 저장하고
매니페스트(JSONL)를 기록하는 스크립트 (안정화 패치 포함).

- 입력 폴더:  ~/data/raw/files   (하위 모든 .pdf)
- 출력 폴더:  ~/data/interim/table_snapshots/<pdf_stem>/
- 매니페스트: ~/data/interim/table_snapshots_manifest.jsonl

핵심 개선:
1) bbox를 페이지 경계로 clamp(보정) → "Bounding box ... not within page" 방지
2) 너무 얇은 bbox 스킵 → PIL "tile cannot extend outside image" 방지
3) RGBA → RGB 변환 및 copy() 저장 → 간헐적 PNG 저장 에러 회피
4) lines 전략 실패 시 text 전략으로 1회 재시도 → 탐지율↑
"""

from pathlib import Path
import pdfplumber
from PIL import Image
import hashlib, json, re
from tqdm import tqdm
import sys
import traceback

# -----------------------------
# 1) 경로 기본값 설정
# -----------------------------
BASE = Path.home() / "data"                                      # 홈/data
PDF_DIR = BASE / "raw" / "files"                                 # 입력 PDF 루트
OUT_DIR = BASE / "interim" / "table_snapshots"                   # 스냅샷 출력 루트
MANIFEST = BASE / "interim" / "table_snapshots_manifest.jsonl"   # 매니페스트 경로

# -----------------------------
# 2) 테이블 탐지 민감도 설정
# -----------------------------
TABLE_SETTINGS_LINES = {              # (기본) 선 기반 탐지
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}
TABLE_SETTINGS_TEXT = {               # (재시도) 텍스트 블록 기반 탐지
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

DPI = 300  # 렌더링 해상도(선명도). 200~300 권장.

# -----------------------------
# 3) 유틸: md5 해시
# -----------------------------
def md5_bytes(data: bytes) -> str:
    # 바이트에 대해 md5 계산
    return hashlib.md5(data).hexdigest()

def md5_file(p: Path) -> str:
    # 파일 내용으로 md5 계산
    return md5_bytes(p.read_bytes())

# -----------------------------
# 4) bbox 클램핑(보정) 유틸
# -----------------------------
def clamp_bbox_to_page(bbox, page, min_w=4, min_h=4):
    """
    bbox를 페이지 경계(0,0,w,h)로 보정.
    너무 얇은 박스(너비/높이 < min_w/min_h)는 None 반환하여 스킵.
    """
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page.bbox  # 예: (0, 0, width, height)

    # 경계 내로 보정
    x0 = max(px0, min(x0, px1))
    y0 = max(py0, min(y0, py1))
    x1 = max(px0, min(x1, px1))
    y1 = max(py0, min(y1, py1))

    # 좌표 이상(역전) 시 교환
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    # 최소 크기 미만이면 스킵
    if (x1 - x0) < min_w or (y1 - y0) < min_h:
        return None
    return (x0, y0, x1, y1)

# -----------------------------
# 5) 텍스트 샘플 추출(미리보기용)
# -----------------------------
def sample_text(page, bbox, max_len=120):
    """
    테이블 bbox 영역의 텍스트를 앞부분만 샘플로 기록(품질/검증용).
    crop 후 extract_text() 사용.
    """
    try:
        crop = page.crop(bbox)                            # CroppedPage
        text = (crop.extract_text() or "").strip()
        text = re.sub(r"\s+", " ", text)                  # 공백 정규화
        return text[:max_len]
    except Exception:
        return ""

# -----------------------------
# 6) 스냅샷 저장 함수
# -----------------------------
def save_snapshot(page, bbox, out_path: Path, dpi=DPI):
    """
    페이지를 bbox로 crop → 렌더링 → PNG 저장.
    - bbox는 페이지 경계로 clamp
    - 너무 얇으면 False 반환(스킵)
    - RGBA 등은 RGB로 변환 + copy() 후 저장(안정화)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # bbox 보정
    clamped = clamp_bbox_to_page(bbox, page)
    if clamped is None:
        return False  # 저장 안 함

    # crop(strict=True 기본) — clamped라 에러 없음
    cropped_page = page.crop(clamped)

    # 렌더링 후 PIL 이미지 얻기
    pil_im = cropped_page.to_image(resolution=dpi).original

    # 모드 정규화 + copy()로 저장 안정화
    if pil_im.mode not in ("RGB", "L"):
        pil_im = pil_im.convert("RGB")
    pil_im = pil_im.copy()

    pil_im.save(str(out_path), format="PNG")
    return True

# -----------------------------
# 7) 테이블 탐지(재시도 포함)
# -----------------------------
def find_tables_with_retry(page):
    """
    1차: 선 기반(lines) → 실패 시
    2차: 텍스트 기반(text)으로 재시도
    """
    try:
        tables = page.find_tables(table_settings=TABLE_SETTINGS_LINES) or []
        if tables:
            return tables, "lines"
        # 실패하면 text 전략으로 재시도
        tables = page.find_tables(table_settings=TABLE_SETTINGS_TEXT) or []
        if tables:
            return tables, "text"
        return [], "none"
    except Exception:
        # 에러가 나도 빈 리스트 반환하여 파이프라인 지속
        return [], "error"

# -----------------------------
# 8) 메인 처리: 한 PDF → 다수 스냅샷
# -----------------------------
def process_pdf(pdf_path: Path, out_dir: Path, manifest_fp):
    """
    하나의 PDF 파일을 열고 모든 페이지를 순회하면서
    table bbox를 찾고, 잘라서 PNG 저장 + 매니페스트 기록.
    """
    pdf_md5 = md5_file(pdf_path)  # 입력 PDF md5(추적/무결성)
    saved_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # 테이블 후보 탐지 (재시도 포함)
                tables, strategy = find_tables_with_retry(page)
                if not tables:
                    continue  # 이 페이지에서 테이블 못 찾으면 스킵

                # 페이지별 테이블 순회
                for t_idx, t in enumerate(tables, start=1):
                    bbox_raw = t.bbox                                 # 원본 감지 박스
                    clamped = clamp_bbox_to_page(bbox_raw, page)      # 보정 박스
                    if clamped is None:
                        continue                                       # 너무 얇으면 스킵

                    # 저장 경로: <pdf_stem>/p{page}_t{idx}.png
                    out_img = out_dir / pdf_path.stem / f"p{page_idx:03d}_t{t_idx:02d}.png"

                    # 스냅샷 저장(성공/실패 체크)
                    success = save_snapshot(page, clamped, out_img, dpi=DPI)
                    if not success:
                        continue  # 실패 시 다음 테이블로

                    # 스냅샷 파일 md5
                    img_md5 = md5_file(out_img)

                    # 텍스트 샘플(미리보기)
                    preview = sample_text(page, clamped)

                    # 매니페스트 기록(원본/보정 bbox 모두 저장)
                    rec = {
                        "pdf_path": str(pdf_path),
                        "pdf_md5": pdf_md5,
                        "page": page_idx,
                        "table_index": t_idx,
                        "detect_strategy": strategy,          # lines/text/none/error
                        "bbox_raw": list(bbox_raw),           # 감지된 원본 박스
                        "bbox": list(clamped),                # 실제 저장에 사용한 보정 박스
                        "dpi": DPI,
                        "image_path": str(out_img),
                        "image_md5": img_md5,
                        "text_preview": preview
                    }
                    manifest_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved_count += 1

    except Exception as e:
        # 개별 PDF 처리 중 오류가 나도 전체 배치가 멈추지 않게 함
        sys.stderr.write(f"[ERROR] {pdf_path.name}: {e}\n")
        traceback.print_exc()

    return saved_count

# -----------------------------
# 9) 배치 실행
# -----------------------------
def run_batch():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 디렉터리에서 pdf 포맷나 선택(대소문자 확장자 모두 포함)
    # - 하위폴더까지 처리하려면 rglob("*.pdf")로 바꾸면 된다!
    pdf_files = sorted([p for p in PDF_DIR.glob("*.pdf") if p.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"[WARN] PDF가 없습니다: {PDF_DIR}")
        return

    # 무엇을 처리하는지 먼저 보여줌(확인용)
    print(f"[INFO] 대상 PDF {len(pdf_files)}개:")
    for p in pdf_files:
        print(" -", p.name)
    
    total_snapshots = 0
    with open(MANIFEST, "w", encoding="utf-8") as mf:
        for pdf_path in tqdm(pdf_files, desc="PDF processing"):
            count = process_pdf(pdf_path, OUT_DIR, mf)
            total_snapshots += count

    print(f"[OK] 완료: 스냅샷 {total_snapshots}개 저장")
    print(f" - 입력 폴더 : {PDF_DIR}")
    print(f" - 출력 폴더 : {OUT_DIR}")
    print(f" - 매니페스트: {MANIFEST}")

if __name__ == "__main__":
    run_batch()


# In[ ]:


# 검증 명령어
# 생성된 이미지 5개만 미리보기(경로만 출력)
get_ipython().system('find ~/data/interim/table_snapshots -type f -name "*.png" | head -n 5')
print("="*50)

# 총 스냅샷 개수
get_ipython().system('find ~/data/interim/table_snapshots -type f -name "*.png" | wc -l')
print("="*50)

# 진짜 PDF 4개만 잡히는지 확인
get_ipython().system('find ~/data/raw/files -maxdepth 1 -type f -iname "*.pdf" -printf "%f\\n"')

# json 스냅샷 확인
get_ipython().system('wc -l ~/data/interim/table_snapshots_manifest.jsonl')
print("="*50)

# 매니페스트 앞부분(보정 전후 bbox 확인)
get_ipython().system('head -n 5 ~/data/interim/table_snapshots_manifest.jsonl')
print("="*50)


# -------------------
# 
# **[사후 필터링 내용]**
# 
# 
# - **행·열 격자 느낌 & 선(가로/세로 ruling) 밀도 & 면적 비율**로 “표 같은지” 판별
# - 통과한 것만 ``~/data/interim/table_snapshots_filtered/``로 복사
# - ``table_snapshots_manifest.filtered.jsonl`` 생성 + 간단 리포트

# In[ ]:


# 사후 필터링

# -*- coding: utf-8 -*-
"""
이미 생성된 스냅샷(이미지) 중에서 '표 같은 것'만 골라
~/data/interim/table_snapshots_filtered/ 로 복사하고
~/data/interim/table_snapshots_manifest.filtered.jsonl 매니페스트를 만든다.

판별 기준(기본값, 문서에 따라 조정):
- 면적 비율: 0.04 ~ 0.70  (페이지 전체 대비 bbox 면적)
- 선(가로/세로) 개수: 각각 >= 2  (pdfplumber page.lines 사용)
- 텍스트 행/열 그룹 수: rows >= 6, cols >= 4  (단어의 y/x 위치를 클러스터)

세 조건 중
- 'lines 전략'에서 온 후보는 [선 조건 OR 행/열 조건] 충족 시 통과
- 'text 전략'에서 온 후보는 [행/열 조건] 반드시 충족 (lines가 약하므로)
"""

from pathlib import Path
import pdfplumber, json, re, math, shutil
from collections import defaultdict

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
SRC_MANIFEST = INTERIM / "table_snapshots_manifest.jsonl"
DST_DIR = INTERIM / "table_snapshots_filtered"
DST_MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"

# ---- 튜닝 파라미터 ----
AREA_FRAC_MIN = 0.03   # 너무 작으면 잡음
AREA_FRAC_MAX = 0.50   # 너무 크면 '본문 전체'일 가능성(본문 대형 박스 컷 0.50~0.55 사이에서 조정하기)
MIN_HLINES = 2         # 최소 가로선 개수
MIN_VLINES = 2         # 최소 세로선 개수
MIN_ROWS = 6           # 최소 행 수(텍스트 기반) ==> 행문턱 up으로 조절
MIN_COLS = 4           # 최소 열 수(텍스트 기반) ==> 열문턱 up으로 조절
ROW_TOL = 4.0          # y(세로) 그룹핑 허용 오차(px)
COL_TOL = 6.0          # x(가로) 그룹핑 허용 오차(px)

# 텍스트 기반 보조 조건 (추가)
NEGATIVE_KWS = [
    "목 차","목차","제 안 요 청 서","제안요청서","사업 개요","사업개요","서론","표지",
    # ▼ 폼/헤더 패턴
    "개정이력","개정 번호","개정일자","개정 사유","문서번호","발행일","페이지",
    "결재","검토","승인","확인","작성","배포","문서 등급","문서 명","문서명"
]
POSITIVE_KWS = ["구분", "항목", "합계", "단위", "금액", "수량", "계", "구성", "평균", "최대", "최소", "비용", "예산", "요구", "요건"]


def is_form_like(page, bbox, preview):
    # 페이지 상단 30% 안쪽에 시작하고, 빈 셀(공백) 위주면 폼으로 간주
    px0, py0, px1, py1 = page.bbox
    x0, y0, x1, y1 = bbox
    starts_top = (y0 - py0) / (py1 - py0) < 0.30
    many_blanks = preview.count(" ") / max(1, len(preview)) > 0.6
    return starts_top and many_blanks


def area_fraction(bbox, page):
    x0, y0, x1, y1 = bbox
    pb = page.bbox
    pw, ph = pb[2]-pb[0], pb[3]-pb[1]
    bw, bh = max(0.0, x1-x0), max(0.0, y1-y0)
    if pw <= 0 or ph <= 0: return 0.0
    return (bw*bh) / (pw*ph)

def count_rulings_in_bbox(page, bbox):
    """가로/세로 선(라인) 개수 세기. 기울기 거의 0 → 가로, 거의 무한대 → 세로로 판정."""
    x0, y0, x1, y1 = bbox
    h = v = 0
    for ln in page.lines:
        lx0, ly0, lx1, ly1 = ln["x0"], ln["top"], ln["x1"], ln["bottom"]
        # bbox 내부에 충분히 겹치면 카운트
        if (min(lx0, lx1) >= x0 and max(lx0, lx1) <= x1 and
            min(ly0, ly1) >= y0 and max(ly0, ly1) <= y1):
            dx = abs(lx1 - lx0)
            dy = abs(ly1 - ly0)
            if dx >= dy * 5:   # 거의 수평
                h += 1
            elif dy >= dx * 5: # 거의 수직
                v += 1
    return h, v

def group_values(vals, tol):
    """1D 값 리스트를 tol 간격으로 군집화하여 그룹 수 반환(간단 버전)."""
    vals = sorted(vals)
    if not vals: return 0
    groups = 1
    cur = vals[0]
    for x in vals[1:]:
        if abs(x - cur) > tol:
            groups += 1
            cur = x
    return groups

def estimate_rows_cols(page, bbox):
    """bbox 내부 단어들의 y-중심, x-시작을 기준으로 '행/열' 개수 추정."""
    crop = page.within_bbox(bbox)
    words = crop.extract_words() or []
    if not words:
        return 0, 0
    y_centers = [ (w["top"] + w["bottom"]) / 2.0 for w in words ]
    x_starts  = [ w["x0"] for w in words ]
    rows = group_values(y_centers, ROW_TOL)
    cols = group_values(x_starts, COL_TOL)
    return rows, cols

def text_digit_ratio(s: str) -> float:
    if not s: return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s))

def has_keywords(s: str, kws) -> bool:
    # 대소문자/공백 정도 normalization
    s_norm = re.sub(r"\s+", " ", (s or "")).lower()
    # return any(k in s for k in kws)
    return any(((k or "").lower() in s_norm) for k in kws)


def is_table_like(page, rec):
    """표 판정: detect_strategy에 따라 기준 다르게 + 텍스트 보조 신호."""
    bbox = rec["bbox"] if "bbox" in rec else rec.get("bbox_raw")
    strategy = rec.get("detect_strategy", "none")
    preview = rec.get("text_preview", "")  # 매니페스트에 이미 있음
    
    # 음성 키워드 컷 : 폼/헤더 컷
    # if has_keywords(preview, NEGATIVE_KWS):
    #     return False
    if has_keywords(preview, NEGATIVE_KWS) or is_form_like(page, bbox, preview):
        return False
    
    # 면적 필터
    af = area_fraction(bbox, page)
    if not (AREA_FRAC_MIN <= af <= AREA_FRAC_MAX):
        return False

    # 선 개수
    h, v = count_rulings_in_bbox(page, bbox)
    # 행/열 개수(텍스트)
    rows, cols = estimate_rows_cols(page, bbox)
    # 숫자 비율(표는 숫자 많을 확률 큼)
    dr = text_digit_ratio(preview)

    
    # 전략별 기준
    if strategy == "lines":
        # 선 기반이면 [선 조건] 강하게 or [텍스트 조건+숫자] 보조
        if (h >= MIN_HLINES and v >= MIN_VLINES):
            return True
        return (rows >= MIN_ROWS and cols >= MIN_COLS and dr >= 0.08)
    elif strategy == "text":
        # 텍스트 기반이면 본문 오검 많아서 더 엄격:
        # - 행/열 충분 + (가로/세로 선이 최소 1개 이상이거나 숫자비율/양성키워드 중 하나 충족)
        if rows >= MIN_ROWS and cols >= MIN_COLS:
            if (h + v) >= 1:
                return True
            if dr >= 0.10:
                return True
            if has_keywords(preview, POSITIVE_KWS):
                return True
        return False
    else:
        # 안전판: 두 조건 모두 충족
        return (h >= MIN_HLINES and v >= MIN_VLINES) and (rows >= MIN_ROWS and cols >= MIN_COLS)


def run():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    # PDF 캐시: 같은 파일을 여러 번 열지 않도록
    pdf_cache = {}

    kept = 0
    total = 0

    with open(SRC_MANIFEST, "r", encoding="utf-8") as f, \
         open(DST_MANIFEST, "w", encoding="utf-8") as w:
        for line in f:
            total += 1
            rec = json.loads(line)
            pdf_path = Path(rec["pdf_path"])
            page_no = int(rec["page"])

            # pdf 열기(캐시)
            if pdf_path not in pdf_cache:
                pdf_cache[pdf_path] = pdfplumber.open(pdf_path)
            pdf = pdf_cache[pdf_path]

            page = pdf.pages[page_no - 1]

            if not is_table_like(page, rec):
                continue

            # 통과 → 이미지 복사 & 레코드 기록
            src_img = Path(rec["image_path"])
            dst_img = DST_DIR / pdf_path.stem / src_img.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                rec["image_path"] = str(dst_img)  # 경로 갱신
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    # 열어둔 PDF 닫기
    for pdf in pdf_cache.values():
        pdf.close()

    print(f"[REPORT] snapshots total={total}, kept={kept}, kept_ratio={kept/max(1,total):.2%}")
    print(f"[OUT] dir={DST_DIR}")
    print(f"[OUT] manifest={DST_MANIFEST}")

if __name__ == "__main__":
    run()


# #### 2-1) QC 리포트: 정밀도 확인

# **[진짜 표 커버리지/누락 확인]**
# 
# - 스냅샷(이미지) vs PDF 텍스트
# - QC(검증) 코드 실행 -> CSV로 저장
#     - 각 스냅샷에 대한
#     - bbox 영역 텍스트 글자수/ 페이지 전체 글자수
#     - 숫자 비율, 추정 행/열 개수, 라인 수(가로/세로)
#     - OCR 필요 추정/텍스트 표 추정 플래그
# 

# In[ ]:


# QC 스크립트

# -*- coding: utf-8 -*-
"""
table_snapshots_filtered(84장)과 PDF 원문을 대조하는 QC 리포트 생성
출력: ~/data/interim/table_snapshots_qc_report.csv
"""

from pathlib import Path
import pdfplumber, json, re, math
import pandas as pd

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"  # 필터링된 매니페스트
OUT_CSV = INTERIM / "table_snapshots_qc_report.csv"

# ------- 유틸 (필터 스크립트의 함수 재사용) -------
def area_fraction(bbox, page):
    x0, y0, x1, y1 = bbox
    pb = page.bbox
    pw, ph = pb[2]-pb[0], pb[3]-pb[1]
    bw, bh = max(0.0, x1-x0), max(0.0, y1-y0)
    if pw <= 0 or ph <= 0: return 0.0
    return (bw*bh) / (pw*ph)

def count_rulings_in_bbox(page, bbox):
    x0, y0, x1, y1 = bbox
    h = v = 0
    for ln in page.lines:
        lx0, ly0, lx1, ly1 = ln["x0"], ln["top"], ln["x1"], ln["bottom"]
        if (min(lx0, lx1) >= x0 and max(lx0, lx1) <= x1 and
            min(ly0, ly1) >= y0 and max(ly0, ly1) <= y1):
            dx = abs(lx1 - lx0)
            dy = abs(ly1 - ly0)
            if dx >= dy * 5:
                h += 1
            elif dy >= dx * 5:
                v += 1
    return h, v

def group_values(vals, tol):
    vals = sorted(vals)
    if not vals: return 0
    groups = 1
    cur = vals[0]
    for x in vals[1:]:
        if abs(x - cur) > tol:
            groups += 1
            cur = x
    return groups

ROW_TOL = 4.0
COL_TOL = 6.0

def estimate_rows_cols(page, bbox):
    crop = page.within_bbox(bbox)
    words = crop.extract_words() or []
    if not words:
        return 0, 0
    y_centers = [ (w["top"] + w["bottom"]) / 2.0 for w in words ]
    x_starts  = [ w["x0"] for w in words ]
    rows = group_values(y_centers, ROW_TOL)
    cols = group_values(x_starts, COL_TOL)
    return rows, cols

def text_digit_ratio(s: str) -> float:
    if not s: return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s))

def token_overlap_ratio(a: str, b: str) -> float:
    """bbox 텍스트가 페이지 텍스트에 어느 정도 포함되는지(자카드 유사도 기반 근사)."""
    tok = lambda t: set(re.findall(r"\w+", (t or "").lower()))
    A, B = tok(a), tok(b)
    if not A: return 0.0
    inter = len(A & B)
    return inter / len(A)

# ------- 메인 -------
rows = []
pdf_cache = {}

with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        pdf_path = Path(rec["pdf_path"])
        pno = int(rec["page"])
        bbox = tuple(rec.get("bbox") or rec.get("bbox_raw"))
        strategy = rec.get("detect_strategy", "unknown")
        preview = rec.get("text_preview", "")
        img_path = rec["image_path"]

        # PDF 캐시 열기
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = pdfplumber.open(pdf_path)
        pdf = pdf_cache[pdf_path]
        page = pdf.pages[pno - 1]

        # 텍스트(페이지/영역)
        page_text = (page.extract_text() or "").strip()
        crop = page.crop(bbox)
        bbox_text = (crop.extract_text() or "").strip()

        # 지표 계산
        af = area_fraction(bbox, page)
        h, v = count_rulings_in_bbox(page, bbox)
        est_rows, est_cols = estimate_rows_cols(page, bbox)
        dr = text_digit_ratio(bbox_text or preview)
        bbox_chars = len(bbox_text)
        page_chars = len(page_text)
        coverage = bbox_chars / max(1, page_chars)
        overlap = token_overlap_ratio(bbox_text, page_text)

        # 간단 분류
        # likely_image_table = (bbox_chars < 20) and ((h + v) >= 1 or strategy == "lines")
        # likely_text_table  = (est_rows >= 5 and est_cols >= 3) or (dr >= 0.10)
        # needs_ocr = likely_image_table and bbox_chars < 10  # 아주 텍스트가 없으면 OCR 후보
        likely_image_table = ((h + v) >= 3) and (bbox_chars < 30 or coverage < 0.02 or overlap < 0.2)
        # has_grid: (h+v) >= 3 --> 격자(선)느낌 충분함 의미
        # sparse text: bbox_chars < 30 or coverage < 0.02 or token_overlap < 0.2 
            # ---> 텍스트가 거의 없음/페이지 본문과 안겹침
        # 이 조건 만족해서 ocr 후보로 넣음
        likely_text_table  = (est_rows >= 5 and est_cols >= 3) or (dr >= 0.10)
        needs_ocr = likely_image_table

        
        rows.append({
            "pdf": pdf_path.name,
            "page": pno,
            "strategy": strategy,
            "area_frac": round(af, 3),
            "h_lines": h, "v_lines": v,
            "est_rows": est_rows, "est_cols": est_cols,
            "digits_ratio": round(dr, 3),
            "bbox_chars": bbox_chars, "page_chars": page_chars,
            "bbox_to_page_text_ratio": round(coverage, 3),
            "token_overlap": round(overlap, 3),
            "likely_image_table": int(likely_image_table),
            "likely_text_table": int(likely_text_table),
            "needs_ocr": int(needs_ocr),
            "image_path": img_path,
            "text_preview": (bbox_text[:140] or preview[:140])
        })

# close pdfs
for pdf in pdf_cache.values():
    pdf.close()

df = pd.DataFrame(rows).sort_values(
    ["pdf","page","strategy","area_frac","bbox_to_page_text_ratio"],
    ascending=[True, True, True, False, False]
)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("[OK] QC report saved:", OUT_CSV)
print("rows:", len(df))
print("="*70)


# 상위 몇 개 확인(의심 케이스)
print("\nTop OCR candidates (needs_ocr=1):")
print(df[df["needs_ocr"]==1].head(10)[["pdf","page","strategy","h_lines","v_lines","bbox_chars","image_path"]])
print("="*70)

print("\nTop text tables (likely_text_table=1):")
print(df[df["likely_text_table"]==1].head(10)[["pdf","page","est_rows","est_cols","digits_ratio","image_path"]])
print("="*70)

# 후보 리스트 파일 바로 만들기
OCR_TXT = INTERIM / "table_snapshots_ocr_candidates.txt"
df[df["needs_ocr"]==1]["image_path"].to_csv(OCR_TXT, index=False, header=False)
print("[OK] OCR candidates saved:", OCR_TXT, "count=", int((df["needs_ocr"]==1).sum()))



# #### 2-2) 누락 감사(gap audit)

# - 떨어뜨린 892장 중에 진짜 표가 끼었는지 재현율(recall)확인 필요
# - 떨어진 스냅샷 중 강한 후보만 재검

# -------
# - 원본 매니페스트(966개)와 필터 결과(74개)를 대조해서
# - **왜 떨어졌는지(룰별 사유)**를 붙이고,
# - 진짜 표일 가능성이 높은데 떨어진 후보만 모아 리뷰 폴더로 복사 + CSV 리포트를 만듦

# In[ ]:


# -*- coding: utf-8 -*-
"""
원본(table_snapshots_manifest.jsonl) vs 필터(table_snapshots_manifest.filtered.jsonl) 대조
- 각 후보의 지표/룰 통과여부/탈락사유 부여
- '진짜 표일 가능성이 높은데 탈락'한 것만 review 폴더로 복사
출력:
  ~/data/interim/table_audit_detail.csv
  ~/data/interim/table_audit_dropped_strong.csv
  ~/data/interim/table_snapshots_review/  (이미지)
  콘솔 요약: 사유별 분포
"""

from pathlib import Path
import pdfplumber, json, re, shutil
import pandas as pd
from collections import Counter

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
SRC_MANIFEST = INTERIM / "table_snapshots_manifest.jsonl"                # 966개
FIL_MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"       # 74개
OUT_DETAIL   = INTERIM / "table_audit_detail.csv"
OUT_DROPPED  = INTERIM / "table_audit_dropped_strong.csv"
REVIEW_DIR   = INTERIM / "table_snapshots_review"

# ==== 기존 필터의 설정과 함수들 (동일 기준 사용) ====
AREA_FRAC_MIN = 0.03
AREA_FRAC_MAX = 0.50
MIN_HLINES = 2
MIN_VLINES = 2
MIN_ROWS = 6
MIN_COLS = 4
ROW_TOL = 4.0
COL_TOL = 6.0

NEGATIVE_KWS = [
    "목 차","목차","제 안 요 청 서","제안요청서","사업 개요","사업개요","서론","표지",
    "개정이력","개정 번호","개정일자","개정 사유","문서번호","발행일","페이지",
    "결재","검토","승인","확인","작성","배포","문서 등급","문서 명","문서명"
]
POSITIVE_KWS = ["구분","항목","합계","단위","금액","수량","계","구성","평균","최대","최소","비용","예산","요구","요건"]

def has_keywords(s: str, kws) -> bool:
    s_norm = re.sub(r"\s+", " ", (s or "")).lower()
    return any(((k or "").lower() in s_norm) for k in kws)

def is_form_like(page, bbox, preview):
    px0, py0, px1, py1 = page.bbox
    x0, y0, x1, y1 = bbox
    starts_top = (y0 - py0) / (py1 - py0) < 0.30
    many_blanks = preview.count(" ") / max(1, len(preview)) > 0.6
    return starts_top and many_blanks

def area_fraction(bbox, page):
    x0, y0, x1, y1 = bbox
    pb = page.bbox
    pw, ph = pb[2]-pb[0], pb[3]-pb[1]
    bw, bh = max(0.0, x1-x0), max(0.0, y1-y0)
    if pw <= 0 or ph <= 0: return 0.0
    return (bw*bh) / (pw*ph)

def count_rulings_in_bbox(page, bbox):
    x0, y0, x1, y1 = bbox
    h = v = 0
    for ln in page.lines:
        lx0, ly0, lx1, ly1 = ln["x0"], ln["top"], ln["x1"], ln["bottom"]
        if (min(lx0, lx1) >= x0 and max(lx0, lx1) <= x1 and
            min(ly0, ly1) >= y0 and max(ly0, ly1) <= y1):
            dx = abs(lx1 - lx0); dy = abs(ly1 - ly0)
            if dx >= dy * 5: h += 1
            elif dy >= dx * 5: v += 1
    return h, v

def group_values(vals, tol):
    vals = sorted(vals)
    if not vals: return 0
    groups, cur = 1, vals[0]
    for x in vals[1:]:
        if abs(x - cur) > tol:
            groups += 1; cur = x
    return groups

def estimate_rows_cols(page, bbox):
    crop = page.within_bbox(bbox)
    words = crop.extract_words() or []
    if not words: return 0, 0
    y_centers = [ (w["top"] + w["bottom"]) / 2.0 for w in words ]
    x_starts  = [ w["x0"] for w in words ]
    return group_values(y_centers, ROW_TOL), group_values(x_starts, COL_TOL)

def text_digit_ratio(s: str) -> float:
    if not s: return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s))

# ==== 준비: kept(필터 통과) 식별 ====
kept_basenames = set()
with open(FIL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        kept_basenames.add(Path(r["image_path"]).name)

# ==== 본 감사 ====
pdf_cache = {}
rows = []
reason_counter = Counter()

with open(SRC_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        pdf_path = Path(rec["pdf_path"])
        page_no  = int(rec["page"])
        bbox     = tuple(rec.get("bbox") or rec.get("bbox_raw"))
        strat    = rec.get("detect_strategy", "none")
        preview  = rec.get("text_preview", "") or ""

        # PDF 열기 캐시
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = pdfplumber.open(pdf_path)
        page = pdf_cache[pdf_path].pages[page_no - 1]

        # 지표
        af = area_fraction(bbox, page)
        h, v = count_rulings_in_bbox(page, bbox)
        rr, cc = estimate_rows_cols(page, bbox)
        dr = text_digit_ratio(preview)

        # 규칙별 플래그
        f_negkw   = has_keywords(preview, NEGATIVE_KWS)
        f_form    = is_form_like(page, bbox, preview)
        f_area_ok = (AREA_FRAC_MIN <= af <= AREA_FRAC_MAX)
        f_lines   = (h >= MIN_HLINES and v >= MIN_VLINES)
        f_text    = (rr >= MIN_ROWS and cc >= MIN_COLS)
        f_poskw   = has_keywords(preview, POSITIVE_KWS)

        # 필터 통과여부(실제)
        basename = Path(rec["image_path"]).name
        kept = basename in kept_basenames

        # 탈락사유(우선순위)
        reason = "kept"
        if not kept:
            if f_negkw: reason = "neg_keyword"
            elif f_form: reason = "form_like"
            elif not f_area_ok: reason = "area_out"
            elif strat == "lines" and not (f_lines or (f_text and dr >= 0.08)):
                reason = "weak_lines_and_text"
            elif strat == "text" and not (f_text and ((h+v)>=1 or dr>=0.10 or f_poskw)):
                reason = "weak_text_structure"
            else:
                reason = "other"
            reason_counter[reason] += 1

        rows.append({
            "pdf": pdf_path.name, "page": page_no, "strategy": strat,
            "image": basename, "kept": int(kept), "reason": reason,
            "area_frac": round(af,3), "h_lines": h, "v_lines": v,
            "rows": rr, "cols": cc, "digits_ratio": round(dr,3),
            "preview": preview[:140]
        })

# close
for pdf in pdf_cache.values(): pdf.close()

df = pd.DataFrame(rows)
df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
print("[OUT] detail:", OUT_DETAIL, "rows=", len(df))

# ==== 사람이 꼭 봐야 할 '강한 신호인데 탈락'만 추출 ====
# 강한 신호: (rows/cols 충분 & digits>=0.08) OR (격자선 합>=3) AND area 0.04~0.65
strong = (
    (((df["rows"]>=MIN_ROWS) & (df["cols"]>=MIN_COLS) & (df["digits_ratio"]>=0.08)) |
     ((df["h_lines"]+df["v_lines"])>=3))
    & (df["area_frac"].between(0.04, 0.65))
)

dropped_strong = df[(df["kept"]==0) & strong & (~df["reason"].isin(["neg_keyword","form_like"]))].copy()
dropped_strong.to_csv(OUT_DROPPED, index=False, encoding="utf-8-sig")
print("[OUT] dropped_strong:", OUT_DROPPED, "rows=", len(dropped_strong))

# 리뷰 썸네일 복사
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
import json
with open(SRC_MANIFEST, "r", encoding="utf-8") as f:
    m = {Path(json.loads(x)["image_path"]).name: json.loads(x)["image_path"] for x in f}

copied = 0
for name in dropped_strong["image"]:
    src = Path(m[name])
    if src.exists():
        dst = REVIEW_DIR / src.name
        shutil.copy2(src, dst); copied += 1
print(f"[COPY] review images -> {REVIEW_DIR}  copied={copied}")

# 사유 분포 요약
print("\n[SUMMARY] drop reasons:")
for k,v in reason_counter.most_common():
    print(f"  - {k:20s}: {v}")


# In[ ]:


# 샘플 그리드 뷰어 (리뷰용 샘플 이미지 확인)

# -*- coding: utf-8 -*-
# Dropped Strong 샘플 육안 확인용 그리드 뷰어

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import textwrap
import random

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
CSV = INTERIM / "table_audit_dropped_strong.csv"   # 앞서 생성한 CSV
REVIEW_DIR = INTERIM / "table_snapshots_review"    # 앞서 복사한 이미지들

# --- 데이터 로드 & 경로 결합
df = pd.read_csv(CSV)
df["img_path"] = df["image"].apply(lambda x: str((REVIEW_DIR / x)))

# 정렬 편의용 스코어
df["score_lines"]   = df["h_lines"] + df["v_lines"]
df["score_struct"]  = df["rows"] + df["cols"]
df["score_numeric"] = df["digits_ratio"]

def _short(s, width=50):
    s = str(s)
    return textwrap.shorten(s, width=width, placeholder="…")

def show_samples(reason=None, sort_by="score_lines", ascending=False, n=12, seed=42, ncols=3, max_w=1200):
    """
    reason: 'weak_lines_and_text' / 'area_out' / 'neg_keyword' / 'weak_text_structure' 등
    sort_by: 'score_lines' | 'score_struct' | 'score_numeric' | 'area_frac' 등 df 컬럼
    """
    random.seed(seed)

    sub = df.copy()
    if reason:
        sub = sub[sub["reason"] == reason]
    if sort_by and sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=ascending)

    # 상위 n개 선택 (부족하면 랜덤 샘플)
    if len(sub) >= n:
        view = sub.head(n)
    else:
        view = sub.sample(min(n, len(sub)), random_state=seed)

    nrows = int(np.ceil(len(view) / ncols))
    plt.figure(figsize=(ncols*5.5, nrows*4.6))

    for i, (_, row) in enumerate(view.iterrows(), start=1):
        ax = plt.subplot(nrows, ncols, i)
        p = Path(row["img_path"])
        if not p.exists():
            ax.text(0.5, 0.5, f"NOT FOUND\n{p.name}", ha="center", va="center")
            ax.axis("off"); continue

        # 이미지 로드 & 리사이즈(너비 제한)
        im = Image.open(p)
        if im.width > max_w:
            h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, h), Image.BILINEAR)

        ax.imshow(im)
        ax.axis("off")

        title = (
            f"{_short(row['pdf'], 40)}  |  p.{int(row['page'])}  |  {row['strategy']}\n"
            f"reason={row['reason']}  area={row['area_frac']:.2f}  "
            f"lines(h/v)={row['h_lines']}/{row['v_lines']}  "
            f"rows/cols={row['rows']}/{row['cols']}  "
            f"digits={row['digits_ratio']:.2f}"
        )
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()

print("[READY] 사용 예시 ↓")
print("show_samples(n=12)  # 기본: 라인 스코어 상위 12장")
print("show_samples(reason='weak_lines_and_text', n=12)")
print("show_samples(reason='area_out', sort_by='area_frac', ascending=False, n=12)")
print("show_samples(sort_by='score_struct', n=12)")


# In[ ]:


# matpltlib 한글폰트 설정(자동탐지) + 마이너스 기호 처리

# Matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import os

candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
chosen = next((n for n in candidates if n in available), None)

if chosen is None:
    print("[WARN] 한글 폰트를 찾지 못했습니다. 아래 중 하나를 설치 후 커널 재시작하세요.")
    print("  Ubuntu/Debian:  sudo apt-get update && sudo apt-get install -y fonts-nanum fonts-noto-cjk")
    print("  RHEL/CentOS:    sudo yum install google-noto-sans-cjk-ttc")
    print("설치 뒤 캐시 삭제(선택): rm -rf ~/.cache/matplotlib")
else:
    rcParams["font.family"] = chosen
    rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
    print(f"[OK] Matplotlib 한글 폰트 적용: {chosen}")


# #### 2-3) 정밀도,재현율 검증 이후: 74장 매니페스트 →  csv로 변환

# - 표는 구조화 정보라서 본문과 따로 보관해야 검색·답변에서 강점이 분명함(배점표/산식/요율 등).
# 
# 
# - ``표→CSV``로 먼저 구조화하면, 나중에 Markdown 텍스트로도 만들고(임베딩용), 원시 CSV도 그대로 보존(감사/재처리용) 가능.
# 
# 
# 
# - 하나의 JSONL 스키마로 PDF 본문/표/HWP를 통합하면, 다운스트림(청크·임베딩·인덱싱/RAG)에서 구현이 단순해짐.

# - 입력: ``~/data/interim/table_snapshots_manifest.filtered.jsonl`` (생성한 74장 매니페스트)
# - 처리: 원본 PDF를 다시 열어 bbox로 ``crop`` → ``pdfplumber table`` 추출
# - 출력:
#     - CSV: ``~/data/interim/table_csv/<pdf_stem>/pXXX_tYY.csv``
# - 실패 로그: 터미널 출력

# In[ ]:


# -*- coding: utf-8 -*-
"""
필터링된 표 스냅샷(74장)을 원본 PDF에서 bbox로 재크롭하여 표 추출 → CSV 저장
출력: ~/data/interim/table_csv/<pdf_stem>/pXXX_tYY.csv
"""

from pathlib import Path
import pdfplumber, json, csv, re
from collections import defaultdict

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"
CSV_DIR  = INTERIM / "table_csv"

# table 설정(2-pass: lines → 실패 시 text)
TS_LINES = dict(
    vertical_strategy="lines", horizontal_strategy="lines",
    snap_tolerance=3, join_tolerance=3, edge_min_length=3,
    min_words_vertical=1, min_words_horizontal=1
)
TS_TEXT = dict(
    vertical_strategy="text",  horizontal_strategy="text",
    snap_tolerance=3, join_tolerance=3, edge_min_length=3,
    min_words_vertical=1, min_words_horizontal=1
)

def clean_cell(s):
    if s is None: return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # 폭이 다른 행도 깨지지 않게 최대 컬럼수 맞추기
    max_cols = max((len(r) for r in rows), default=0)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for r in rows:
            rr = [clean_cell(x) for x in r] + [""] * (max_cols - len(r))
            w.writerow(rr)

def pick_best(tables):
    """여러 표가 오면 '비어있지 않은 셀'이 많은 것을 선택."""
    best = None
    best_score = -1
    for t in tables or []:
        filled = sum(1 for r in t for c in r if c not in (None, "", " "))
        if filled > best_score:
            best, best_score = t, filled
    return best

def extract_table_from_bbox(page, bbox, strategy_hint="lines"):
    crop = page.crop(bbox)
    # 1-pass: 힌트 기반
    ts1 = TS_LINES if strategy_hint == "lines" else TS_TEXT
    t1 = crop.extract_tables(table_settings=ts1) or []
    best = pick_best(t1)
    if best: return best, "pass1:" + strategy_hint

    # 2-pass: 반대 전략
    ts2 = TS_TEXT if strategy_hint == "lines" else TS_LINES
    t2 = crop.extract_tables(table_settings=ts2) or []
    best = pick_best(t2)
    if best: return best, "pass2:" + ("text" if strategy_hint == "lines" else "lines")

    # 3-pass(옵션): 직선 탐색 기본값
    t3 = crop.extract_tables() or []
    best = pick_best(t3)
    if best: return best, "pass3:auto"

    return None, "fail"

def run():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    pdf_cache = {}
    ok = fail = 0
    by_pdf = defaultdict(int)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pdf_path = Path(rec["pdf_path"])
            page_no  = int(rec["page"])
            bbox     = tuple(rec.get("bbox") or rec.get("bbox_raw"))
            strat    = rec.get("detect_strategy", "lines")
            img_path = Path(rec["image_path"])

            if pdf_path not in pdf_cache:
                pdf_cache[pdf_path] = pdfplumber.open(pdf_path)
            page = pdf_cache[pdf_path].pages[page_no - 1]

            table, how = extract_table_from_bbox(page, bbox, "lines" if strat=="lines" else "text")
            out_csv = CSV_DIR / pdf_path.stem / f"{img_path.stem}.csv"

            if table:
                save_csv(table, out_csv)
                ok += 1; by_pdf[pdf_path.name] += 1
            else:
                fail += 1
                print(f"[WARN] table extract fail: {pdf_path.name} p{page_no} ({how}) -> {out_csv}")

    for pdf in pdf_cache.values():
        pdf.close()

    print(f"[OK] tables saved={ok}, failed={fail}")
    print("[BY PDF]")
    for k,v in sorted(by_pdf.items()):
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    run()


# ### 3) PDF 클린 JSONL, 테이블 스냅샷 정합성 체크

# In[ ]:


# -*- coding: utf-8 -*-
# 테이블 매니페스트(74장)에 등장한 (문서, 페이지)가
# 기존 '클린 페이지 JSONL'에 정확히 존재하는지 점검

from pathlib import Path
import json, re
from collections import defaultdict

BASE = Path.home() / "data"
INTERIM = BASE / "interim"

# ① 테이블 매니페스트(필터 후, 74줄)
FIL_MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"
# ② 너희가 예전에 만든 "클린 페이지 JSONL" 경로로 맞춰줘
CLEAN_PAGES = INTERIM / "pages_pdf.cleaned.jsonl"

def norm_key(s: str) -> str:
    """파일명/문서ID 정규화: 소문자, 확장자 제거, 공백/괄호/특수기호 축약"""
    s = s or ""
    s = s.lower()
    s = re.sub(r"\.[a-z0-9]{1,5}$", "", s)          # 확장자 제거
    s = re.sub(r"[\[\]\(\)【】<>]", " ", s)         # 괄호류 제거
    s = re.sub(r"[_\-·•]+", " ", s)                # 구분기호 통합
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 1) 테이블 측 키셋
need = set()
with open(FIL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pdf_path = Path(r["pdf_path"])
        key = norm_key(pdf_path.stem)
        need.add((key, int(r["page"])))

# 2) 클린 페이지 JSONL 인덱스 만들기
have = set()
with open(CLEAN_PAGES, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        # 문서 식별자 후보: source_path 또는 doc_id 또는 filename
        if "source_path" in r:
            k = norm_key(Path(r["source_path"]).stem)
        elif "doc_id" in r:
            k = norm_key(r["doc_id"])
        elif "filename" in r:
            k = norm_key(Path(r["filename"]).stem)
        else:
            # 마지막 수단: title/name 같은 필드가 있다면 추가
            k = norm_key(r.get("title",""))
        have.add((k, int(r["page"])))

missing = sorted(list(need - have))   # 테이블이 있는데 클린 JSONL엔 없는 페이지
extra   = sorted(list(have - need))   # 클린 JSONL엔 있지만 테이블엔 없는 페이지(정상일 수 있음)

print("[CHECK] tables_needed:", len(need))
print("[CHECK] pages_in_clean:", len(have))
print("[CHECK] missing_pairs:", len(missing))
for x in missing[:10]:
    print("  MISSING:", x)


# ### 4) 기존 클린 JSONL + 74개 표 CSV  → ``rep_pdf.jsonl``

# In[ ]:


# -*- coding: utf-8 -*-
# 기존 'pages_pdf.cleaned.jsonl' + 표 CSV 메타를 합쳐 rep_pdf.jsonl 생성

from pathlib import Path
import json, csv, re

BASE = Path.home() / "data"
INTERIM = BASE / "interim"
CLEAN_PAGES = INTERIM / "pages_pdf.cleaned.jsonl"
FIL_MANIFEST = INTERIM / "table_snapshots_manifest.filtered.jsonl"
CSV_DIR = INTERIM / "table_csv"
OUT_JSONL = INTERIM / "rep_pdf.jsonl"

def norm_key(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\.[a-z0-9]{1,5}$", "", s)
    s = re.sub(r"[\[\]\(\)【】<>]", " ", s)
    s = re.sub(r"[_\-·•]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def csv_to_markdown(csv_path: Path, max_rows=200):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.reader(f)):
            rows.append(r)
            if i+1 >= max_rows: break
    if not rows: return ""
    w = max(len(r) for r in rows)
    rows = [r + [""]*(w-len(r)) for r in rows]
    header = rows[0]; sep = ["---"]*w; body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)

# 1) 페이지 텍스트를 먼저 그대로 흘려 쓰기
with open(OUT_JSONL, "w", encoding="utf-8") as out:
    with open(CLEAN_PAGES, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # 키 표준화
            if "source_path" in rec:
                src = Path(rec["source_path"])
                doc_key = norm_key(src.stem)
                source_path = str(src)
            else:
                # source_path가 없으면 doc_id로 대체
                doc_key = norm_key(rec.get("doc_id",""))
                source_path = rec.get("source_path") or rec.get("filename") or rec.get("doc_id","")
            out_rec = {
                "doctype": "pdf",
                "source_path": source_path,
                "page": int(rec["page"]),
                "type": "page_text",
                "content": rec.get("text",""),
                "lang": rec.get("lang","ko")
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # 2) 표 레코드 추가
    with open(FIL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pdf_path = Path(r["pdf_path"])
            img_path = Path(r["image_path"])
            csv_path = CSV_DIR / pdf_path.stem / f"{img_path.stem}.csv"
            md = csv_to_markdown(csv_path) if csv_path.exists() else ""

            out_rec = {
                "doctype": "pdf",
                "source_path": str(pdf_path),
                "page": int(r["page"]),
                "type": "table",
                "bbox": r.get("bbox") or r.get("bbox_raw"),
                "image_path": str(img_path),
                "csv_path": str(csv_path),
                "content": md,   # 임베딩/검색용 텍스트 표현(Markdown)
                "lang": "ko",
                "meta": {
                    "detect_strategy": r.get("detect_strategy"),
                    "dpi": r.get("dpi"),
                    "text_preview": r.get("text_preview", "")
                }
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

print("[OK] rep_pdf.jsonl:", OUT_JSONL)


# **[확인]**
# 
# - 페이지 본문이 담긴 "클린 버전" 사용
# - 표는 CSV -> Markdown 붙여 동일 JSONL로 병합

# ## 3-2. HWP 파일 →  HTML/XML

# - html 텍스트 : pyhwp패키지 hwp5tml
# - xml 텍스트 : 수동 포맷변환 

# ### 디렉터리 구조 생성

# In[ ]:


# 1) 기본 디렉토리 생성
get_ipython().system('mkdir -p ~/data/raw/converted_2/html')
get_ipython().system('mkdir -p ~/data/raw/converted_2/xml')

# 2) 원본 HWP 위치 확인(예시)
#   - 예: ~/data/raw/files 안에 96개의 .hwp가 있다고 가정
#   - 실제 경로에 맞게 바꿔서 사용하세요.


# In[ ]:


# # 1) 필요한 툴 설치 : 터미널
# # (이미 myenv 활성화되어 있다고 가정)
# pip install -U pip
# pip install lxml olefile chardet
# pip install git+https://github.com/mete0r/pyhwp.git#egg=pyhwp


# ### 3-2-1. HWP -> HTML

# #### 1) html 텍스트 추출

# In[ ]:


# 경로설정

from pathlib import Path
import subprocess, shlex
import sys, textwrap, datetime

# 입력/출력/로그 경로 설정
IN_DIR   = Path("/home/spai0308/data/raw/files")
# 주석: .hwp 원본 폴더 (문서별 폴더가 있어도 rglob 로 전부 찾을 예정)

OUT_TXT  = Path("/home/spai0308/data/raw/converted/docs")
# 주석: .txt 출력 경로

OUT_HTML = Path("/home/spai0308/data/raw/converted/html")
# 주석: .html / .css 출력 경로

LOG_PATH = Path("/home/spai0308/data/interim/hwp_convert_errors.log")
# 주석: 변환 실패/에러 메시지 기록 파일

# 출력/로그 폴더 생성
OUT_TXT.mkdir(parents=True, exist_ok=True)
# 주석: docs 폴더가 없으면 생성

OUT_HTML.mkdir(parents=True, exist_ok=True)
# 주석: html 폴더가 없으면 생성

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
# 주석: 로그 폴더가 없으면 생성

print("환경 설정 완료:", IN_DIR, OUT_TXT, OUT_HTML, LOG_PATH, sep="\n- ")
# 주석: 설정값 확인용 출력


# In[ ]:


# 안전 실행 유틸: (1) 실행파일 찾기 → (2) runpy로 모듈 실행

import shutil, sys, io, contextlib, runpy, importlib
from pathlib import Path
import subprocess, datetime

def _safe_find_module(name: str) -> bool:
    """모듈 존재 여부를 안전하게 체크 (부모 import 실패해도 계속 진행)"""
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None
    except Exception:
        return False

def _run_module_as_script(module_name: str, argv: list[str]) -> tuple[int, str, str]:
    """
    module_name을 __main__처럼 실행(runpy)하고 stdout/stderr를 캡처해서 리턴.
    반환: (returncode, stdout, stderr)  — 성공이면 0, 실패면 1
    """
    old_argv = sys.argv[:]                 # 원래 argv 백업
    sys.argv = [module_name] + argv[:]     # 모듈이 argparse로 argv를 읽을 수 있게 설정
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return 0, stdout_buf.getvalue(), stderr_buf.getvalue()
    except SystemExit as e:
        # argparse가 sys.exit()를 호출할 수 있음 — 코드 보존
        code = int(e.code) if isinstance(e.code, int) else 1
        return code, stdout_buf.getvalue(), stderr_buf.getvalue()
    except Exception as e:
        return 1, stdout_buf.getvalue(), (stderr_buf.getvalue() + f"\n[run_module error] {e}")
    finally:
        sys.argv = old_argv

class Hwp5Runner:
    """
    hwp5txt/hwp5html을 실행파일 또는 모듈로 실행해주는 래퍼.
    - 먼저 실행파일(shutil.which) 시도
    - 실패 시 모듈 후보들을 runpy로 실행
    """
    def __init__(self, exe_name: str, module_candidates: list[str]):
        self.exe_path = shutil.which(exe_name)            # 예: 'hwp5txt' → '/usr/bin/hwp5txt'
        self.module = None
        if not self.exe_path:
            # 모듈 후보를 순서대로 탐색
            for m in module_candidates:
                if _safe_find_module(m):
                    self.module = m
                    break
        if (not self.exe_path) and (not self.module):
            raise FileNotFoundError(
                f"'{exe_name}' 실행 파일도 모듈도 찾지 못했습니다. "
                f"주피터 커널이 (myenv)와 같은지 확인하거나, pyhwp를 해당 커널에 설치하세요.\n"
                f"예) %pip install pyhwp"
            )

    def run(self, args: list[str]) -> tuple[int, str, str]:
        """
        args 예시 (hwp5txt): [<hwpfile>, '--output', <txt_out>]
        args 예시 (hwp5html): [<hwpfile>, '--html', '--output', <html_out>]
        """
        if self.exe_path:
            # 실행파일 경로로 subprocess 실행
            cmd = [self.exe_path] + args
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = p.communicate()
            return p.returncode, out, err
        else:
            # 모듈로 runpy 실행
            return _run_module_as_script(self.module, args)

def log_append(msg: str, log_path: Path):
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    line = f"[{ts}] {msg}\n"
    if log_path.exists():
        log_path.write_text(log_path.read_text(encoding="utf-8") + line, encoding="utf-8")
    else:
        log_path.write_text(line, encoding="utf-8")

print("안전 실행 유틸 로드 완료")


# In[ ]:


# 유틸 함수(외부 명령 실행, 안전 출력)

def run_cmd(cmd_list):
    # 외부 명령을 실행하고 (returncode, stdout, stderr)를 반환
    # 주석: subprocess.Popen 으로 커맨드를 실행해 결과와 에러를 받음
    p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err
    # 주석: 종료코드 0=성공, 아니면 실패

def log_append(msg: str):
    # 로그 파일에 한 줄 추가
    # 주석: 타임스탬프 포함해서 실패 이유 등을 남김
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n"
    LOG_PATH.write_text(LOG_PATH.read_text(encoding="utf-8") + line if LOG_PATH.exists() else line, encoding="utf-8")
    # 주석: 기존 로그가 있으면 이어 붙이고, 없으면 새로 씀


# In[ ]:


from pathlib import Path

# 경로 설정
IN_DIR   = Path("/home/spai0308/data/raw/files")                 
OUT_TXT  = Path("/home/spai0308/data/raw/converted/docs")        
OUT_HTML = Path("/home/spai0308/data/raw/converted/html")        
LOG_PATH = Path("/home/spai0308/data/interim/hwp_convert_errors.log")

OUT_TXT.mkdir(parents=True, exist_ok=True)
OUT_HTML.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# 러너 준비: 모듈 후보는 보편적으로 'hwp5.hwp5txt' / 'hwp5.hwp5html' 이 존재합니다.
txt_runner  = Hwp5Runner("hwp5txt",  ["hwp5.hwp5txt",  "hwp5.bin.hwp5txt"])
html_runner = Hwp5Runner("hwp5html", ["hwp5.hwp5html", "hwp5.bin.hwp5html"])

sample = next(IN_DIR.rglob("*.hwp"), None)
if sample is None:
    raise FileNotFoundError("IN_DIR 아래에서 .hwp를 못 찾았습니다. 경로/파일을 확인하세요.")

base = sample.stem
txt_out  = OUT_TXT  / f"{base}.txt"
html_out = OUT_HTML / f"{base}.html"
css_out  = OUT_HTML / f"{base}.css"

# 1) .hwp → .txt
rc, out, err = txt_runner.run([str(sample), "--output", str(txt_out)])
print("TXT:", rc, txt_out)
if rc != 0:
    log_append(f"[TXT FAIL] {sample} :: {err.strip()}", LOG_PATH)
    print(err)

# 2) .hwp → .html
rc, out, err = html_runner.run([str(sample), "--html", "--output", str(html_out)])
print("HTML:", rc, html_out)
if rc != 0:
    log_append(f"[HTML FAIL] {sample} :: {err.strip()}", LOG_PATH)
    print(err)

# 3) (선택) .hwp → .css
rc, out, err = html_runner.run([str(sample), "--css", "--output", str(css_out)])
print("CSS:", rc, css_out)
if rc != 0:
    log_append(f"[CSS FAIL] {sample} :: {err.strip()}", LOG_PATH)
    print(err)


# In[ ]:


import subprocess

IN_DIR   = Path("/home/spai0308/data/raw/files")                    # 주석: .hwp 원본
OUT_TXT  = Path("/home/spai0308/data/raw/converted/docs")           # 주석: txt 출력
OUT_HTML = Path("/home/spai0308/data/raw/converted/html")           # 주석: html/css 출력
OUT_TXT.mkdir(parents=True, exist_ok=True); OUT_HTML.mkdir(parents=True, exist_ok=True)

sample = next(IN_DIR.rglob("*.hwp"), None)
assert sample is not None, "HWP 원본이 없습니다."

base = sample.stem
TEST_TXT  = OUT_TXT  / f"{base}.__test__.txt"
TEST_HTML = OUT_HTML / f"{base}.__test__.html"
TEST_CSS  = OUT_HTML / f"{base}.__test__.css"

# 잔여물 제거
for f in (TEST_TXT, TEST_HTML, TEST_CSS):
    if f.exists(): f.unlink()

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

# 1) .hwp → .txt
rc, out, err = run([str(HWP5TXT), str(sample), "--output", str(TEST_TXT)])
print("[TXT rc]", rc, "| exists:", TEST_TXT.exists(), "| size:", (TEST_TXT.stat().st_size if TEST_TXT.exists() else -1))
if rc!=0: print("[stderr]\n", err[:1000])

# 2) .hwp → .html
rc, out, err = run([str(HWP5HTML), str(sample), "--html", "--output", str(TEST_HTML)])
print("[HTML rc]", rc, "| exists:", TEST_HTML.exists(), "| size:", (TEST_HTML.stat().st_size if TEST_HTML.exists() else -1))
if rc!=0: print("[stderr]\n", err[:1000])

# 3) .hwp → .css (선택)
rc, out, err = run([str(HWP5HTML), str(sample), "--css", "--output", str(TEST_CSS)])
print("[CSS rc]", rc, "| exists:", TEST_CSS.exists(), "| size:", (TEST_CSS.stat().st_size if TEST_CSS.exists() else -1))
if rc!=0: print("[stderr]\n", err[:1000])


# In[ ]:


from tqdm import tqdm
import subprocess, textwrap

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

IN_DIR   = Path("/home/spai0308/data/raw/files")
OUT_TXT  = Path("/home/spai0308/data/raw/converted/docs")
OUT_HTML = Path("/home/spai0308/data/raw/converted/html")
LOG_PATH = Path("/home/spai0308/data/interim/hwp_convert_errors.log")
OUT_TXT.mkdir(parents=True, exist_ok=True); OUT_HTML.mkdir(parents=True, exist_ok=True); LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_append(msg: str):
    prev = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    LOG_PATH.write_text(prev + msg + "\n", encoding="utf-8")

files = list(IN_DIR.rglob("*.hwp"))
print(f"HWP 대상: {len(files)}개")

txt_ok=html_ok=css_ok=0
txt_fail=html_fail=css_fail=0

for f in tqdm(files, desc="HWP→TXT/HTML/CSS (existence-checked)"):
    base = f.stem
    txt  = OUT_TXT  / f"{base}.txt"
    html = OUT_HTML / f"{base}.html"
    css  = OUT_HTML / f"{base}.css"

    # TXT
    if (not txt.exists()) or (f.stat().st_mtime > txt.stat().st_mtime):
        rc, out, err = run([str(HWP5TXT), str(f), "--output", str(txt)])
        if rc==0 and txt.exists() and txt.stat().st_size>0: txt_ok+=1
        else:
            txt_fail+=1; log_append(f"[TXT FAIL] {f} :: rc={rc} exists={txt.exists()} size={(txt.stat().st_size if txt.exists() else -1)} :: {err[:500]}")

    # HTML
    if (not html.exists()) or (f.stat().st_mtime > html.stat().st_mtime):
        rc, out, err = run([str(HWP5HTML), str(f), "--html", "--output", str(html)])
        if rc==0 and html.exists() and html.stat().st_size>0: html_ok+=1
        else:
            html_fail+=1; log_append(f"[HTML FAIL] {f} :: rc={rc} exists={html.exists()} size={(html.stat().st_size if html.exists() else -1)} :: {err[:500]}")

    # CSS (선택)
    if (not css.exists()) or (f.stat().st_mtime > css.stat().st_mtime):
        rc, out, err = run([str(HWP5HTML), str(f), "--css", "--output", str(css)])
        if rc==0 and css.exists() and css.stat().st_size>0: css_ok+=1
        else:
            css_fail+=1; log_append(f"[CSS FAIL] {f} :: rc={rc} exists={css.exists()} size={(css.stat().st_size if css.exists() else -1)} :: {err[:500]}")

print(textwrap.dedent(f"""
=== 파일 실존 기준 재집계 ===
- TXT OK/Fail: {txt_ok} / {txt_fail}
- HTML OK/Fail:{html_ok} / {html_fail}
- CSS  OK/Fail:{css_ok} / {css_fail}
- 로그: {LOG_PATH}
"""))


# In[ ]:


# 주석: 콘솔 스크립트는 일반적으로 아래 경로들 중 하나에 설치됨
#  - sysconfig.get_paths()['scripts']  (가상환경의 bin)
#  - Path(sys.executable).parent       (가상환경의 bin)
#  - ~/.local/bin                      (사용자 로컬)
#  - PATH 상에 있는 경로들

from pathlib import Path
import sysconfig, shutil, os

candidates = []
scripts_dir = Path(sysconfig.get_paths()['scripts'])
candidates.append(scripts_dir)
candidates.append(Path(sys.executable).parent)
candidates.append(Path.home()/".local/bin")

# PATH에 등록된 디렉터리들도 후보에 추가
for p in os.environ.get("PATH","").split(os.pathsep):
    if p:
        candidates.append(Path(p))

# 중복 제거
seen = set()
paths = []
for p in candidates:
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    if rp not in seen:
        seen.add(rp); paths.append(rp)

def find_exe(name: str) -> Path|None:
    # 1) shutil.which로 먼저 검색
    w = shutil.which(name)
    if w:
        return Path(w).resolve()
    # 2) 후보 디렉터리에서 직접 탐색
    for d in paths:
        for fname in (name, name+".py"):
            f = d / fname
            if f.exists() and os.access(f, os.X_OK):
                return f.resolve()
    return None

HWP5TXT  = find_exe("hwp5txt")
HWP5HTML = find_exe("hwp5html")
HWP5ODT  = find_exe("hwp5odt")

print("[HWP5TXT]", HWP5TXT)
print("[HWP5HTML]", HWP5HTML)
print("[HWP5ODT]", HWP5ODT)

# 안전장치: 하나도 못 찾으면 즉시 안내
if not HWP5TXT or not HWP5HTML:
    raise RuntimeError(
        "현재 커널에 hwp5 콘솔 스크립트가 보이지 않습니다. 위 1) 셀의 설치 로그를 확인해 주세요.\n"
        "설치가 정상인데도 못 찾는다면, Scripts 디렉토리를 PATH에 추가하거나, HWP5TXT/HWP5HTML 절대경로를 직접 지정해야 합니다."
    )


# In[ ]:


# 품질 점검

from pathlib import Path

OUT_TXT  = Path("/home/spai0308/data/raw/converted/docs")
OUT_HTML = Path("/home/spai0308/data/raw/converted/html")

# 1) 개수 확인
txt_files  = list(OUT_TXT.glob("*.txt"))     # 주석: 변환된 .txt 목록
html_files = list(OUT_HTML.glob("*.html"))   # 주석: 변환된 .html 목록
css_files  = list(OUT_HTML.glob("*.css"))    # 주석: 변환된 .css 목록

print(f"TXT: {len(txt_files)}, HTML: {len(html_files)}, CSS: {len(css_files)}")
# 주석: 수량이 HWP 수(96)와 같은지 확인

# 2) 빈 파일/짧은 파일 검사
bad_txt  = [p for p in txt_files  if p.stat().st_size < 20]   # 주석: 20B 미만이면 의심
bad_html = [p for p in html_files if p.stat().st_size < 50]   # 주석: 50B 미만이면 의심

print("의심 TXT(20B 미만):", len(bad_txt))
print("의심 HTML(50B 미만):", len(bad_html))
if bad_txt[:3]:  # 주석: 앞에서 3개만 프린트
    print("→ 샘플:", [p.name for p in bad_txt[:3]])

# 3) 샘플 미리보기(한글 깨짐 여부 등)
if txt_files:
    s = txt_files[0]
    print("\n[TXT 샘플]", s.name)
    print("-"*60)
    print("\n".join(s.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]))

if html_files:
    s = html_files[0]
    print("\n[HTML 샘플]", s.name)
    print("-"*60)
    print("\n".join(s.read_text(encoding="utf-8", errors="ignore").splitlines()[:40]))


# In[ ]:


# 최소 클린 (공백 정리, 페이지 번호 단독 줄 제거 등)

from pathlib import Path
import re

RAW_TXT   = Path("/home/spai0308/data/raw/converted/docs")
CLEAN_TXT = Path("/home/spai0308/data/processed/clean_txt")
CLEAN_TXT.mkdir(parents=True, exist_ok=True)

def clean_txt(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 페이지 번호처럼 보이는 단독 숫자/하이픈/로마자만 있는 줄 제거
    s = re.sub(r"(?m)^\s*(?:[0-9]+|[ⅰ-ⅴⅠ-Ⅴ]+|[-–—]+)\s*$\n?", "", s)
    # 양쪽 공백 정리
    s = "\n".join(line.rstrip() for line in s.splitlines())
    # 빈 줄 3회 이상 → 2회로 축소
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

cnt=0
for p in RAW_TXT.glob("*.txt"):
    t = p.read_text(encoding="utf-8", errors="ignore")
    t = clean_txt(t)
    (CLEAN_TXT / p.name).write_text(t, encoding="utf-8")
    cnt+=1
print("클린 완료:", cnt, "개 →", CLEAN_TXT)


# In[ ]:


# html -> plain 텍스트 추출 (beautifulsoup 사용)

from pathlib import Path
from bs4 import BeautifulSoup

RAW_HTML   = Path("/home/spai0308/data/raw/converted/html")
HTML_PLAIN = Path("/home/spai0308/data/processed/html_plain")
HTML_PLAIN.mkdir(parents=True, exist_ok=True)

def html_to_text(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "html.parser")
    # 스크립트/스타일 제거
    for tag in soup(["script","style"]): tag.decompose()
    text = soup.get_text(separator="\n")
    # 연속 공백 정리
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln.strip() != ""])
    return text.strip()

cnt=0
for p in RAW_HTML.glob("*.html"):
    h = p.read_text(encoding="utf-8", errors="ignore")
    plain = html_to_text(h)
    (HTML_PLAIN / (p.stem + ".txt")).write_text(plain, encoding="utf-8")
    cnt+=1
print("HTML→plain 완료:", cnt, "개 →", HTML_PLAIN)


# #### 2) html 내장 이미지 추출 : 수동 변환 후 ocr 

# #### 폴더 평탄화 & 안전한 리네임/이동

# - 링크 깨짐 방지를 위해 디렉터리 구조를 보존하면서
# - 문서별로 폴더로 이동(자산 경로 수정 불필요)

# In[ ]:


from pathlib import Path
import shutil, re

ROOT = Path("/home/spai0308/data/raw/converted")
DEST_HTML = ROOT/"html"; DEST_HTML.mkdir(parents=True, exist_ok=True)
DEST_XML  = ROOT/"xml";  DEST_XML.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    # 공백 정리
    s = re.sub(r"\s+", " ", name).strip()
    # 끝의 " html"/" xml" 제거
    s = re.sub(r"\s+(html|xml)$", "", s, flags=re.I)
    # 파일에 위험한 문자 치환
    s = s.replace("/", "_").replace("\\", "_")
    s = s.replace(":", "：").replace("*","·").replace("?","？").replace('"',"'")
    s = s.replace("<","(").replace(">",")").replace("|","·")
    # 앞뒤 점 제거
    s = s.strip(" .")
    return s or "doc"

def unique_dir(base_dir: Path, name: str) -> Path:
    d = base_dir/name
    i = 2
    while d.exists():
        d = base_dir/f"{name}-{i}"
        i += 1
    return d

def pick_doc_id_from_dir(d: Path, kind: str) -> str:
    # 디렉터리명이 의미 있는 케이스(“… html”, “… xml”)
    nm = d.name
    if re.search(r"\s+(html|xml)$", nm, flags=re.I):
        return slugify(nm)
    # 내부 대표 파일로 추정
    exts = {".html",".htm"} if kind=="html" else {".xml"}
    reps = sorted([p for p in d.glob("*") if p.suffix.lower() in exts],
                  key=lambda p: p.stat().st_size if p.exists() else 0,
                  reverse=True)
    if reps:
        return slugify(reps[0].stem)
    # 폴더 이름을 그대로
    return slugify(nm)

def move_contents(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir/item.name
        if target.exists():
            # 충돌 시 덮지 않고 이름 바꿔서 보존
            stem, suf = item.stem, item.suffix
            alt = dst_dir/f"{stem}__migrated{item.stat().st_mtime_ns}{suf}"
            target = alt
        shutil.move(str(item), str(target))

def migrate_under(folder: Path):
    # 하위에서 html 후보, xml 후보 찾기 (이름에 'html'/'xml' 포함)
    subs = [d for d in folder.iterdir() if d.is_dir()]
    html_dirs = [d for d in subs if "html" in d.name.lower()]
    xml_dirs  = [d for d in subs if "xml"  in d.name.lower()]

    moved = []
    for d in html_dirs:
        doc_id = pick_doc_id_from_dir(d, "html")
        dest   = unique_dir(DEST_HTML, doc_id)
        print(f"[HTML] {d}  →  {dest}")
        move_contents(d, dest)
        moved.append(d)

    for d in xml_dirs:
        doc_id = pick_doc_id_from_dir(d, "xml")
        dest   = unique_dir(DEST_XML, doc_id)
        print(f"[XML ] {d}  →  {dest}")
        move_contents(d, dest)
        moved.append(d)

    # 비워졌으면 정리
    for d in moved:
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        if not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass

# 1) "새 폴더 (...)" 전부 처리 + 루트에 이상한 곳도 한 번 더 훑기
cands = [d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("새 폴더")]
for c in sorted(cands, key=lambda p: p.name):
    migrate_under(c)

# 2) 혹시 루트 직속에 '... html' / '... xml' 폴더가 있다면 추가 처리
for d in ROOT.iterdir():
    if not d.is_dir(): 
        continue
    nm = d.name.lower()
    if nm.endswith(" html") or nm == "html":
        migrate_under(d)
    elif nm.endswith(" xml") or nm == "xml":
        migrate_under(d)

print("\n[정리 끝] html 문서 폴더 수:", sum(1 for _ in DEST_HTML.iterdir()),
      "| xml 문서 폴더 수:", sum(1 for _ in DEST_XML.iterdir()))


# #### 3) **자산 OCR (폴더 구조 불문, 재귀 스캔)**

# In[ ]:


# api 키 등록 확인
from dotenv import load_dotenv
import os

# 홈의 ~/.env를 명시적으로 로드(작업 dB가 어디든 확실히 로드된다.)
load_dotenv(os.path.expanduser("~/.env"))
print(os.getenv("OPENAI_API_KEY"))


# ##### **디버깅 없는 통합 자산 OCR 스크립트(HTML/XML, assets 유무 무관)**

# ##### **아래는 디버깅을 통한 자산 OCR (성공)**

# In[ ]:


# -*- coding: utf-8 -*-
import os, io, json, glob, base64, hashlib, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageSequence, UnidentifiedImageError
from tqdm import tqdm
from openai import OpenAI

# ====== 경로 ======
DATA = Path("/home/spai0308/data")
CONVERTED = DATA / "raw" / "converted"
INTERIM   = DATA / "interim"

OUT_HTML  = INTERIM / "assets_html_ocr.jsonl"
OUT_XML   = INTERIM / "assets_xml_ocr.jsonl"
CACHE_HTML = INTERIM / "assets_html_ocr.openai.cache.json"
CACHE_XML  = INTERIM / "assets_xml_ocr.openai.cache.json"

# ====== 시간대 ======
KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat()

# ====== OpenAI ======
client = OpenAI()

# ====== 모델 정책 (정리) ======
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY",   "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4.1"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4o"),
]
MAX_ESCALATION_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "1"))
DOC_UPGRADE_BUDGET   = int(os.getenv("OPENAI_OCR_DOC_UPGRADE_BUDGET", "3"))
MAX_REQ_PER_MIN      = int(os.getenv("OPENAI_OCR_RPM", "20"))

MIN_CHARS       = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))
MIN_ALPHA_RATIO = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))
HANGUL_BONUS    = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))

VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp",".jfif",".heic",".heif"}

def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_doc_id_from_path(img_path: Path, kind: str) -> str:
    """
    /converted/<kind>/<doc_id>/.../* 에서 <doc_id> 추출
    """
    parts = img_path.parts
    if "converted" in parts and kind in parts:
        i = parts.index(kind)
        if i+1 < len(parts):
            return parts[i+1]
    # 차선책
    return img_path.parent.name

def compress_image(pil_img: Image.Image, max_side=1280, fmt="JPEG", quality=85) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w,h) > max_side:
        scale = max_side / float(max(w,h))
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(image_bytes: bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def call_vision(model: str, pil_img: Image.Image) -> str:
    img_bytes = compress_image(pil_img, max_side=1280, fmt="JPEG", quality=85)
    data_url = to_data_url(img_bytes, "image/jpeg")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
            {"role":"user","content":[
                {"type":"text","text":"Extract all text from the image. If it's a table, read left-to-right, top-to-bottom. Return text only."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ],
        temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip()

def text_quality_is_poor(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CHARS:
        return True
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = alpha / total if total else 0
    ratio += HANGUL_BONUS if hangul > 0 else 0.0
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())

def ocr_with_escalation(pil_img: Image.Image, doc_id: str, doc_budget_used: dict):
    used_model = MODEL_LADDER[0]
    text = call_vision(used_model, pil_img)
    steps = 0
    while text_quality_is_poor(text) and steps < MAX_ESCALATION_STEPS:
        if doc_budget_used.get(doc_id, 0) >= DOC_UPGRADE_BUDGET:
            break
        steps += 1
        next_idx = min(steps, len(MODEL_LADDER)-1)
        used_model = MODEL_LADDER[next_idx]
        text = call_vision(used_model, pil_img)
        doc_budget_used[doc_id] = doc_budget_used.get(doc_id, 0) + 1
    return text, used_model, steps

def iter_asset_images(kind: str):
    """
    구조 불문:
    - /converted/<kind>/<doc_id>/**  재귀 탐색해서 이미지 확장자면 모두 대상
    - 최상위(= <doc_id>) 폴더명을 문서 ID로 사용
    """
    root = CONVERTED/kind
    if not root.exists():
        return
    for doc_dir in root.iterdir():
        if not doc_dir.is_dir(): 
            continue
        for p in doc_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                # HEIC/HEIF 열다가 실패할 수 있으므로 try/except
                try:
                    pil = Image.open(str(p))
                except UnidentifiedImageError:
                    continue
                yield p, getattr(pil, "is_animated", False), pil

def run_openai_assets_ocr(kind: str, out_jsonl: Path, cache_json: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_json.exists():
        try:
            cache = json.load(open(cache_json, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    rpm_win = []
    doc_budget_used = {}
    total = 0

    with open(out_jsonl, "a", encoding="utf-8") as out_f:
        for img_path, is_multi, pil in tqdm(iter_asset_images(kind), desc=f"OpenAI OCR ({kind})"):
            try:
                md5 = file_md5(img_path)
                if md5 in cache:
                    continue
                doc_id = extract_doc_id_from_path(img_path, kind)

                if is_multi:
                    for frame_index, frame in enumerate(ImageSequence.Iterator(pil)):
                        ratelimit(rpm_win)
                        text, used_model, steps = ocr_with_escalation(frame.convert("RGB"), doc_id, doc_budget_used)
                        rec = {
                            "doc_id": doc_id,
                            "source_path": str(img_path),
                            "frame_index": int(frame_index),
                            "text": text,
                            "avg_conf": -1.0,
                            "lang": "ko+en",
                            "preprocess": {"resize_max_side":1280,"format":"jpeg","quality":85},
                            "ts": now_kst_iso(),
                            "source_type": f"asset_ocr_{kind}",
                            "provider": "openai",
                            "model": used_model,
                            "escalation_steps": steps
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    ratelimit(rpm_win)
                    text, used_model, steps = ocr_with_escalation(pil.convert("RGB"), doc_id, doc_budget_used)
                    rec = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "frame_index": 0,
                        "text": text,
                        "avg_conf": -1.0,
                        "lang": "ko+en",
                        "preprocess": {"resize_max_side":1280,"format":"jpeg","quality":85},
                        "ts": now_kst_iso(),
                        "source_type": f"asset_ocr_{kind}",
                        "provider": "openai",
                        "model": used_model,
                        "escalation_steps": steps
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                cache[md5] = True
                total += 1

            except Exception as e:
                err = {
                    "doc_id": "unknown",
                    "source_path": str(img_path),
                    "error": repr(e),
                    "ts": now_kst_iso(),
                    "source_type": f"asset_ocr_{kind}_error",
                    "provider": "openai",
                }
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

    with open(cache_json, "w", encoding="utf-8") as cf:
        json.dump(cache, cf, ensure_ascii=False, indent=2)

    print(f"[OK] OpenAI OCR complete ({kind}) → {out_jsonl} (total {total} images)")

# 실행
KIND = os.getenv("ASSETS_KIND")  # "html" 또는 "xml"
if KIND in ("html","xml"):
    run_openai_assets_ocr(KIND, OUT_HTML if KIND=="html" else OUT_XML,
                          CACHE_HTML if KIND=="html" else CACHE_XML)
else:
    run_openai_assets_ocr("html", OUT_HTML, CACHE_HTML)
    run_openai_assets_ocr("xml",  OUT_XML,  CACHE_XML)


# In[ ]:


# 원클릭 상태 점검
from pathlib import Path
import json
from collections import Counter

INTERIM = Path("/home/spai0308/data/interim")
out = INTERIM/"assets_xml_ocr.jsonl"
cache = INTERIM/"assets_xml_ocr.openai.cache.json"

lines=errs=0; docs=set(); steps=[]
if out.exists():
    with open(out,"r",encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            lines+=1
            try:
                j=json.loads(ln)
                if str(j.get("source_type","")).endswith("_error"): errs+=1
                if j.get("doc_id"): docs.add(j["doc_id"])
                if "escalation_steps" in j: steps.append(int(j["escalation_steps"]))
            except: pass

cached = 0
if cache.exists():
    try: cached = len(json.load(open(cache,"r",encoding="utf-8")))
    except: pass

print(f"[XML OCR] jsonl_lines={lines}  errors={errs}  unique_docs={len(docs)}  cached_images={cached}")
if steps: print("escalation_counts:", Counter(steps))


# In[ ]:


# 원클릭 품질, 커버리지 점검

from pathlib import Path
import json, re
from collections import defaultdict

BASE = Path("/home/spai0308/data")
CONV = BASE/"raw"/"converted"
INTERIM = BASE/"interim"
OCR_JL = INTERIM/"assets_xml_ocr.jsonl"

# 1) OCR에 반영된 doc_id → 이미지 수/문자수 간단 통계
per_doc_cnt = defaultdict(int); per_doc_chars = defaultdict(int)
with open(OCR_JL,"r",encoding="utf-8") as f:
    for ln in f:
        j = json.loads(ln)
        if str(j.get("source_type","")).endswith("_error"): continue
        did = (j.get("doc_id") or "").strip() or "unknown"
        txt = (j.get("text") or "")
        per_doc_cnt[did]+=1
        per_doc_chars[did]+=len(txt)

print("[TOP 10 img 많은 문서]")
for did in sorted(per_doc_cnt, key=lambda d: per_doc_cnt[d], reverse=True)[:10]:
    print(f" - {did}: imgs={per_doc_cnt[did]} chars={per_doc_chars[did]}")

# 2) XML 경로에 이미지가 있는데 OCR 결과가 '전혀' 없는 doc 후보
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}
def collect_xml_imgs():
    paths=[]
    # 표준 위치
    if (CONV/"xml").exists():
        paths += list((CONV/"xml").rglob("*"))
    # '새 폴더 (*)/xml'
    for p in CONV.iterdir():
        if p.is_dir() and p.name.startswith("새 폴더"):
            sub = p/"xml"
            if sub.exists(): paths += list(sub.rglob("*"))
    imgs=[p for p in paths if p.is_file() and p.suffix.lower() in VALID_EXT]
    return imgs

imgs = collect_xml_imgs()
def get_doc_id_from_path(p: Path):
    # 새 폴더 (n)/xml/<파일> → 파일 stem을 기본 doc_id로
    # 혹은 assets/<파일>이면 상위 폴더를 doc_id로 추정
    if "assets" in p.parts:
        i = p.parts.index("assets")
        if i-1 >= 0: return p.parts[i-1]
    return re.sub(r"\s+", " ", p.stem).strip()

img_docs = set(get_doc_id_from_path(p) for p in imgs)
ocr_docs  = set(per_doc_cnt.keys())
missing = sorted(d for d in img_docs if d not in ocr_docs)

print(f"\n[이미지는 있는데 OCR 문서 엔트리가 없는 수] {len(missing)}")
print("샘플:", missing[:10])


# **[결과]**
# 
# - ``missing=633`` : 문서 ID 매핑 방식이 서로 달라서 생긴 가짜 경고일 가능성
# -  체크 코드와 ocr이 문서를 세는 기준이 달라서 생겼을 가능성 큼
#     - 체크 스크립트는 이미지 파일명(``BindDataItem1.png``)자체를 ``doc_id``로 사용했고
#     - OCR은 같은 폴더의 주 XML 파일명(stem)을 ``doc_id``로 썼을 확률 높음
# > 따라서 한 문서 안의 수십 개 이미지가 전부 ``BindDataItem``로 집계 되어 ``ocr이 없는 문서``처럼 보인 것!
# 
# 

# In[ ]:


# 이미지 --> 실제 문서 doc_id 매핑 교정해서 다시 missing 계산

from pathlib import Path
import json, re
from collections import defaultdict

BASE = Path("/home/spai0308/data")
CONV = BASE/"raw"/"converted"
INTERIM = BASE/"interim"
OCR_JL = INTERIM/"assets_xml_ocr.jsonl"

# 1) OCR 결과 집계: source_path → doc_id
ocr_by_path = {}
with open(OCR_JL, "r", encoding="utf-8") as f:
    for ln in f:
        j = json.loads(ln)
        if str(j.get("source_type","")).endswith("_error"): 
            continue
        ocr_by_path[j["source_path"]] = j["doc_id"]

# 2) 이미지 수집
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}
def collect_xml_imgs():
    roots = []
    if (CONV/"xml").exists():
        roots.append(CONV/"xml")
    for p in CONV.iterdir():
        if p.is_dir() and p.name.startswith("새 폴더"):
            sub = p/"xml"
            if sub.exists():
                roots.append(sub)
    imgs=[]
    for r in roots:
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                imgs.append(p)
    return imgs

imgs = collect_xml_imgs()

# 3) 이미지가 속한 "문서 doc_id"를 추정:
#    - 같은 폴더에 .xml이 하나면 그 stem을 문서 ID로 사용
#    - 여러 개면 가장 큰 파일(바이트) 기준으로 대표 선정
#    - 없으면 상위 폴더명 사용(최후의 fallback)
def doc_id_for_image(img_path: Path) -> str:
    xmls = list(img_path.parent.glob("*.xml"))
    if len(xmls) == 1:
        return xmls[0].stem
    elif len(xmls) > 1:
        xmls.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        return xmls[0].stem
    # fallback
    return img_path.parent.name

# 4) 교정된 doc_id로 missing 재계산
img_docs_corrected = set(doc_id_for_image(p) for p in imgs)
ocr_docs = set(ocr_by_path.values())

missing_corrected = sorted(d for d in img_docs_corrected if d not in ocr_docs)

print(f"[교정 후] 이미지가 있는데 OCR 문서 엔트리가 없는 수: {len(missing_corrected)}")
print("샘플:", missing_corrected[:10])





# - missing 숫자 633 -> 13으로 급감!
# - **doc_id(테이블 칼럼) 맞춰주기** (컬럼에 따라서 결과가 좌우된다)

# In[ ]:


# ocr 엔트리없는 나머지 13개 실제 이미지/확장자/경로 확인

# 확인가능 사항 : 이미지 총 개수, 확장자 분포
# ocr 처리에 해당하는지(경로/정규화 ID 둘다)
# 샘플 경로


from pathlib import Path
import json, unicodedata
from collections import defaultdict, Counter

BASE = Path("/home/spai0308/data")
CONV = BASE/"raw"/"converted"
INTERIM = BASE/"interim"
OCR_JL = INTERIM/"assets_xml_ocr.jsonl"

MISSING_DOCS = [
"(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 ",
"KOICA 전자조달_[긴급] [지문] [국제] 우즈베키스탄 열린 의정활동 상하원 ",
"국가과학기술지식정보서비스_통합정보시스템 고도화 용역",
"그랜드코리아레저(주)_2024년도 GKL  그룹웨어 시셈 구축 용역",
"대한적십자사 의료원_적십자병원 병원정보 재해복구시스템 구축 용역 ",
"서울특별시교육청_서울특별시교육청 지능정보화전략계획(ISP) 수립(2차) ",
"인천공항운영서비스(주)_인천공항운영서비스㈜ 차세대 ERP시셈 구축 ",
"인천광역시 동구_수도국산달동네박물관 전시해설 시셈 구축(협상에 ",
"한국생산기술연구원_2세대 전자조달셈  기반구축사업",
"한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역"
]

VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}
def norm(s:str)->str:
    # 공백/줄끝, 유니코드 정규화
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = " ".join(s.split())
    return s

# OCR 결과 로드
ocr_paths = set()
ocr_doc_ids = set()
ocr_doc_ids_norm = set()
with open(OCR_JL, "r", encoding="utf-8") as f:
    for ln in f:
        j = json.loads(ln)
        if str(j.get("source_type","")).endswith("_error"): 
            continue
        ocr_paths.add(j["source_path"])
        ocr_doc_ids.add(j["doc_id"])
        ocr_doc_ids_norm.add(norm(j["doc_id"]))

# XML 이미지 모두 수집
def collect_xml_imgs():
    roots = []
    if (CONV/"xml").exists(): roots.append(CONV/"xml")
    for p in CONV.iterdir():
        if p.is_dir() and p.name.startswith("새 폴더"):
            sub = p/"xml"
            if sub.exists(): roots.append(sub)
    imgs=[]
    for r in roots:
        for p in r.rglob("*"):
            if p.is_file():
                imgs.append(p)
    return imgs

all_files = collect_xml_imgs()
all_imgs = [p for p in all_files if p.suffix.lower() in VALID_EXT]

# 이미지 → 대표 xml 파일(stem)로 문서 ID 추정
def doc_id_for_image(img_path: Path) -> str:
    xmls = list(img_path.parent.glob("*.xml"))
    if len(xmls) == 1:
        return xmls[0].stem
    elif len(xmls) > 1:
        xmls.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        return xmls[0].stem
    return img_path.parent.name

imgs_by_doc = defaultdict(list)
for p in all_imgs:
    imgs_by_doc[doc_id_for_image(p)].append(p)

def report_one(doc_id_raw: str, max_show=5):
    did = doc_id_raw
    did_n = norm(did)
    imgs = imgs_by_doc.get(did, []) or imgs_by_doc.get(did_n, [])
    ext_counts = Counter(p.suffix.lower() for p in imgs)
    ocr_hit_doc = (did in ocr_doc_ids) or (did_n in ocr_doc_ids) or (did_n in ocr_doc_ids_norm)

    # 이 문서의 이미지 경로 중, OCR가 이미 처리한 경로가 있는지
    hit_by_path = any(str(p) in ocr_paths for p in imgs)

    print("="*90)
    print(f"[DOC] raw='{did}' | norm='{did_n}'")
    print(f"  imgs={len(imgs)}  ext={dict(ext_counts)}")
    print(f"  OCR hit by doc_id? {ocr_hit_doc} | by path? {hit_by_path}")
    for p in imgs[:max_show]:
        print("   -", p)

for d in MISSING_DOCS:
    report_one(d)


# ##### 근본 원인 추정
# 
# 1. **스캔 범위/도큐먼트 ID 추출 로직**
#     - 초깃값은 `/**/assets/*`만 훑고, `extract_doc_id`도 `…/assets/` 전제였음.
#     - 지금 구조는 **이미지가 xml 폴더 바로 아래 섞여 있음** → 누락 가능.
# 2. **캐시만 있고 JSONL엔 없는 경우**
#     - 예전에 돌린 MD5가 캐시에 남아 있으면 **이번 실행에서 스킵되는데** JSONL엔 기록이 없을 수 있음 → “이미지 있는데 OCR 엔트리 없음”.
# 3. **동일 문서의 이름 변형(공백/괄호/언더바/NFKC 차이)**
#     - 같은 문서가 서로 다른 폴더명/파일명으로 존재 → 일부는 처리, 일부는 미처리.
# 
# ----------------------------------------
# 
# ##### 개선 순서
# 
# 1. **스캐너/ID 추출 보강**
#     - `iter_asset_images("xml")`가 **`/**/assets/*` + `/**/<doc_id>/*`** 모두 훑게.
#     - `extract_doc_id`는 **(a) 형제 .xml 스템**, 없으면 **(b) 상위 폴더명**으로 잡게.
# 2. **캐시 충돌 검사 후 정리(수술식)**
#     - “미싱” 이미지들 **MD5가 캐시에만 있고 JSONL엔 없는지** 확인 → 있으면 그 키만 지우고 재실행(또는 leftover만 재-OCR).
# 3. **이름 정규화 키 통일**
#     - NFKC + 괄호/연속공백/언더바 정리 같은 **`normalize_doc_id()`*로 최종 병합 키 통일.
# 4. **leftover만 재-OCR**
#     - 지원 확장자(jpg/png/gif/bmp/…)인데 JSONL에 경로가 없는 것만 타깃 재-OCR.

# In[ ]:


# 스캔/id 추출 보강
def extract_doc_id(img_path: Path) -> str:
    parts = img_path.parts
    # 1) 가장 가까운 형제 .xml 파일 스템 우선
    xmls = list(img_path.parent.glob("*.xml"))
    if xmls:
        # 가장 큰(본문일 확률 높은) xml을 대표로
        xmls.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        return xmls[0].stem
    # 2) 기존 assets 규칙
    if "converted" in parts:
        i = parts.index("converted")
        if i + 3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
        # 3) 일반 구조: .../converted/<kind>/<doc_id>/<image>
        if i + 3 < len(parts):
            return parts[i+2]
    # 4) 마지막 보루: 상위 폴더명
    return img_path.parent.name

def iter_asset_images(kind: str):
    roots = [CONVERTED / kind]
    # '새 폴더 (n)' 구조도 포함
    for p in CONVERTED.iterdir():
        if p.is_dir() and p.name.startswith("새 폴더") and (p / kind).exists():
            roots.append(p / kind)

    for root in roots:
        # (A) 전통 assets/*
        for p in root.rglob("assets/*"):
            path = Path(p)
            if path.suffix.lower() in VALID_EXT:
                try:
                    pil = Image.open(str(path))
                except UnidentifiedImageError:
                    continue
                yield path, getattr(pil, "is_animated", False), pil
        # (B) 인라인 이미지: 문서 폴더 바로 아래 이미지들
        for docdir in [d for d in root.iterdir() if d.is_dir()]:
            # 이미지가 있고 .xml이 1개 이상 있는 폴더만
            if not list(docdir.glob("*.xml")):
                continue
            for p in docdir.iterdir():
                if p.is_file() and p.suffix.lower() in VALID_EXT:
                    try:
                        pil = Image.open(str(p))
                    except UnidentifiedImageError:
                        continue
                    yield p, getattr(pil, "is_animated", False), pil




# In[ ]:


from pathlib import Path
import json

CONVERTED_XML = Path("/home/spai0308/data/raw/converted/xml")
JSONL = Path("/home/spai0308/data/interim/assets_xml_ocr.jsonl")
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

def list_all_xml_images():
    return [p for p in CONVERTED_XML.rglob("*") if p.suffix.lower() in VALID_EXT and p.is_file()]

def jsonl_ocr_paths(jsonl_path: Path):
    S=set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    sp=rec.get("source_path")
                    if sp: S.add(str(Path(sp)))
                except: pass
    return S

all_imgs   = list_all_xml_images()
ocr_paths  = jsonl_ocr_paths(JSONL)
leftovers  = [p for p in all_imgs if str(p) not in ocr_paths]
print("[SCAN] total imgs:", len(all_imgs), "| in jsonl:", len(ocr_paths), "| leftovers:", len(leftovers))
print("sample leftovers:", [str(p) for p in leftovers[:5]])


# In[ ]:


# 캐시 충돌 점검 & 캐시만 있고 jsonl에 없는 항목 골라서 캐시에서만 재시도

import hashlib, json

def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

cache_path = INTERIM / "assets_xml_ocr.openai.cache.json"
cache = json.load(open(cache_path, "r", encoding="utf-8")) if cache_path.exists() else {}

victims = []
for p in leftovers:
    md5 = file_md5(p)
    if md5 in cache:
        victims.append((p, md5))

print(f"[캐시에만 있고 JSONL엔 없는 항목] {len(victims)}")
print("샘플 victims:", [str(v[0]) for v in victims[:10]])

# 필요할 때만 '수술식' 삭제
for _, md5 in victims:
    cache.pop(md5, None)
json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("[OK] cache trimmed.")


# In[ ]:


import json, hashlib

CACHE = Path("/home/spai0308/data/interim/assets_xml_ocr.openai.cache.json")
def file_md5(p: Path) -> str:
    h=hashlib.md5()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

cache = json.load(open(CACHE, "r", encoding="utf-8")) if CACHE.exists() else {}
in_cache = [p for p in leftovers if file_md5(p) in cache]
print(f"[CHECK] in_cache: {len(in_cache)} / leftovers: {len(leftovers)}")


# In[ ]:


# ===== force re-OCR (ignore cache) =====
import io, base64, time, json
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageSequence, UnidentifiedImageError
from openai import OpenAI

JSONL = Path("/home/spai0308/data/interim/assets_xml_ocr.jsonl")
CACHE = Path("/home/spai0308/data/interim/assets_xml_ocr.openai.cache.json")

KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat()

client = OpenAI()

# 모델 정책(원본과 동일)
import os
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY",   "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4.1"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4o"),
]
MAX_ESCALATION_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "1"))
DOC_UPGRADE_BUDGET   = int(os.getenv("OPENAI_OCR_DOC_UPGRADE_BUDGET", "3"))
MAX_REQ_PER_MIN      = int(os.getenv("OPENAI_OCR_RPM", "20"))

MIN_CHARS       = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))
MIN_ALPHA_RATIO = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))
HANGUL_BONUS    = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))

def compress_image(pil_img, max_side=1280, fmt="JPEG", quality=85) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w,h) > max_side:
        scale = max_side/float(max(w,h))
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(image_bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def call_vision(model: str, pil_img: Image.Image) -> str:
    img_bytes = compress_image(pil_img, max_side=1280, fmt="JPEG", quality=85)
    data_url = to_data_url(img_bytes, "image/jpeg")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
            {"role":"user","content":[
                {"type":"text","text":"Extract all text from the image. If it's a table, read left-to-right, top-to-bottom. Return text only."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ],
        temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip()

def text_quality_is_poor(text: str) -> bool:
    t=(text or "").strip()
    if len(t) < MIN_CHARS: return True
    total=len(t); alpha=sum(ch.isalpha() for ch in t)
    hangul=sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = alpha/total if total else 0
    ratio += HANGUL_BONUS if hangul>0 else 0.0
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now=time.time()
    win[:] = [t for t in win if now-t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())

def extract_doc_id_any(img_path: Path) -> str:
    parts = img_path.parts
    if "xml" in parts:
        i = parts.index("xml")
        if i+1 < len(parts):
            return parts[i+1]
    if "html" in parts:
        i = parts.index("html")
        if i+1 < len(parts):
            return parts[i+1]
    return img_path.parent.name  # 최후의 보루

def ocr_with_escalation(pil_img: Image.Image, doc_id: str, doc_budget_used: dict):
    used_model = MODEL_LADDER[0]
    text = call_vision(used_model, pil_img)
    steps = 0
    while text_quality_is_poor(text) and steps < MAX_ESCALATION_STEPS:
        if doc_budget_used.get(doc_id, 0) >= DOC_UPGRADE_BUDGET:
            break
        steps += 1
        next_idx = min(steps, len(MODEL_LADDER)-1)
        used_model = MODEL_LADDER[next_idx]
        text = call_vision(used_model, pil_img)
        doc_budget_used[doc_id] = doc_budget_used.get(doc_id, 0) + 1
    return text, used_model, steps

# --- 실행 ---
from tqdm import tqdm
import hashlib

# (안전) 카운트 스냅샷
def line_count(p: Path) -> int:
    if not p.exists(): return 0
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

before = line_count(JSONL)
print("[JSONL] before:", before)

# 캐시 로드(여기선 '무시'하지만, 성공한 건 다시 기록해 재실행 방지)
cache = {}
if CACHE.exists():
    try:
        cache = json.load(open(CACHE,"r",encoding="utf-8"))
    except: 
        cache = {}

rpm_win=[]; doc_budget_used={}
JSONL.parent.mkdir(parents=True, exist_ok=True)

appended=0; skipped=0
with open(JSONL, "a", encoding="utf-8") as out_f:
    for p in tqdm(leftovers, desc="FORCE re-OCR leftovers(xml)"):
        try:
            # 캐시는 '무시' (md5 키가 있어도 진행)
            pil = Image.open(str(p))
            doc_id = extract_doc_id_any(p)
            if getattr(pil, "is_animated", False):
                for frame_index, frame in enumerate(ImageSequence.Iterator(pil)):
                    ratelimit(rpm_win)
                    text, used_model, steps = ocr_with_escalation(frame.convert("RGB"), doc_id, doc_budget_used)
                    rec = {
                        "doc_id": doc_id,
                        "source_path": str(p),
                        "frame_index": int(frame_index),
                        "text": text,
                        "avg_conf": -1.0,
                        "lang": "ko+en",
                        "preprocess": {"resize_max_side":1280,"format":"jpeg","quality":85},
                        "ts": now_kst_iso(),
                        "source_type": "asset_ocr_xml",
                        "provider": "openai",
                        "model": used_model,
                        "escalation_steps": steps
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1
            else:
                ratelimit(rpm_win)
                text, used_model, steps = ocr_with_escalation(pil.convert("RGB"), doc_id, doc_budget_used)
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(p),
                    "frame_index": 0,
                    "text": text,
                    "avg_conf": -1.0,
                    "lang": "ko+en",
                    "preprocess": {"resize_max_side":1280,"format":"jpeg","quality":85},
                    "ts": now_kst_iso(),
                    "source_type": "asset_ocr_xml",
                    "provider": "openai",
                    "model": used_model,
                    "escalation_steps": steps
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1

            # 성공 시 캐시 업데이트(다음 번 전체 스캔에서 또 안 건드리게)
            md5 = hashlib.md5(open(p,"rb").read()).hexdigest()
            cache[md5] = True

        except UnidentifiedImageError:
            skipped += 1
        except Exception as e:
            # 에러라도 한 줄 남기기
            err = {
                "doc_id": "unknown",
                "source_path": str(p),
                "error": repr(e),
                "ts": now_kst_iso(),
                "source_type": "asset_ocr_xml_error",
                "provider": "openai",
            }
            out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

# 캐시 저장
with open(CACHE, "w", encoding="utf-8") as cf:
    json.dump(cache, cf, ensure_ascii=False, indent=2)

after = line_count(JSONL)
print(f"[DONE] appended files: {appended}, skipped(bad): {skipped}")
print("[JSONL] after:", after, " (delta:", after-before, ")")


# In[ ]:


# leftovers 리스트에 있는 이미지 확인
# --> 차후 추가 추출 필요함

from PIL import Image
from IPython.display import display

preview = leftovers[:] if 'leftovers' in globals() and leftovers else []
if not preview:
    # leftovers가 비어 있으면, 미싱 목록에서 첫 문서의 이미지 3장 표본
    first = MISSING_DOCS[0]
    imgs = imgs_by_doc.get(first, []) or imgs_by_doc.get(norm(first), [])
    preview = imgs[:3]

for p in preview:
    print("\n[PREVIEW]", p)
    try:
        img = Image.open(p)
        display(img)
    except Exception as e:
        print("  (열기 실패)", e)


# In[ ]:


# leftover 돌린뒤 잘 들어갔는지 확인
# JSONL 라인 증감을 확인 (before/after)

from pathlib import Path
import json

JSONL = Path("/home/spai0308/data/interim/assets_xml_ocr.jsonl")

def line_count(p: Path) -> int:
    if not p.exists(): return 0
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

before = line_count(JSONL)
print("before lines:", before)

# --- 여기서 reocr_leftovers(leftovers) 실행 ---
# reocr_leftovers(leftovers)

after = line_count(JSONL)
print("after  lines:", after, " (delta:", after-before, ")")


# In[ ]:


# 총 라인
get_ipython().system('wc -l /home/spai0308/data/interim/assets_xml_ocr.jsonl')

# 끝 5줄
get_ipython().system('tail -n 5 /home/spai0308/data/interim/assets_xml_ocr.jsonl')

# 특정 문서 경로 필터
get_ipython().system('grep -n "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역" /home/spai0308/data/interim/assets_xml_ocr.jsonl | head')


# #### 4) OCR 결과를 문서(doc_id) 단위로 집계 → ``ocr_text_by_doc.jsonl``

# In[ ]:


# aggregate_ocr_by_doc.py
# -*- coding: utf-8 -*-
"""
[입력]
- /home/spai0308/data/interim/assets_html_ocr.jsonl
- /home/spai0308/data/interim/assets_xml_ocr.jsonl

[출력]
- /home/spai0308/data/interim/ocr_text_by_doc.jsonl
  (doc_id별로 OCR 텍스트를 병합한 라인)

[특징]
- source_type이 asset_ocr_html/xml 인 레코드만 사용, error 라인은 스킵
- 같은 doc_id 안에서 라인 단위 중복 제거(원문 순서 보존) + 너무 짧은 라인 제거
- kind(html/xml)별 텍스트도 넣고, combined(둘 합친 것)도 넣음
- 이미지 수/문자수/모델/승급단계 카운트 등 간단 통계 포함
"""

from pathlib import Path
import json, re
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime, timezone, timedelta

DATA      = Path("/home/spai0308/data")
INTERIM   = DATA / "interim"

IN_HTML   = INTERIM / "assets_html_ocr.jsonl"
IN_XML    = INTERIM / "assets_xml_ocr.jsonl"
OUT_PATH  = INTERIM / "ocr_text_by_doc.jsonl"

# 라인 필터 파라미터(필요하면 조정)
MIN_LINE_LEN        = 2       # 이보다 짧은 라인은 버림
DROP_PUNCT_ONLY     = True    # 구두점/기호만 있는 라인은 버림
CONDENSE_SPACES     = True    # 연속 공백 축소

KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat()

def iter_jsonl(p: Path):
    if not p.exists(): 
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 깨진 라인은 그냥 무시
                continue

def normalize_line(s: str) -> str:
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = s.strip()
    if CONDENSE_SPACES:
        s = re.sub(r"[ \t\u3000]+", " ", s)
    return s

def good_line(s: str) -> bool:
    if not s or len(s) < MIN_LINE_LEN:
        return False
    if DROP_PUNCT_ONLY and re.fullmatch(r"[\s\W_·•○●■□◆◇▶▷◀◁※☆★∆∙‧・⦁\-\–\—\·\.\,\/\\\|\+\=\~\(\)\[\]\{\}\:\;\“\”\"\'\<\>\!\?]+", s):
        return False
    return True

def dedup_join(texts: list[str]) -> str:
    """
    여러 조각 텍스트를 합치면서 라인단위 중복 제거.
    (등장 순서 유지)
    """
    seen = OrderedDict()
    for chunk in texts:
        if not chunk:
            continue
        for ln in normalize_line(chunk).split("\n"):
            ln = ln.rstrip()
            if not good_line(ln):
                continue
            # 라인 단위 dedup
            if ln not in seen:
                seen[ln] = True
    return "\n".join(seen.keys()).strip()

def main():
    # doc별 수집 버킷
    buf_texts = defaultdict(lambda: {"html": [], "xml": []})
    stats = defaultdict(lambda: {
        "images": {"html": 0, "xml": 0},
        "chars" : {"html": 0, "xml": 0},
        "models": set(),
        "escalation": Counter()
    })

    src_files = []
    if IN_HTML.exists(): src_files.append(("html", IN_HTML))
    if IN_XML.exists():  src_files.append(("xml",  IN_XML))

    total_in = 0
    used_in  = 0

    for kind, path in src_files:
        for rec in iter_jsonl(path):
            total_in += 1
            st = rec.get("source_type","")
            if st not in (f"asset_ocr_{kind}"):   # error 라인 등 스킵
                # 혹시 다른 소스가 섞였으면 건너뜀
                continue
            doc_id = rec.get("doc_id") or "unknown"
            text   = rec.get("text","") or ""
            if not text.strip():
                # 빈 텍스트는 카운트만 반영하고 텍스트 누적은 생략
                stats[doc_id]["images"][kind] += 1
                stats[doc_id]["models"].add(rec.get("model",""))
                stats[doc_id]["escalation"][str(rec.get("escalation_steps",0))] += 1
                continue

            # 누적
            buf_texts[doc_id][kind].append(text)
            stats[doc_id]["images"][kind] += 1
            stats[doc_id]["chars"][kind]  += len(text)
            stats[doc_id]["models"].add(rec.get("model",""))
            stats[doc_id]["escalation"][str(rec.get("escalation_steps",0))] += 1
            used_in += 1

    # doc별로 정리해서 JSONL 작성
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for doc_id in sorted(set(list(buf_texts.keys()) + list(stats.keys()))):
            html_text = dedup_join(buf_texts[doc_id]["html"]) if buf_texts[doc_id]["html"] else ""
            xml_text  = dedup_join(buf_texts[doc_id]["xml"])  if buf_texts[doc_id]["xml"]  else ""
            combined  = dedup_join([html_text, xml_text])

            rec = {
                "doc_id": doc_id,
                "ts": now_kst_iso(),
                "source_type": "asset_ocr_agg",
                "ocr_text": combined,          # 최종 결합 텍스트
                "ocr_text_html": html_text,    # 참고용(원하면 이후 파이프라인에서 제거 가능)
                "ocr_text_xml":  xml_text,     # 참고용
                "stats": {
                    "images": stats[doc_id]["images"],
                    "chars":  stats[doc_id]["chars"],
                    "models": sorted([m for m in stats[doc_id]["models"] if m]),
                    "escalation": dict(stats[doc_id]["escalation"])
                }
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    # 요약 출력
    docs_with_any = sum(1 for d in buf_texts if (buf_texts[d]["html"] or buf_texts[d]["xml"]))
    print(f"[OCR 집계] 입력 라인: {total_in}  사용 라인: {used_in}")
    print(f"[OCR 집계] 문서 수(텍스트 보유): {docs_with_any}  / 출력 라인: {wrote}")
    # TOP 문서 샘플
    top = sorted(stats.items(), key=lambda kv: (kv[1]["images"]["html"]+kv[1]["images"]["xml"], kv[1]["chars"]["html"]+kv[1]["chars"]["xml"]), reverse=True)[:10]
    if top:
        print("[TOP 10 by images/chars]")
        for doc, st in top:
            imgs = st["images"]["html"] + st["images"]["xml"]
            chrs = st["chars"]["html"]  + st["chars"]["xml"]
            print(f" - {doc}: imgs={imgs} chars={chrs}")

if __name__ == "__main__":
    main()


# **[정리]**
# - 입력 890 → 실제 텍스트가 있던 레코드 737
# - 텍스트 가진 문서 67개 (이미지 없는 문서는 당연히 제외됨)
# - **TOP 리스트에 같은 문서의 이명(예: 언더스코어/괄호 차이) 가 보이니, 최종 병합 전에 doc_id 정규화 필요**

# ### 3-2-2. XML 텍스트 추출

# - ``~/data/raw/converted`` 아래를 재귀로 훑어서 모든 xml폴더(최상위 xml, ``...<doc> xml``의 ``.xml``을 읽고, 문서별(``doc_id`` 단위)로 텍스트를 모음
# - HWP/HWPX 변환물에서 흔한 바이너리/이미지/수식 태그는 스킵
# - ``<doc> xml`` 폴더명에서 뒤의 ``xml`` 꼬리표 제거하여 ``doc_id`` 추출
# - 다수 xml 파일이 한 문서에 있을 때, ``doc_id``로 합쳐 한 줄 (JSONL 한 레코드)로 저장
# - 경로 : ``~/data/intrim/xml_text.jsonl``
# - **최종 병합 단계와 포맷 호환 : ``doc_id``, ``source_type: "xml_text"``, ``text``, ``stats``**

# In[ ]:


# xml_text_extract_with_joinkey.py
# - XML 본문 텍스트 추출 (binData 등 바이너리 계열 태그 무시)
# - doc_id 단위로 합쳐 JSONL로 저장 (덮어쓰기)
# - join_key 필드 추가 (머지 시 안정적인 매칭 키)
# - 콘솔에 전체 랭킹(정렬) 출력

from pathlib import Path
import json, re, unicodedata
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

# ====== 경로 ======
CONVERTED = Path("/home/spai0308/data/raw/converted")
OUT_JSONL = Path("/home/spai0308/data/interim/xml_text.jsonl")

# ====== 시간대 ======
KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat(timespec="seconds")

# ====== 조인키(normalized join key) ======
def join_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    # 공백류 → _
    s = re.sub(r"[\s\u3000]+", "_", s)
    # 괄호류 제거
    s = s.translate(str.maketrans({"(":"", ")":"", "[":"", "]":""}))
    # 한글/영문/숫자/밑줄만 남기기
    s = re.sub(r"[^0-9a-z가-힣_]+", "", s)
    # 밑줄 정리
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ====== doc_id 추출 ======
def extract_doc_id_from_xml_path(p: Path) -> str:
    parts_lower = [seg.lower() for seg in p.parts]
    if "xml" in parts_lower:
        i = parts_lower.index("xml")
        if i + 1 < len(p.parts):
            return p.parts[i + 1]  # /.../xml/<doc_id>/xxx.xml
        else:
            return p.stem          # /.../xml/<file>.xml (예외)
    # 예외: 중간 세그가 '... xml'로 끝나는 폴더인 경우도 있었으므로 보정
    for idx, seg in enumerate(p.parts[:-1]):
        if seg.lower().endswith(" xml") and idx + 1 < len(p.parts):
            return p.parts[idx + 1]
    # 최후 보정
    return p.stem

# ====== XML -> 텍스트 ======
SKIP_TAGS = {"bindata", "imagedata", "binary", "picture", "drawing", "equation", "shape", "container"}
def localname(tag: str) -> str:
    # '{ns}Tag' → 'Tag'
    return tag.rsplit("}", 1)[-1].lower()

def extract_text_from_element(elem) -> str:
    # 특정 태그는 통째로 스킵
    if localname(elem.tag) in SKIP_TAGS:
        return ""
    texts = []
    if elem.text:
        texts.append(elem.text)
    for ch in list(elem):
        texts.append(extract_text_from_element(ch))
        if ch.tail:
            texts.append(ch.tail)
    return "".join(texts)

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 과도한 공백/빈줄 정리
    lines = [ln.rstrip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def main():
    # 대상 XML 수집: /converted/**/xml/**.xml
    xml_files = []
    for p in CONVERTED.rglob("*.xml"):
        parts_lower = [seg.lower() for seg in p.parts]
        if "xml" in parts_lower:   # xml 세그먼트 하위만 수집
            xml_files.append(p)

    # doc_id 단위로 텍스트 합치기
    doc_texts = defaultdict(list)
    doc_paths  = defaultdict(list)
    bad = 0

    for x in xml_files:
        doc_id = extract_doc_id_from_xml_path(x)
        try:
            tree = ET.parse(x)
            root = tree.getroot()
            raw = extract_text_from_element(root)
            txt = clean_text(raw)
            if txt:
                doc_texts[doc_id].append(txt)
                doc_paths[doc_id].append(str(x))
        except Exception as e:
            bad += 1
            # 필요시 로그 찍고 계속
            # print("[XML parse error]", x, e)

    # 기록 만들기
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for doc_id in sorted(doc_texts.keys(), key=lambda d: join_key(d)):
            merged = "\n\n".join(doc_texts[doc_id]).strip()
            rec = {
                "doc_id": doc_id,
                "join_key": join_key(doc_id),
                "text": merged,
                "stats": {
                    "chars": len(merged),
                    "xml_files": len(doc_texts[doc_id]),
                },
                "ts": now_kst_iso(),
                "source_type": "xml_text",
                "provider": "local"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 요약/랭킹(전체 출력)
    wrote_lines = len(doc_texts)
    print(f"[XML 텍스트] xml_files={len(xml_files)}  docs={wrote_lines}  wrote_lines={wrote_lines}  → {OUT_JSONL}")
    if bad:
        print(f"[warn] XML parse errors: {bad}")

    # 전체 랭킹: chars desc로 전부 출력
    ranking = sorted(
        ((doc, sum(len(t) for t in doc_texts[doc]), len(doc_texts[doc])) for doc in doc_texts),
        key=lambda x: (-x[1], x[0])
    )
    print("[ALL by xml chars]")
    for doc, ch, nfiles in ranking:
        print(f" - {doc}: chars={ch:,}, xml_files={nfiles}")

if __name__ == "__main__":
    main()


# In[ ]:


# 검증 : 머지 키 일치도

from pathlib import Path
import json

interim = Path("/home/spai0308/data/interim")
xml_j = interim / "xml_text.jsonl"
ocr_j = interim / "ocr_text_by_doc.jsonl"

xml_ids = set()
for ln in open(xml_j, encoding="utf-8"):
    o = json.loads(ln); xml_ids.add(o["doc_id"])

ocr_ids = set()
for ln in open(ocr_j, encoding="utf-8"):
    o = json.loads(ln); ocr_ids.add(o["doc_id"])

print("xml docs:", len(xml_ids))
print("ocr docs:", len(ocr_ids))
print("intersection:", len(xml_ids & ocr_ids))
print("sample overlap:", list(sorted(xml_ids & ocr_ids))[:10])


# [**점검+수정 스니펫**]
# 
# - **기능 A:** only-in-OCR / only-in-XML 목록 산출(도큐먼트/경로 샘플 포함)
# - **기능 B:** `join_key` 충돌 탐지 → 자동 분해용 `merge_key` 생성(충돌 없는 건 기존 키 재사용)
#     - 생성된 `merge_key`를 **두 JSONL(xml/ocr)** 모두에 추가한 새 파일로 저장(원본 보존).
#     - 원하면 `INPLACE=True`로 돌려 **덮어쓰기**도 가능

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re, hashlib
from collections import defaultdict, Counter

# ===== 경로 =====
BASE        = Path("/home/spai0308/data")
INTERIM     = BASE / "interim"
XML_JSONL   = INTERIM / "xml_text.jsonl"
OCR_JSONL   = INTERIM / "ocr_text_by_doc.jsonl"

AUDIT_TXT   = INTERIM / "join_audit.txt"
AUDIT_CSV   = INTERIM / "join_audit.csv"
FIXES_JSON  = INTERIM / "joinkey_fixes.json"

# 출력(원본 보존; INPLACE=True로 바꾸면 덮어씀)
INPLACE     = False
XML_OUT     = XML_JSONL if INPLACE else XML_JSONL.with_name(XML_JSONL.stem + ".with_mergekey.jsonl")
OCR_OUT     = OCR_JSONL if INPLACE else OCR_JSONL.with_name(OCR_JSONL.stem + ".with_mergekey.jsonl")

# ===== 조인키 규칙(이전에 썼던 normalize와 동일/유사) =====
def make_join_key(s: str) -> str:
    if not s: return ""
    t = s
    t = t.replace("\u200b","")                             # zero-width 제거
    t = re.sub(r"[(){}\[\]<>]+"," ", t)                    # 괄호류 제거→공백
    t = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+","_",t)  # 한글/영문/숫자/_만
    t = re.sub(r"_+","_",t).strip("_")
    t = t.lower()
    return t

def short_hash(s: str, n=6) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]

def load_jsonl(p: Path):
    rows = []
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 방어적으로 스킵
                pass
    return rows

def get_paths_any(rec):
    # 소스별 필드 이름이 다를 수 있어 가능한 후보 모두 시도
    for k in ("source_paths","paths","xml_paths","image_paths","source_paths_sample"):
        v = rec.get(k)
        if isinstance(v, list) and v: return v
    p = rec.get("source_path")
    if isinstance(p, str) and p: return [p]
    return []

# ===== 1) 데이터 적재 =====
xml_rows = load_jsonl(XML_JSONL)
ocr_rows = load_jsonl(OCR_JSONL)

# join_key 보정(없으면 생성), 기본 통계
def ensure_keys(rows, kind):
    fixed = 0
    for r in rows:
        if not r.get("doc_id") and r.get("doc"):
            r["doc_id"] = r["doc"]
        if "join_key" not in r or not r["join_key"]:
            r["join_key"] = make_join_key(r.get("doc_id",""))
            fixed += 1
        r["_kind"] = kind
    return fixed

fixed_xml = ensure_keys(xml_rows, "xml")
fixed_ocr = ensure_keys(ocr_rows, "ocr")

# ===== 2) only-in-OCR / only-in-XML (join_key 기준) =====
xml_by_jk = defaultdict(list)
ocr_by_jk = defaultdict(list)
for r in xml_rows: xml_by_jk[r["join_key"]].append(r)
for r in ocr_rows: ocr_by_jk[r["join_key"]].append(r)

xml_jks = set(xml_by_jk.keys())
ocr_jks = set(ocr_by_jk.keys())

only_in_ocr = sorted(ocr_jks - xml_jks)
only_in_xml = sorted(xml_jks - ocr_jks)
intersect    = sorted(xml_jks & ocr_jks)

# ===== 3) join_key 충돌 탐지 (동일 join_key에 다른 doc_id가 2개 이상) =====
#    └ 전체(두 소스 합친) 기준으로 본다.
all_by_jk = defaultdict(lambda: defaultdict(list))   # jk -> doc_id -> [rows...]
for r in xml_rows + ocr_rows:
    all_by_jk[r["join_key"]][r.get("doc_id","")].append(r)

collision_groups = {jk: grp for jk, grp in all_by_jk.items() if len(grp.keys()) > 1}

# ===== 4) 충돌 해소용 merge_key 생성 규칙 =====
# - 충돌 없는 jk: merge_key = join_key 유지
# - 충돌 있는 jk: 각 doc_id에 고유 접미사 _{md5(doc_id)[:6]} 부여
#   단, 대표 1개(doc_id)는 원키 유지(선호도: xml에 존재하는 doc_id가 있으면 그 중 lexicographically min)
fix_map = {}   # (join_key, doc_id) -> merge_key
for jk, by_id in all_by_jk.items():
    ids = sorted(by_id.keys())
    if len(ids) == 1:
        fix_map[(jk, ids[0])] = jk
        continue
    # 대표 선택: xml 존재하는 doc_id들 우선, 그중 사전순 최솟값
    xml_ids = sorted([i for i in ids if any(r["_kind"]=="xml" for r in by_id[i])])
    representative = (xml_ids[0] if xml_ids else ids[0])
    for i in ids:
        if i == representative:
            fix_map[(jk, i)] = jk
        else:
            fix_map[(jk, i)] = f"{jk}_{short_hash(i)}"

# ===== 5) merge_key 적용해 새 JSONL 작성 =====
def write_with_merge_key(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            jk = r["join_key"]
            did = r.get("doc_id","")
            mk  = fix_map.get((jk, did), jk)  # 기본은 원키
            r2 = dict(r)
            r2["merge_key"] = mk
            # 힌트: 충돌로 인해 바뀐 경우 이유/대표 여부 기록
            if mk != jk:
                r2["join_fix_reason"] = "collision"
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            wrote += 1
    return wrote

xml_wrote = write_with_merge_key(xml_rows, XML_OUT)
ocr_wrote = write_with_merge_key(ocr_rows, OCR_OUT)

# ===== 6) 리포트 파일 작성 =====
def head_paths(rows, k=2):
    acc = []
    for r in rows[:k]:
        ps = get_paths_any(r)
        acc.append(ps[0] if ps else "")
    return [p for p in acc if p]

lines = []
lines.append(f"[소스] xml_rows={len(xml_rows)} (join_key auto-fixed={fixed_xml}), ocr_rows={len(ocr_rows)} (join_key auto-fixed={fixed_ocr})")
lines.append(f"[분포] intersect={len(intersect)}  only_in_ocr={len(only_in_ocr)}  only_in_xml={len(only_in_xml)}")
lines.append("")

lines.append("[ONLY IN OCR] (join_key 기준)")
for jk in only_in_ocr:
    dids = sorted({r.get("doc_id","") for r in ocr_by_jk[jk]})
    paths = head_paths(ocr_by_jk[jk], k=2)
    lines.append(f" - {jk} | doc_id={dids} | sample_paths={paths}")
lines.append("")

lines.append("[ONLY IN XML] (join_key 기준)")
for jk in only_in_xml:
    dids = sorted({r.get("doc_id","") for r in xml_by_jk[jk]})
    paths = head_paths(xml_by_jk[jk], k=2)
    lines.append(f" - {jk} | doc_id={dids} | sample_paths={paths}")
lines.append("")

lines.append(f"[join_key 충돌] groups={len(collision_groups)}")
for jk, grp in collision_groups.items():
    ids = sorted(grp.keys())
    preview = []
    for i in ids:
        kinds = sorted({r["_kind"] for r in grp[i]})
        ch = None
        # 정보 힌트: 소스별 문자수/이미지수 있으면 보여주기
        for r in grp[i]:
            ch = ch or r.get("chars") or r.get("char_count") or r.get("xml_chars")
        preview.append(f"{i} (src={kinds}, chars≈{ch}) → merge_key={fix_map[(jk,i)]}")
    lines.append(f" - {jk}\n    " + "\n    ".join(preview))

AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")

# CSV(머신가독)도 덤으로
with AUDIT_CSV.open("w", encoding="utf-8") as f:
    f.write("join_key,side,doc_id,merge_key,sample_path\n")
    for jk in sorted(all_by_jk.keys()):
        for side, bucket in (("xml", xml_by_jk), ("ocr", ocr_by_jk)):
            rows = bucket.get(jk, [])
            if not rows:
                # 반대편에만 있는 경우도 한 줄 남김
                f.write(f"{jk},{side},,,\n")
                continue
            for r in rows:
                did = r.get("doc_id","")
                mk  = fix_map.get((jk, did), jk)
                pth = ""
                ps = get_paths_any(r)
                if ps: pth = ps[0]
                f.write(f"{jk},{side},{did},{mk},{pth}\n")

# fixes.json (머지 스크립트에서 사용할 수 있게)
fixes_payload = []
for (jk,did), mk in sorted(fix_map.items()):
    entry = {
        "join_key": jk,
        "doc_id": did,
        "merge_key": mk,
        "changed": mk != jk
    }
    # 소스 힌트
    kinds = sorted({r["_kind"] for r in all_by_jk[jk][did]})
    entry["sources"] = kinds
    fixes_payload.append(entry)

with FIXES_JSON.open("w", encoding="utf-8") as f:
    json.dump(fixes_payload, f, ensure_ascii=False, indent=2)

print("[OK] Audit & Fix complete")
print(f" - XML OUT: {XML_OUT}")
print(f" - OCR OUT: {OCR_OUT}")
print(f" - AUDIT TXT: {AUDIT_TXT}")
print(f" - AUDIT CSV: {AUDIT_CSV}")
print(f" - FIXES JSON: {FIXES_JSON}")
print(f" - collisions: {len(collision_groups)} | only_in_ocr: {len(only_in_ocr)} | only_in_xml: {len(only_in_xml)}")


# #### 개선 포인트
# 
# - `collisions: 4`
#     
#     → **같은 `join_key`를 두 개 이상 서로 다른 `doc_id`가 공유한 경우가 4건**뿐이라는 뜻. 충돌 자체는 많지 않음.
#     
# - `only_in_ocr: 59`, `only_in_xml: 60`
#     
#     → **OCR 쪽 `join_key`와 XML 쪽 `join_key`가 서로 거의 안 맞는다**는 시그널.
#     
#     바로 직전에 계산했던 `xml docs=64 / ocr docs=67 / intersection=63`과 완전히 상충하는 상황
#     
#     이유는 **이번 스크립트가 기존 JSONL에 들어있던 ``join_key``를 존중(보존)**했기 때문임
#     
#     → 즉, 이전 단계에서 XML과 OCR 각각이 **서로 다른 규칙**으로 `join_key`를 만들어둔 상태라서, 새로 만든 통일 규칙(`make_join_key`)이 적용되지 않은 항목들이 그대로 남아 있고, 그래서 `only_in_*`가 크게 나온 것
#     
# 
# - 결론: **`join_key`를 양쪽 모두 같은 규칙으로 강제 리빌드**해야 교집합이 정상(네가 봤던 63 전후)으로 돌아와.

# **[빠른 해결 (패치)]**
# 
# - ``ensure_keys``에 ``force=True``를 넣어 기존 join_key를 무시하고 재계산하도록 돌리기
# - 그 외 로직은 그대로 두고 재실행하면 출력 두 파일(*.with_mergekey.jsonl)이 갱신됨

# In[ ]:


# -*- coding: utf-8 -*-
from pathlib import Path
import json, re, hashlib
from collections import defaultdict, Counter

# ===== 경로 =====
BASE        = Path("/home/spai0308/data")
INTERIM     = BASE / "interim"
XML_JSONL   = INTERIM / "xml_text.jsonl"
OCR_JSONL   = INTERIM / "ocr_text_by_doc.jsonl"

AUDIT_TXT   = INTERIM / "join_audit.txt"
AUDIT_CSV   = INTERIM / "join_audit.csv"
FIXES_JSON  = INTERIM / "joinkey_fixes.json"

# 출력(원본 보존; INPLACE=True로 바꾸면 덮어씀)
INPLACE     = False
XML_OUT     = XML_JSONL if INPLACE else XML_JSONL.with_name(XML_JSONL.stem + ".with_mergekey.jsonl")
OCR_OUT     = OCR_JSONL if INPLACE else OCR_JSONL.with_name(OCR_JSONL.stem + ".with_mergekey.jsonl")
# ▼ 추가: 강제 재계산 스위치
FORCE_REBUILD_JOIN_KEY = True


# ===== 조인키 규칙(이전에 썼던 normalize와 동일/유사) =====
def make_join_key(s: str) -> str:
    if not s: return ""
    t = s
    t = t.replace("\u200b","")                             # zero-width 제거
    t = re.sub(r"[(){}\[\]<>]+"," ", t)                    # 괄호류 제거→공백
    t = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+","_",t)  # 한글/영문/숫자/_만
    t = re.sub(r"_+","_",t).strip("_")
    t = t.lower()
    return t

def short_hash(s: str, n=6) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]

def load_jsonl(p: Path):
    rows = []
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 방어적으로 스킵
                pass
    return rows

def get_paths_any(rec):
    # 소스별 필드 이름이 다를 수 있어 가능한 후보 모두 시도
    for k in ("source_paths","paths","xml_paths","image_paths","source_paths_sample"):
        v = rec.get(k)
        if isinstance(v, list) and v: return v
    p = rec.get("source_path")
    if isinstance(p, str) and p: return [p]
    return []

# ===== 1) 데이터 적재 =====
xml_rows = load_jsonl(XML_JSONL)
ocr_rows = load_jsonl(OCR_JSONL)



# join_key 보정(없으면 생성), 기본 통계
def ensure_keys(rows, kind, force=False):
    fixed = 0
    for r in rows:
        # doc_id 정리(양끝 공백 제거)
        if not r.get("doc_id") and r.get("doc"):
            r["doc_id"] = r["doc"]
        if r.get("doc_id"):
            r["doc_id"] = str(r["doc_id"]).strip()

        # ▼ 핵심: force=True면 기존 join_key를 버리고 새로 만든다
        if force or ("join_key" not in r) or not r["join_key"]:
            r["join_key"] = make_join_key(r.get("doc_id",""))
            fixed += 1
        r["_kind"] = kind
    return fixed

# 기존 호출을 아래처럼 변경
fixed_xml = ensure_keys(xml_rows, "xml", force=FORCE_REBUILD_JOIN_KEY)
fixed_ocr = ensure_keys(ocr_rows, "ocr", force=FORCE_REBUILD_JOIN_KEY)

# ===== 2) only-in-OCR / only-in-XML (join_key 기준) =====
xml_by_jk = defaultdict(list)
ocr_by_jk = defaultdict(list)
for r in xml_rows: xml_by_jk[r["join_key"]].append(r)
for r in ocr_rows: ocr_by_jk[r["join_key"]].append(r)

xml_jks = set(xml_by_jk.keys())
ocr_jks = set(ocr_by_jk.keys())

only_in_ocr = sorted(ocr_jks - xml_jks)
only_in_xml = sorted(xml_jks - ocr_jks)
intersect    = sorted(xml_jks & ocr_jks)

# ===== 3) join_key 충돌 탐지 (동일 join_key에 다른 doc_id가 2개 이상) =====
#    └ 전체(두 소스 합친) 기준으로 본다.
all_by_jk = defaultdict(lambda: defaultdict(list))   # jk -> doc_id -> [rows...]
for r in xml_rows + ocr_rows:
    all_by_jk[r["join_key"]][r.get("doc_id","")].append(r)

collision_groups = {jk: grp for jk, grp in all_by_jk.items() if len(grp.keys()) > 1}

# ===== 4) 충돌 해소용 merge_key 생성 규칙 =====
# - 충돌 없는 jk: merge_key = join_key 유지
# - 충돌 있는 jk: 각 doc_id에 고유 접미사 _{md5(doc_id)[:6]} 부여
#   단, 대표 1개(doc_id)는 원키 유지(선호도: xml에 존재하는 doc_id가 있으면 그 중 lexicographically min)
fix_map = {}   # (join_key, doc_id) -> merge_key
for jk, by_id in all_by_jk.items():
    ids = sorted(by_id.keys())
    if len(ids) == 1:
        fix_map[(jk, ids[0])] = jk
        continue
    # 대표 선택: xml 존재하는 doc_id들 우선, 그중 사전순 최솟값
    xml_ids = sorted([i for i in ids if any(r["_kind"]=="xml" for r in by_id[i])])
    representative = (xml_ids[0] if xml_ids else ids[0])
    for i in ids:
        if i == representative:
            fix_map[(jk, i)] = jk
        else:
            fix_map[(jk, i)] = f"{jk}_{short_hash(i)}"

# ===== 5) merge_key 적용해 새 JSONL 작성 =====
def write_with_merge_key(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            jk = r["join_key"]
            did = r.get("doc_id","")
            mk  = fix_map.get((jk, did), jk)  # 기본은 원키
            r2 = dict(r)
            r2["merge_key"] = mk
            # 힌트: 충돌로 인해 바뀐 경우 이유/대표 여부 기록
            if mk != jk:
                r2["join_fix_reason"] = "collision"
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            wrote += 1
    return wrote

xml_wrote = write_with_merge_key(xml_rows, XML_OUT)
ocr_wrote = write_with_merge_key(ocr_rows, OCR_OUT)

# ===== 6) 리포트 파일 작성 =====
def head_paths(rows, k=2):
    acc = []
    for r in rows[:k]:
        ps = get_paths_any(r)
        acc.append(ps[0] if ps else "")
    return [p for p in acc if p]

lines = []
lines.append(f"[소스] xml_rows={len(xml_rows)} (join_key auto-fixed={fixed_xml}), ocr_rows={len(ocr_rows)} (join_key auto-fixed={fixed_ocr})")
lines.append(f"[분포] intersect={len(intersect)}  only_in_ocr={len(only_in_ocr)}  only_in_xml={len(only_in_xml)}")
lines.append("")

lines.append("[ONLY IN OCR] (join_key 기준)")
for jk in only_in_ocr:
    dids = sorted({r.get("doc_id","") for r in ocr_by_jk[jk]})
    paths = head_paths(ocr_by_jk[jk], k=2)
    lines.append(f" - {jk} | doc_id={dids} | sample_paths={paths}")
lines.append("")

lines.append("[ONLY IN XML] (join_key 기준)")
for jk in only_in_xml:
    dids = sorted({r.get("doc_id","") for r in xml_by_jk[jk]})
    paths = head_paths(xml_by_jk[jk], k=2)
    lines.append(f" - {jk} | doc_id={dids} | sample_paths={paths}")
lines.append("")

lines.append(f"[join_key 충돌] groups={len(collision_groups)}")
for jk, grp in collision_groups.items():
    ids = sorted(grp.keys())
    preview = []
    for i in ids:
        kinds = sorted({r["_kind"] for r in grp[i]})
        ch = None
        # 정보 힌트: 소스별 문자수/이미지수 있으면 보여주기
        for r in grp[i]:
            ch = ch or r.get("chars") or r.get("char_count") or r.get("xml_chars")
        preview.append(f"{i} (src={kinds}, chars≈{ch}) → merge_key={fix_map[(jk,i)]}")
    lines.append(f" - {jk}\n    " + "\n    ".join(preview))

AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")

# CSV(머신가독)도 덤으로
with AUDIT_CSV.open("w", encoding="utf-8") as f:
    f.write("join_key,side,doc_id,merge_key,sample_path\n")
    for jk in sorted(all_by_jk.keys()):
        for side, bucket in (("xml", xml_by_jk), ("ocr", ocr_by_jk)):
            rows = bucket.get(jk, [])
            if not rows:
                # 반대편에만 있는 경우도 한 줄 남김
                f.write(f"{jk},{side},,,\n")
                continue
            for r in rows:
                did = r.get("doc_id","")
                mk  = fix_map.get((jk, did), jk)
                pth = ""
                ps = get_paths_any(r)
                if ps: pth = ps[0]
                f.write(f"{jk},{side},{did},{mk},{pth}\n")

# fixes.json (머지 스크립트에서 사용할 수 있게)
fixes_payload = []
for (jk,did), mk in sorted(fix_map.items()):
    entry = {
        "join_key": jk,
        "doc_id": did,
        "merge_key": mk,
        "changed": mk != jk
    }
    # 소스 힌트
    kinds = sorted({r["_kind"] for r in all_by_jk[jk][did]})
    entry["sources"] = kinds
    fixes_payload.append(entry)

with FIXES_JSON.open("w", encoding="utf-8") as f:
    json.dump(fixes_payload, f, ensure_ascii=False, indent=2)

print("[OK] Audit & Fix complete")
print(f" - XML OUT: {XML_OUT}")
print(f" - OCR OUT: {OCR_OUT}")
print(f" - AUDIT TXT: {AUDIT_TXT}")
print(f" - AUDIT CSV: {AUDIT_CSV}")
print(f" - FIXES JSON: {FIXES_JSON}")
print(f" - collisions: {len(collision_groups)} | only_in_ocr: {len(only_in_ocr)} | only_in_xml: {len(only_in_xml)}")


# - **only_in_ocr: 0** → OCR 쪽에 있는 모든 문서는 XML 쪽과 **1:1 매칭**됨.
# - **only_in_xml: 1** → XML 쪽 문서 중 **단 1건만** OCR 짝이 없음(이미지가 없거나, 집계 단계에서 텍스트가 비어 스킵됐을 가능성).
# - **collisions: 4** → 동일 `join_key`를 공유한 서로 다른 `doc_id`가 **4그룹**. 이건 코드가 `merge_key`로 안전하게 분기해놨으니 그대로 조인하면 됨(대표 1개는 원키 유지, 나머지는 `_md5(6)` 접미사).
# 
# > 조인키 정합성은 거의 끝났고, 나중 병합은 **`merge_key` 기준**으로 하면 안전하게 붙음

# In[ ]:


# 매칭 현황 + only_in_xml 상세보기

from pathlib import Path
import json

INTERIM = Path("/home/spai0308/data/interim")
XML_OUT = INTERIM / "xml_text.with_mergekey.jsonl"
OCR_OUT = INTERIM / "ocr_text_by_doc.with_mergekey.jsonl"

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

xml = load_jsonl(XML_OUT)
ocr = load_jsonl(OCR_OUT)

xml_keys = {r["merge_key"] for r in xml}
ocr_keys = {r["merge_key"] for r in ocr}

print("xml keys:", len(xml_keys))
print("ocr keys:", len(ocr_keys))
print("intersection:", len(xml_keys & ocr_keys))

only_xml_keys = sorted(xml_keys - ocr_keys)
only_ocr_keys = sorted(ocr_keys - xml_keys)
print("only_in_xml:", len(only_xml_keys))
print("only_in_ocr:", len(only_ocr_keys))

# only_in_xml 상세
for mk in only_xml_keys:
    items = [r for r in xml if r["merge_key"] == mk]
    r0 = items[0]
    doc_id = r0.get("doc_id")
    chars = r0.get("xml_chars") or r0.get("chars") or r0.get("char_count")
    paths = r0.get("source_paths") or r0.get("xml_paths") or r0.get("paths") or []
    if not paths and r0.get("source_path"):
        paths = [r0["source_path"]]
    print("\n[ONLY_IN_XML]")
    print(" merge_key :", mk)
    print(" doc_id    :", doc_id)
    print(" chars     :", chars)
    print(" sample_path:", (paths[0] if paths else ""))


# In[ ]:


# join_key 충돌 4건 자세히 보기

from collections import defaultdict

def bucket_by_join_key(rows):
    d = defaultdict(lambda: defaultdict(list))  # jk -> doc_id -> [rows...]
    for r in rows:
        jk = r.get("join_key")
        did = r.get("doc_id", "")
        d[jk][did].append(r)
    return d

all_by_jk = bucket_by_join_key(xml + ocr)
collisions = {jk: grp for jk, grp in all_by_jk.items() if len(grp) > 1}
print("collision groups:", len(collisions))

for jk, grp in collisions.items():
    print(f"\n[join_key] {jk}")
    for did, bucket in sorted(grp.items()):
        sources = sorted({b.get("_kind", "?") for b in bucket})
        mk = bucket[0].get("merge_key")
        ch = None
        for b in bucket:
            ch = ch or b.get("xml_chars") or b.get("chars") or b.get("char_count")
        print(f" - doc_id={did} | sources={sources} | merge_key={mk} | chars≈{ch}")


# #### 결과:
# 
# - `xml_text.with_mergekey.jsonl` : 64 docs
# - `ocr_text_by_doc.with_mergekey.jsonl` : 67 docs
# - 교집합 63 → 최종 유니온 예상 68개
# - 충돌 4건은 `merge_key`로 안전하게 분리됨(해시 접미사 부여).
# - `only_in_xml` 1건, `only_in_ocr` 4건 존재 → 머지 시 해당 소스만 가진 문서로 포함.

# ## 3-3. 문서별 통합 텍스트 머지

# #### 지금까지 확보한 소스 (정리)
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

# In[ ]:


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


# # 4. Issues

# ## 4-1. ``docs_merged.jsonl`` 파일 text 41개 누락

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

# ### 문제 : text 필드 69개 --> 현황 확인 필요

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


# ### 현황 요약
# 
# 1. 전체 문서
#     - 총 110건
# 2. merged 텍스트 필드(`texts.merged`)
#     - 내용 있음: 69건
#       - “merged 필드가 69개” == **`texts.merged`가 채워진 문서가 69건**
#         - 이는 **`chars.merged > 0` 69건**과 정확히 일치 → 데이터 일관성 OK.
#     - 내용 없음/미존재(빈 문자열 포함): 41건
#       - 나머지 **41건은 `texts` 하위의 모든 소스(xml/html/ocr/pdf/merged)가 비어 있음**
# 
#         → 실질 텍스트가 없어 **청킹 불가 대상**.
# 
# 3. 병합 텍스트 길이 지표(`chars.merged`)
#     - `> 0`: 69건 (위 ②와 정확히 일치)
#     - `= 0`: 41건
#     - 누락: 0건
# 
# 
# 4. 기타 병합 필드(`merged_text`)
#    - 사용되지 않음(존재/값 모두 0)
#    - `merged_text` 필드는 현재 파이프라인에서 **아예 쓰이지 않음**(0건).
#    - 즉, 병합 결과는 `texts.merged`/`chars.merged`로만 관리 중.
# 
# 
# 5. 어떤 소스든 텍스트가 있는지(`texts.{xml,html,ocr,pdf,merged}` 중 하나라도 비어있지 않음)
#     - 있음: 69건
#     - 없음: 41건
# 
# 
# ----------------------
# 
# #### 41건이 빈 이유(가능성)
# 
# - 원문 파싱 실패(HTML/XML 추출 실패, OCR 미수행/실패)
# - 원본이 스캔 이미지인데 OCR 단계 누락
# - 소스 파일 경로나 접근권한 문제
# - 전처리에서 필터(예: 극단적 짧은 텍스트)로 제거
# 
# --------------------------
# 
# #### 바로 해볼 점검
# 
# - 빈 41건의 `merge_key` 목록 추출해 원본 경로(`source_paths_sample`) 확인
# - `sources`/`counts`/`chars`를 같이 찍어 **어느 단계에서 끊겼는지** 구분
#     - (예: `pdf_records>0`인데 `ocr_images_total=0`이면 OCR 누락)
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


# [해결 방법]
# - source_flag_only가 41건이면
# - “소스 플래그는 있는데 ``texts/*``랑 ``chars/*``가 비어 있음” 케이스
# - 따라서, 샘플 경로에서 원문을 다시 읽어 채워 넣고(texts.*, chars.*, merged) 복구
# 

# #### 샘플경로 복구 작업

# - source_paths_sample[0]에서 텍스트 파일을 읽는다.
# - 경로/플래그로 **소스 타입(html/xml/ocr/pdf)**를 추정한다.
# - texts[src_type], texts["merged"]를 채우고,
# - chars[src_type], chars["merged"]를 각 길이로 채운다.
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


# #### other 4개 추가 복구

# - 2차 복구 스크립트는 ``/processed/html_plain/*.txt`` 같은 “평문 txt”만 뒤졌고
# - 미해결 4건은 ``/home/spai0308/data/raw/converted_personal_test/{html,xml}/…/*.html|*.xml`` 폴더 구조에 있어서 후보 인덱스에 안 잡혔음 == 문제의 원인
# 
# 
# - 아래 ``v2.1 복구 스크립트``는 다음을 추가로 지원함
# ---------
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

# In[ ]:


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

# ## 5-1. 1차 2025.09.25 : 청킹 인덱스, 임베딩 처리 이후(코사인 유사도 이슈로 사용 x) 

# - `/processed/`에 `docs_merged.jsonl`, `docs_chunks.jsonl`, 리포트(`.txt/*.csv`),
#     
#     표 인덱스(`tables_index.jsonl` or `rep_pdf.jsonl` 내 포함) 정리.

# In[ ]:


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

# ## 5-2. 아티팩트 병합 & 정리(2025.09.29)

# ### docs_merged.jsonl 구조/품질 재구성(누락된 text 41 포함) 후

# - 오늘자 release 아래 디렉터리 생성
# - 필요한 파일 복사/정규화
# - 점검 리포트/체크섬/매니페스트 생성

# **[정리되는 파일 리스트]**
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
# - ``MANIFEST.csv``
#     → 어떤 파일이 어디에 복사되었는지 일람 확인.
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

# In[ ]:


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




