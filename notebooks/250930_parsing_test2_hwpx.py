#!/usr/bin/env python
# coding: utf-8

# # 기본세팅

# ## 가상환경 활성화

# ## 깃세팅

# ## 백업  파일 생성

# # DataLoad

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


# 개인폴더 넣고 실행
FOLDER_LINK = "```폴더 링크 넣기```"
download_drive_folder(FOLDER_LINK, RAW)


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


# In[ ]:


# CSV ↔ 디렉토리 교차검증 (정규화 적용)

# 1. CSV 로드
df = pd.read_csv(CSV_PATH)
print("CSV 컬럼:", list(df.columns))
assert "파일명" in df.columns, "'파일명' 컬럼이 없습니다."
# '파일형식' 컬럼이 없으면 확장자 추정은 건너뜀

# 2. CSV 기대 파일명(정규화) 목록
csv_names_norm = [normalize_csv_row(r) for _, r in df.iterrows()]
csv_set = set(csv_names_norm)

# 3. 디렉토리 실제 파일명(재귀, 정규화)
dir_names_norm = [normalize_name(p.name) for p in FILES_DIR.rglob("*") if p.is_file()]
dir_set = set(dir_names_norm)

# 4. 차집합 계산
missing_in_dir = sorted(csv_set - dir_set)   # CSV엔 있는데 폴더엔 없음
extra_in_dir   = sorted(dir_set - csv_set)   # 폴더엔 있는데 CSV엔 없음

print(f"\nCSV 고유 파일 수: {len(csv_set)}")
print(f"디렉토리 파일 수: {len(dir_set)}")
print(f"누락(폴더에 없음): {len(missing_in_dir)}개")
print(f"초과(CSV에 없음): {len(extra_in_dir)}개")

print("\n[누락 예시 상위 10]")
for n in missing_in_dir[:10]:
    print(" -", n)

print("\n[초과 예시 상위 10]")
for n in extra_in_dir[:10]:
    print(" -", n)


# In[ ]:


# CSV 중복 파일명 점검
dups = pd.Series(csv_names_norm).value_counts()
dups = dups[dups > 1]
if len(dups) > 0:
    print("\nCSV 중복 파일명 상위 10")
    print(dups.head(10))
else:
    print("\nCSV 중복 파일명 없음")


# # 통합 JSONL 만들기

# ## 공통 준비 & CSV 매핑 (정규화 키)

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


# ## PDF 파싱(페이지 단위)

# In[ ]:


# PDF 파싱 (페이지 단위로)

import fitz #PyMuPDF

pdf_rows, pdf_fail = [], []

for p in FILES_DIR.rglob("*.pdf"):
    key = normalize_name(p.name)   #파일명 정규화키로
    meta = csv_map.get(key, {})   # 해당파일의 csv 메타를 붙임
    try:
        doc = fitx.open(p)
        for i, page in enumerate(doc):
            text = page.get_text("test") or ""   #페이지 텍스트
            pdf_rows.append({
                "doc_id" : p.stem,   # 파일명(확장자 제외) -> 간결한 id
                "page" : i+1,      # 인용을 위한 페이지 번호
                "text": text,
                "source": str(p),    # 원본 경로(디버깅, 링크용)
                "format": "pdf",
                "meta": meta    # 메타 통째로 부착함(기관, 사업, 금액, 기한 등)
            })

    except Exception as e:
        pdf_fail.append({"file": str(p), "error": str(e)})


# jsonl: 라인 단위 append가 편하고, 대용량 스트리밍에 강함!
with open(INTERIM / "pages_pdf.jsonl", "w", encoding="utf-8") as f:
    for r in pdf_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            


# ## 수정 : pdf, hwp 모두 파싱 안됨

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


# ---> 공통 유틸 : 파일명 정규화 + csv 매핑

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


# pdf 파싱 강제 재빌드 : 빈 파일이라도 생성함

import fitz, os

pdf_rows, pdf_fail = [], []
pdf_list = list(FILES_DIR.rglob("*.pdf"))
print("PDF files found:", len(pdf_list))

for p in pdf_list:
    key = normalize_name(p.name)
    meta = csv_map.get(key, {})
    try:
        doc = fitz.open(p)
        for i, page in enumerate(doc):
            # 기본 모드 실패 대비 간단 폴백
            text = page.get_text("text")
            if not text:
                text = page.get_text() or ""
            pdf_rows.append({
                "doc_id": p.stem,
                "page": i+1,
                "text": text,
                "source": str(p),
                "format": "pdf",
                "meta": meta
            })
    except Exception as e:
        pdf_fail.append({"file": str(p), "error": str(e)})

# 결과 저장 (0줄이어도 파일은 만듦)
pdf_out = INTERIM / "pages_pdf.jsonl"
with open(pdf_out, "w", encoding="utf-8") as f:
    for r in pdf_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[PDF] rows={len(pdf_rows)} | fails={len(pdf_fail)} | saved→ {pdf_out}")
if pdf_fail:
    pd.DataFrame(pdf_fail).to_csv(INTERIM / "pdf_parse_failures.csv", index=False)
    print("PDF 실패 목록:", INTERIM / "pdf_parse_failures.csv")


# In[ ]:


# 어떤 파일에서 경고가 나왔는지 수집해서 보고

import os, sys, contextlib, tempfile, fitz, json, pandas as pd
from pathlib import Path

@contextlib.contextmanager
def capture_stderr():   # stderr를 임시파일로 리다이렉트 → 내용 읽어옴
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
        with capture_stderr() as errbuf:     # <-- 경고 캡처
            doc = fitz.open(p)
            for i, page in enumerate(doc):
                text = page.get_text("text") or page.get_text() or ""
                pdf_rows.append({
                    "doc_id": p.stem, "page": i+1, "text": text,
                    "source": str(p), "format": "pdf", "meta": meta
                })
        errbuf.flush(); errbuf.seek(0)
        msg = errbuf.read().decode("utf-8", "ignore").strip()
        if msg:
            warn_log.append({"file": str(p), "stderr": msg[:2000]})
    except Exception as e:
        pdf_fail.append({"file": str(p), "error": str(e)})

# 저장
pdf_out = INTERIM / "pages_pdf.jsonl"
with open(pdf_out, "w", encoding="utf-8") as f:
    for r in pdf_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[PDF] rows={len(pdf_rows)} | fails={len(pdf_fail)} | saved→ {pdf_out}")
if pdf_fail:
    pd.DataFrame(pdf_fail).to_csv(INTERIM / "pdf_parse_failures.csv", index=False)
    print("PDF 실패 목록:", INTERIM / 'pdf_parse_failures.csv')
if warn_log:
    pd.DataFrame(warn_log).to_csv(INTERIM / "pdf_parse_warnings.csv", index=False)
    print("경고 로그 저장:", INTERIM / "pdf_parse_warnings.csv")


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

print("== FILE CHECK ==")
info(pdf_path)
info(hwp_path)
info(all_path)


# ## hwp 포맷변환(테스트)

# ### hwp →  hwpt

# [전략]
# 1. hwp -> hwpx
# 2. 코드에서는 hwpx 우선  파싱

# - 필요 라이브러리 설치
# - hwp -> hwpt 변환 & 업로드
# - 디렉터리 구조 생성(포맷별 - 파일명 디렉터리 - (html, xml은 assets으로 이미지 파일 별도 구분)

# In[ ]:


# beautifulsoup4 : html 파싱
# lxml : xml 파싱 가속(표준 xml.etree도 가능, 하지만 성능/안정성이 높음)
# !pip install beautifulsoup4 lxml

# # error 발생으로 terminal에서 가상환경에 설치
# source ~/venvs/rag/bin/activate
# pip install --upgrade pip
# pip install beautifulsoup4 lxml ipykernel


# In[ ]:


# 2) 설치 정상 확인(terminal, jupyter python 둘다)
# !which python
# !python -c


# In[ ]:


# windows + 한컴 한글 설치 (자동 일괄 변환 스크립트)

# batch_hwp_to_hwpx_html.py
# (로컬에서만 가능)windows에서 한컴 한글이 설치되어 있어야 함 (COM 자동화)
# 나중에 팀 작업할 때, 한명 window socket을 vm instance와 연결하기


# In[ ]:


# 포맷 버전별 파일 생성
# hwpx, html, xml, doc, pdf

get_ipython().system('mkdir ~/data/raw/converted/{hwpx,html,xml,docs,pdf}')
# {}안에 꼭 붙여써야 4가지 디렉토리가 각각 생성됨


# In[ ]:


get_ipython().system('ls -R ~/data/raw/converted')


# In[ ]:


# 이제 로컬에서 포맷 변환한 파일들 업로드 : 총 4개


# In[ ]:


# html, xml로 변환된 파일의 이미지 파일들의 텍스트를
# 읽어와야 하기에 디렉터리 구조 추가


# In[ ]:


get_ipython().system('ls ~/data/raw/converted/')


# In[ ]:


get_ipython().system('mkdir -p ~/data/raw/converted/html/{한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보,한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축,한국한의학연구원_통합정보시스템_고도화_용역,한국철도공사_용역_예약발매시스템_개량_ISMP_용역}')
get_ipython().system('mkdir -p ~/data/raw/converted/xml/{한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보,한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축,한국한의학연구원_통합정보시스템_고도화_용역,한국철도공사_용역_예약발매시스템_개량_ISMP_용역}')


# In[ ]:


get_ipython().system('ls ~/data/raw/converted/html')
print("="*5)
get_ipython().system('ls ~/data/raw/converted/xml')


# In[ ]:


get_ipython().system('mkdir -p ~/data/raw/converted/html/한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국한의학연구원_통합정보시스템_고도화_용역/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국철도공사_용역_예약발매시스템_개량_ISMP_용역/assets')


# In[ ]:


get_ipython().system('mkdir -p ~/data/raw/converted/xml/한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/xml/한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/xml/한국한의학연구원_통합정보시스템_고도화_용역/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/xml/한국철도공사_용역_예약발매시스템_개량_ISMP_용역/assets')


# In[ ]:


# 트리구조로 ~/data 구조 확인
get_ipython().system('tree ~/data')

# 트리구조로 ~/data 구조 확인
get_ipython().system('tree -L 4 ~/data')

# 디렉터리만 트리구조로 확인 : 디렉터리 구조 제대로 나옴확인
get_ipython().system('tree -d ~/data')


# ### 1) HWPX -> page_hwpx.jsonl

# In[ ]:


#######사전 라이브러리 설치(터미널 가상환경에서)

# pip install beautifulsoup4 lxml tqdm pdfminer.six python-docx pytesseract opencv-python Pillow numpy
# # bs4+lxml: HTML/XML 파싱
# # tqdm: 진행바
# # pdfminer.six: PDF 텍스트 추출
# # python-docx: 필요 시 DOCX 파싱(지금은 사용 안 해도 괜찮음)
# # pytesseract/opencv/Pillow/numpy: OCR + 전처리
# sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-kor

# # tesseract 한글 데이터 설치 (OCR 필수)
  # tesseract 설치에 관리자 권한이 필요하여 외부 api 연결


# #### jupyterhub에서 홈 디렉토리에 API KEY를 환경변수 저장

# - 이후 필요할 때마다 불러오기
# - 이미지에서 텍스트를 추출하기 위함

# In[ ]:


# # terminal 저장법

# #1. python dotenv 설치(개인계정 가상환경에)
# source ~/myenv/bin/activate   # 개인 가상환경 활성화
# pip install python-dotenv

# # 홈 디렉토리에 .env 파일 만들기
# nano ~/.env
# # .env 에 등록
# OPENAI_API_KEY=[여기에 키 넣기]
# # ctrl+O -> enter -> ctrl+x -> enter 나오기
# # 권한 제어 (읽기만)
# chmod 600 ~/.env


# # 등록 확인
# cat ~/.env
# # 권한 확인
# ls -l ~/.env


# #### 막간 : api key 로드 코드

# In[ ]:


# api key 로드
from dotenv import load_dotenv
import os

# 홈의 ~/.env를 명시적으로 로드
#(작업 디렉토리가 어디든 확실히 로드됨)
load_dotenv(os.path.expanduser("~/.env"))

print(os.getenv("OPENAI_API_KEY"))  
# sk-... 나오면 성공


# #### 1-2) HWPX(.HWPX) 본문 텍스트 추출

# HWPX(.hwpx) 본문 텍스트 추출 -> JSONL 저장

# In[ ]:


import os, json, zipfile, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
import lxml.etree as ET


# ##### 스크립트 생성 

# In[ ]:


########## 경로 설정##########
# -*- coding: utf-8 -*-
"""
HWPX (.hwpx) 본문 텍스트 추출 → JSONL 저장
입력: /home/spai0308/data/raw/converted/hwpx/*.hwpx
출력: /home/spai0308/data/interim/pages_hwpx.jsonl
"""
HOME = Path.home()
CONVERTED = HOME / "data" / "raw" / "converted"
INTERIM = HOME / "data" / "interim"

OUT_HWPX = INTERIM / "pages_hwpx.jsonl"

KST = timezone(timedelta(hours=9))
def now_kst_iso():
    return datetime.now(KST).isoformat()

########## 파싱함수 ##########
def extract_hwpx_text(hwpx_path: Path) -> str:
    texts = []
    with zipfile.ZipFile(hwpx_path, "r") as z:
        # section*.xml 파일들만 순회
        for name in z.namelist():
            if name.startswith("Contents/section") and name.endswith(".xml"):
                xml_bytes = z.read(name)
                root = ET.fromstring(xml_bytes)
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        texts.append(elem.text.strip())
    return "\n".join(texts)

######### JSONL 저장 ##########
def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] {out_path} ({len(records)} records)")


def run_hwpx():
    records = []
    for path in glob.glob(str(CONVERTED / "hwpx" / "*.hwpx")):
        try:
            text = extract_hwpx_text(Path(path))
            records.append({
            "doc_id": Path(path).stem,
            "source_path": path,
            "text": text,
            "ts": now_kst_iso(),
            "source_type": "hwpx_text"
            })
        except Exception as e:
            records.append({
                "doc_id": Path(path).stem,
                "source_path": path,
                "error": str(e),
                "ts": now_kst_iso(),
                "source_type": "hwpx_text_error"
            })

    write_jsonl(records, OUT_HWPX)

if __name__ == "__main__":
    run_hwpx()


# - hwpx 변환본 4개 모두 성공적으로 파싱 --> 위 경로에 저장됨

# - hwpx 변환본 파싱 확인하기

# In[ ]:


# 줄 수 확인
get_ipython().system('wc -l /home/spai0308/data/interim/pages_hwpx.jsonl')


# In[ ]:


# 샘플 보기 : 첫번째 줄만
get_ipython().system('head -n 1 ~/data/interim/pages_hwpx.jsonl | jq .')


# In[ ]:


# 샘플 각 줄 씩 확인
get_ipython().system("sed -n '4p' /home/spai0308/data/interim/pages_hwpx.jsonl | jq .")

# sed -n 'Np' file : 파일에서 N번째 줄만 출력


# #### 1-3) HWPX 안의 이미지 추출 + 매니페스트

# ##### HWPX 이미지 추출기

# - 벡터 포맷은 그대로 저장
# - 비트맵은 시그니처(매직 넘버) 로도 탐지해서 ``PNG``/``JPG``..로 저장
# - 매니페스토에 ``saved_bitmap[]`` / ``saved_vector[]``를 구분기록해서
#   > OCR 대상 파악이 쉬워짐

# In[ ]:


# -*- coding: utf-8 -*-
"""
HWPX(zip) 내부 이미지/그림 추출 v2 (비트맵/벡터 구분 + 시그니처 판별)
- 비트맵: PNG/JPG/GIF/BMP/TIFF/WEBP 시그니처로 탐지하여 저장
- 벡터 : SVG/WMF/EMF 등은 그대로 파일로 저장 (OCR은 바로 불가, 후처리 필요)
출력:
  /home/spai0308/data/interim/hwpx_assets/<doc_id>/*.{png,jpg,svg,emf,wmf,...}
  /home/spai0308/data/interim/hwpx_image_manifest.jsonl
"""
import io, json, glob, zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA = Path("/home/spai0308/data")
HWPX_DIR = DATA / "raw" / "converted" / "hwpx"
OUT_DIR  = DATA / "interim" / "hwpx_assets"
MANIFEST = DATA / "interim" / "hwpx_image_manifest.jsonl"

KST = timezone(timedelta(hours=9))

# 벡터/비트맵 확장자 카탈로그
VEC_EXTS = {".svg", ".emf", ".wmf"}
RASTER_EXTS = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}
ALL_EXTS = VEC_EXTS | RASTER_EXTS

def detect_raster_by_magic(b: bytes):
    """파일 앞쪽 바이트로 비트맵 포맷을 추정하고 확장자를 돌려줌 (없으면 None)"""
    if len(b) < 12: return None
    # PNG
    if b[:8] == b"\x89PNG\r\n\x1a\n": return ".png"
    # JPEG
    if b[:2] == b"\xFF\xD8": return ".jpg"
    # GIF87a / GIF89a
    if b[:6] in (b"GIF87a", b"GIF89a"): return ".gif"
    # BMP
    if b[:2] == b"BM": return ".bmp"
    # TIFF
    if b[:4] in (b"II*\x00", b"MM\x00*"): return ".tif"
    # WEBP (RIFF....WEBP)
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP": return ".webp"
    return None

def extract_images_from_hwpx(hwpx_path: Path) -> dict:
    doc_id = hwpx_path.stem
    out_folder = OUT_DIR / doc_id
    out_folder.mkdir(parents=True, exist_ok=True)

    saved_bitmap, saved_vector = [], []

    with zipfile.ZipFile(hwpx_path, "r") as zf:
        names = zf.namelist()
        # Resources/ 우선
        candidates = [n for n in names if n.lower().startswith("resources/")] or names

        for idx, name in enumerate(sorted(candidates)):
            try:
                data = zf.read(name)
            except KeyError:
                continue

            p = Path(name)
            ext = p.suffix.lower()

            # 1) 벡터 확장자면 그대로 저장
            if ext in VEC_EXTS:
                out = out_folder / f"vec_{idx+1:04d}{ext}"
                out.write_bytes(data)
                saved_vector.append(str(out))
                continue

            # 2) 비트맵 확장자면 그대로 저장
            if ext in RASTER_EXTS:
                out = out_folder / f"img_{idx+1:04d}{ext}"
                out.write_bytes(data)
                saved_bitmap.append(str(out))
                continue

            # 3) 확장자가 애매하면 시그니처로 비트맵 판별 후 저장
            guessed = detect_raster_by_magic(data[:64])
            if guessed:
                out = out_folder / f"img_{idx+1:04d}{guessed}"
                out.write_bytes(data)
                saved_bitmap.append(str(out))
                continue

            # 4) 그 외는 무시 (텍스트/도형정의/기타 바이너리)
            continue

    return {
        "doc_id": doc_id,
        "source_path": str(hwpx_path),
        "image_count": len(saved_bitmap) + len(saved_vector),
        "bitmap_count": len(saved_bitmap),
        "vector_count": len(saved_vector),
        "saved_bitmap": saved_bitmap,
        "saved_vector": saved_vector,
        "ts": datetime.now(KST).isoformat(),
        "source_type": "hwpx_assets"
    }

def main():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as mf:
        for path in glob.glob(str(HWPX_DIR/"*.hwpx")):
            rec = extract_images_from_hwpx(Path(path))
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[OK] {Path(path).name}: {rec['bitmap_count']} bitmap, {rec['vector_count']} vector")

if __name__ == "__main__":
    main()


# In[ ]:


### 결과 확인
# hwpx 이미지 추출 산출물 폴더에 실제 파일 생성됐는지
get_ipython().system('wc -l /home/spai0308/data/interim/hwpx_image_manifest.jsonl')
print("="*20)

# 매니페스토가 있다면 1줄만 확인
get_ipython().system('head -n 1 /home/spai0308/data/interim/hwpx_image_manifest.jsonl | jq .')
print("="*20)

# 아무 폴더나 하나 골라 실제 확장자 확인
get_ipython().system('find /home/spai0308/data/interim/hwpx_assets -maxdepth 2 -type f | head -n 10')


# **[결과 요약]**
# - bitmap만 찍히고, vector_count가 없기 때문에
#   > 이미지는 PNG, JPG, bmp가 전부

# - **`image_count: 24`**
# - **`bitmap_count: 24`**
# - **`vector_count: 0`**
# - `saved_bitmap`에 `.jpg / .png / .bmp` 파일들이 쭉 기록됨

# **[결과]**
# 
# 
# 
# ##### 비트맵(bitmap) vs 벡터(vector)
# 
# - **비트맵**: 사진처럼 **픽셀로 이루어진 이미지**. 확장자 예) `JPG, PNG, BMP, GIF, TIFF, WEBP`
#     - 확대하면 계단처럼 깨짐.
#     - **OCR(글자 읽기)** 바로 가능
# - **벡터**: 선/도형/텍스트 명령어로 그린 **도면/일러스트**. 확장자 예) `SVG, EMF, WMF`
#     - 아무리 확대해도 안 깨짐.
#     - **그대로는 OCR 불가** → PNG 같은 비트맵으로 “렌더링(변환)”해야 OCR 가능.
# 
# 
# 
# 
# ##### **요약**
# 
# - **비트맵만 24장** 뽑혔고 **벡터는 0장**
#     
#     → **이번 문서는 OCR 대상이 되는 “사진형 이미지(비트맵)”만 존재**   
#     → `saved_bitmap`에 기록된 경로들이 그 24장 실제 파일이고, **바로 OCR에 먹일 수 있음**
#     
# 
# > 예:
# > 
# > 
# > `/home/spai0308/data/interim/hwpx_assets/…/img_0013.png` (바로 OCR 가능)
# > 
# > `/home/spai0308/data/interim/hwpx_assets/…/img_0016.bmp` (바로 OCR 가능)
# > 
# 
# 벡터가 아예 없으니 **SVG→PNG 변환 같은 추가작업은 필요 없음**.

# #### 1-3-1) hwpx 이미지 OCR (mini/nano 우선, 자동 승급)

# **[조건]**
# - gpt-4.1-nano로 먼저 시도 -> 텍스트 결과가 너무 짧거나(기준 : 기본 20자 미만), 공백/잡문자 위주일 경우 gpt-4.1 mini로 재시도
# - 위 결과도 안 좋을 경우 -> gpt-4.1 -> gpt-4o순으로 최대 2단계로 승급하여 실행
# - **모든 임계값/예산/모델은 환경변수로 쉽게 조절 가능**

# In[ ]:


import os, io, json, glob, base64, time, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image
from tqdm import tqdm
from openai import OpenAI


# In[ ]:


# ===== 경로 =====
DATA = Path("/home/spai0308/data")
ASSETS_DIR = DATA / "interim" / "hwpx_assets"
OUT_JSONL = DATA / "interim" / "assets_hwpx_ocr.jsonl"


# ===== 시간 =====
KST = timezone(timedelta(hours=9))


# ======= OPENAI 클라이언트 ======
client = OpenAI()

# ======== 모델 우선순위, 매개변수 =========== 
# env로 편하게 조정가능함
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY", "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4.1"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4o"),
]
MAX_ESCALATION_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "2")) 
# nano→mini→4.1(또는 4o) 까지 최대 2회
DOC_UPGRADE_BUDGET = int(os.getenv("OPENAI_OCR_DOC_UPGRADE_BUDGET", "5"))     
# 각 doc_id별 승급 허용 이미지 수
MAX_REQ_PER_MIN = int(os.getenv("OPENAI_OCR_RPM", "30"))                    
# 간단한 RPM 제한
MIN_CHARS = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))                      
# 품질 판정: 최소 글자수
MIN_ALPHA_RATIO = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))      
# 영문/한글 등 문자 비율 임계
HANGUL_BONUS = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))          
# 한글이 포함되면 점수 가산



VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}



def compress_image(pil_img: Image.Image, max_side=1280, fmt="JPEG", quality=85) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w,h))
        img = img.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality = quality, optimize=True)
    return buf.getvalue()

def to_data_url(image_bytes : bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64, {base64.b64encode(image_bytes).decode('utf-8')}"

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
    hangul = sum(0xAC00 <= ord(ch) <0xD7A3 for ch in t)
    ratio = alpha / total if total else 0
    ratio += HANGUL_BONUS if hangul > 0 else 0.0
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())
    
def now_kst_iso():
    return datetime.now(KST).isoformat()

def ocr_with_escalation(pil_img: Image.Image, doc_id: str, doc_budget_used: dict):
    """
    모델 사다리(MODEL_LADDER)를 따라가며 품질이 나쁘면 승급.
    각 doc_id별 승급 횟수에 예산(DOC_UPGRADE BUDGET) 적용
    """
    steps = 0 # 'steps' 변수 초기화
    used_model = MODEL_LADDER[0] # 'MODEL_LADDERP' 오타 수정
    text = call_vision(used_model, pil_img)

    # 승급 허용량 확인
    while text_quality_is_poor(text) and steps < MAX_ESCALATION_STEPS:
        # 문서 승급 예산 넘으면 중단
        if doc_budget_used.get(doc_id, 0) >= DOC_UPGRADE_BUDGET:
            break
        steps += 1
        next_idx = min(steps, len(MODEL_LADDER)-1)
        used_model = MODEL_LADDER[next_idx]
        text = call_vision(used_model, pil_img)
        doc_budget_used[doc_id] = doc_budget_used.get(doc_id, 0) + 1

    return text, used_model, steps

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rpm_win = []
    doc_upgrade_used = {} # doc_id -> 자동 사용 횟수
    total = 0

    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for doc_dir in sorted(ASSETS_DIR.glob("*")):
            if not doc_dir.is_dir():
                continue
            doc_id = doc_dir.name
            # 'VALIED_EXT' 오타 수정
            img_files = sorted([p for p in doc_dir.iterdir() if p.suffix.lower() in VALID_EXT]) 
            for idx, img_path in enumerate(tqdm(img_files, desc=f"OCR {doc_id}")):
                try:
                    pil = Image.open(str(img_path)).convert("RGB")
                    ratelimit(rpm_win)
                    text, used_model, steps = ocr_with_escalation(pil, doc_id, doc_upgrade_used)
                    rec = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "frame_index": 0,
                        "text": text,
                        "avg_conf": -1.0,
                        "lang": "ko+en",
                        "preprocess": {"resize_max_side":1280, "format":"jpeg", "quality":85},
                        "ts": now_kst_iso(),
                        "source_type": f"asset_ocr_hwpx_image", # 'kind' 변수 대신 고정 문자열 사용
                        "provider": "openai",
                        "model": used_model,
                        "escalation_steps": steps
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                except Exception as e:
                    err = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "error": repr(e),
                        "ts": now_kst_iso(),
                        "source_type": "asset_ocr_hwpx_error",
                        "provider": "openai",
                    }
                    out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(f"[OK] OCR complete -> {OUT_JSONL} (total {total} images)")

if __name__ == "__main__":
    main()


# In[ ]:


# !find /home/spai0308/data/interim/hwpx_assets -type f | wc -l
# !head -n 3 /home/spai0308/data/interim/hwpx_image_manifest.jsonl | jq .
# ls -l /home/spai0308/data/interim/hwpx_assets/* | head -n 20

# ## hwpx에서 따로 이미지 추출 안해서 이미지 없음 

# ocr 결과 확인
get_ipython().system('wc -l /home/spai0308/data/interim/assets_hwpx_ocr.jsonl')
print("="*50)
get_ipython().system('head -n 14 /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | jq .')


# In[ ]:


# 철도공사에서 ocr 시간이 오래 결려서 원인 탐구


# In[ ]:


# ocr이 실제로 몇 장을 어떤 모델로 처리했는지
get_ipython().system("jq -r '.model' /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | sort | uniq -c")


# In[ ]:


# 20자 미만의 짧은 결과가 얼마나 되는지
get_ipython().system('jq -r \'.text\' /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | awk \'{print length}\' | awk \'{s+=$1; if($1<20)c++} END{print "avg_len:", s/NR, "short(<20):", c "/" NR}\'')


# In[ ]:


# 어떤 모델을 썼었는지
get_ipython().system('jq -r \'select(.doc_id=="한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역") | .model\'    /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | sort | uniq -c')
print("="*50)

# 승급 단계 분포 확인: 0 = 승급 없음, 1~=승급 시도
get_ipython().system('jq -r \'select(.doc_id=="한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역") | .escalation_steps\'    /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | sort | uniq -c')
print("="*50)

# 오류 건수 확인
get_ipython().system('grep -c \'"source_type":"asset_ocr_hwpx_error"\' /home/spai0308/data/interim/assets_hwpx_ocr.jsonl')
print("="*50)

# 그 문서의 확장자 구성: 큰 bmp가 많은지 확인
get_ipython().system('ls /home/spai0308/data/interim/hwpx_assets/"한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역"   | awk -F. \'{print $NF}\' | sort | uniq -c')


# **[결과 해석]**
# 
# - **모델 사용**: nano 21장 / mini 1장 / 4.1 2장 → 대부분 nano로 처리됨
# - **승급**: 0단계 21건, 1단계 1건, 2단계 2건 → 승급 자체는 **소수**
# - **오류**: 0 → 실패 후 재시도로 시간 낭비된 흔적도 없음
# 
# 즉, 느렸던 건 주로 **모델 응답 지연/일시적 네트워크 지연** 혹은 **레이트리밋 백오프** 영향일 확률이 큼. (승급 2건이 특히 오래 걸렸을 수는 있지만, 24장 전체 16분을 좌우할 정도로 BMP만으로는 설명이 안 됨)

# #### 1-3-2) ocr "run_hwpx_assets_openai"로 모델 바꿔서 재시도

# - “패치 버전(``run_hwpx_assets_ocr_openai.py``)”로 실행
#     - 이미 공유한 패치엔 다음이 포함되어 있어 중복/멈춤을 예방함
#     - ``data``: URL 공백 제거
#     - ``timeout`` + 재시도(지수 백오프)
#     - ``resume-safe``(이미 처리된 ``source_path`` 스킵)
#     - 진행 중 파일명 로깅, 즉시 ``flush``
#     - ``source_type``를 ``asset_ocr_hwpx``로 통일
#     - 승급(에스컬레이션) 완전 끄기 : ``MAX_ESCALATION_STEPS=2`` -> ``0``으로 변경
#     - 1차 패스, ``nano``/``mini``만 돌리기 : 모델 교체

# In[ ]:


# -*- coding: utf-8 -*-
import os, io, json, base64, time, random
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError

# ===== 경로 =====
DATA = Path("/home/spai0308/data")
ASSETS_DIR = DATA / "interim" / "hwpx_assets"
OUT_JSONL = DATA / "interim" / "assets_hwpx_ocr.jsonl"

# ===== 시간 =====
KST = timezone(timedelta(hours=9))
def now_kst_iso() -> str:
    return datetime.now(KST).isoformat()

# ===== OpenAI =====
client = OpenAI()

# ===== 모델/정책 =====
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY",   "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4.1-mini"),
]
MAX_ESCALATION_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "2"))
DOC_UPGRADE_BUDGET   = int(os.getenv("OPENAI_OCR_DOC_UPGRADE_BUDGET", "5"))
MAX_REQ_PER_MIN      = int(os.getenv("OPENAI_OCR_RPM", "30"))

MIN_CHARS            = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))
MIN_ALPHA_RATIO      = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))
HANGUL_BONUS         = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))

VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

# ===== 유틸 =====
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
    # 쉼표 뒤 공백 제거 (중요)
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def text_quality_is_poor(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CHARS:
        return True
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    # 상한 포함(<=)로 수정
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = alpha / total if total else 0.0
    ratio += HANGUL_BONUS if hangul > 0 else 0.0
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())

# 안전한 OpenAI 호출 (타임아웃 + 지수백오프 재시도)
def call_vision(model: str, pil_img: Image.Image, timeout=60, max_retries=4) -> str:
    img_bytes = compress_image(pil_img, max_side=1280, fmt="JPEG", quality=85)
    data_url = to_data_url(img_bytes, "image/jpeg")
    for attempt in range(max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
                    {"role":"user","content":[
                        {"type":"text","text":"Extract all text from the image. If it's a table, read left-to-right, top-to-bottom. Return text only."},
                        {"type":"image_url","image_url":{"url":data_url}}
                    ]}
                ],
                temperature=0.0,
                timeout=timeout,  # ⛏️ 타임아웃
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError as e:
            sleep = min(2**attempt + random.uniform(0,1), 20)
            tqdm.write(f"[RateLimit] retry in {sleep:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep)
        except APIError as e:
            sleep = min(2**attempt + random.uniform(0,1), 20)
            tqdm.write(f"[APIError] {e}. retry in {sleep:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep)
        except Exception as e:
            if attempt >= max_retries:
                raise
            sleep = min(2**attempt + random.uniform(0,1), 10)
            tqdm.write(f"[Error] {e}. retry in {sleep:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep)
    return ""

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

# 이미 처리한 이미지 스킵(Resume-safe)
def load_processed(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                sp = j.get("source_path")
                # 성공/실패 상관없이 재시도 원하면 여기 조건 조절
                if sp and j.get("source_type","").startswith("asset_ocr_hwpx"):
                    done.add(sp)
    return done

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rpm_win = []
    doc_upgrade_used = {}   # 문서별 승급 사용량
    total = 0

    processed = load_processed(OUT_JSONL)
    tqdm.write(f"[resume] skip {len(processed)} already-processed images")

    with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
        for doc_dir in sorted(ASSETS_DIR.glob("*")):
            if not doc_dir.is_dir():
                continue
            doc_id = doc_dir.name
            img_files = sorted([p for p in doc_dir.iterdir() if p.suffix.lower() in VALID_EXT])

            for idx, img_path in enumerate(tqdm(img_files, desc=f"OCR {doc_id}", leave=True)):
                if str(img_path) in processed:
                    continue

                try:
                    # 진행 중 어떤 파일인지 보이게
                    tqdm.write(f"→ {doc_id} / {img_path.name}")

                    pil = Image.open(str(img_path)).convert("RGB")
                    ratelimit(rpm_win)
                    text, used_model, steps = ocr_with_escalation(pil, doc_id, doc_upgrade_used)

                    rec = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "frame_index": 0,
                        "text": text,
                        "avg_conf": -1.0,
                        "lang": "ko+en",
                        "preprocess": {"resize_max_side":1280,"format":"jpeg","quality":85},
                        "ts": now_kst_iso(),
                        "source_type": "asset_ocr_hwpx",
                        "provider": "openai",
                        "model": used_model,
                        "escalation_steps": steps
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total += 1

                except Exception as e:
                    err = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "error": repr(e),
                        "ts": now_kst_iso(),
                        "source_type": "asset_ocr_hwpx_error",
                        "provider": "openai",
                    }
                    out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    out_f.flush()

    print(f"[OK] OCR complete -> {OUT_JSONL} (total {total} images)")

if __name__ == "__main__":
    main()


# In[ ]:


# 전수 처리 후 짧은 결과만 재시도:
get_ipython().system('jq -r \'select(.source_type=="asset_ocr_hwpx" and (.text|length)<20) | .source_path\'    /home/spai0308/data/interim/assets_hwpx_ocr.jsonl | sort -u > /home/spai0308/data/interim/retry_list.txt')


# In[ ]:


# -*- coding: utf-8 -*-
# 부실컷 목록(retry_list.txt)만 읽어 상위 모델로 재시도해 별도 JSONL에 기록
import os, io, json, base64, time, random
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

DATA = Path("/home/spai0308/data")
RETRY_LIST = DATA / "interim" / "retry_list.txt"
OUT_JSONL  = DATA / "interim" / "assets_hwpx_ocr_retry.jsonl"

KST = timezone(timedelta(hours=9))
def now_ts(): return datetime.now(KST).isoformat()

# 재시도는 mini→4.1→4o(필요 시)로 조금 더 세게
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY",   "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4o"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4o"),
]
MAX_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "2"))
RPM = int(os.getenv("OPENAI_OCR_RPM", "20"))
MIN_CHARS = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))
MIN_ALPHA_RATIO = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))
HANGUL_BONUS = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))

client = OpenAI()

def compress(pil, max_side=1024, q=75):
    import io
    img = pil.convert("RGB")
    w,h = img.size
    if max(w,h) > max_side:
        s = max_side/float(max(w,h))
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q, optimize=True)
    return buf.getvalue()

def data_url(b): return "data:image/jpeg;base64," + base64.b64encode(b).decode()

def call(model, pil, timeout=45):
    b = compress(pil)
    url = data_url(b)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
            {"role":"user","content":[
                {"type":"text","text":"Extract all text from the image. If it's a table, read left-to-right, top-to-bottom. Return text only."},
                {"type":"image_url","image_url":{"url":url}}
            ]}
        ],
        temperature=0.0,
        timeout=timeout
    )
    return (r.choices[0].message.content or "").strip()

def poor(text:str)->bool:
    t=(text or "").strip()
    if len(t) < MIN_CHARS: return True
    total=len(t); alpha=sum(ch.isalpha() for ch in t); hangul=sum(0xAC00<=ord(ch)<=0xD7A3 for ch in t)
    ratio = (alpha/total if total else 0.0) + (HANGUL_BONUS if hangul>0 else 0.0)
    return ratio < MIN_ALPHA_RATIO

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRY_LIST, "r", encoding="utf-8") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]
    print(f"[retry] {len(paths)} images")

    rpm=[]
    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for p in tqdm(paths, desc="Retry OCR"):
            try:
                doc_id = p.parent.name
                pil = Image.open(str(p)).convert("RGB")
                # nano는 건너뛰고 mini부터
                used = MODEL_LADDER[0]; text = call(used, pil); steps=0
                while poor(text) and steps < MAX_STEPS:
                    steps += 1
                    used = MODEL_LADDER[min(steps, len(MODEL_LADDER)-1)]
                    text = call(used, pil)
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(p),
                    "frame_index": 0,
                    "text": text,
                    "avg_conf": -1.0,
                    "lang": "ko+en",
                    "preprocess": {"resize_max_side":1024,"format":"jpeg","quality":75},
                    "ts": now_ts(),
                    "source_type": "asset_ocr_hwpx_retry",
                    "provider": "openai",
                    "model": used,
                    "escalation_steps": steps
                }
                out.write(json.dumps(rec, ensure_ascii=False)+"\n"); out.flush()
                # 간단 RPM 제어
                now=time.time(); rpm[:]=[t for t in rpm if now-t<60]; rpm.append(now)
                if len(rpm) >= RPM: time.sleep(60-(now-rpm[0]))
            except Exception as e:
                out.write(json.dumps({
                    "doc_id": p.parent.name, "source_path": str(p),
                    "error": repr(e), "ts": now_ts(),
                    "source_type":"asset_ocr_hwpx_retry_error","provider":"openai"
                }, ensure_ascii=False)+"\n"); out.flush()

if __name__ == "__main__":
    main()


# #### 1-4) hwpx 텍스트 x ocr 머지 스크립트 (안전버전 : 중간파일 -> 전체병합)

# [구조]
# 
# - [문서 본문(``doc_text``) 1블록 + 이미지별 ocr 블록 + ocr 합본 블록] 으로 생성
# - 페이지 번호/ 앵커가 없어도 안전하게 사용 가능
# - 나중에 청킹 단계에서 type별로 가중치 줄 수 있음
# - ``pages_all_merged.jsonl``에 합칠 땐, 최종에 append만 하면 됨

# [품질 포인트]
# 
# 
# 1. **중복/짧은 OCR 제거**
# - `MIN_CHARS=12` + 영/한/숫자 비율 필터로 노이즈 컷
# 2. **스키마 일치**
# - 당신의 OCR 출력 필드(`text`, `source_path`, `model`, `escalation_steps`, `avg_conf`) 그대로 유지해 `meta`에 보관
# 3. **검색 친화 블록 타입**
# - `type: doc_text / ocr_image / ocr_concat` → 청킹/리랭킹에서 가중치/우선순위 조절하기 쉬움
# 4. **안전한 병합 플로우**
# - 먼저 `pages_hwpx_merged.jsonl`로 만들고 → 검수 → `pages_all_merged.jsonl`에 appendㄴ
# 
# 

# In[ ]:


# -*- coding: utf-8 -*-
import json, re, hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

HOME = Path.home()
INTERIM = HOME / "data" / "interim"

IN_TEXT   = INTERIM / "pages_hwpx.jsonl"              # HWPX 전체 텍스트 (문서당 1레코드)
IN_OCR    = INTERIM / "assets_hwpx_ocr.jsonl"         # HWPX 이미지 OCR 결과 (이미지당 1레코드)
OUT_MERGE = INTERIM / "pages_hwpx_merged.jsonl"       # HWPX 전용 병합 산출물
ALL_MERGED= INTERIM / "pages_all_merged.jsonl"        # 전체 통합 파일(옵션 append)

KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat()
# └ 한국시간 타임스탬프 생성

def load_jsonl(path: Path):
    if not path.exists(): return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
# └ JSONL 로더(메모리 적재형). 문서 수가 아주 크면 제너레이터로 바꿔도 됨

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()
# └ 공백 정규화(중복 판단/길이 필터에 필요)

def md5(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()
# └ 텍스트 해시(중복 제거용)

# ---- 1) 입력 로드 ----
text_recs = load_jsonl(IN_TEXT)     # [{doc_id, text, ...}]
ocr_recs  = load_jsonl(IN_OCR)      # [{doc_id, text, source_path, model, ...}]

# ---- 2) 인메모리 맵 구성 ----
doc_text = {}                       # doc_id -> 본문 텍스트(1건)
ocr_by_doc = defaultdict(list)      # doc_id -> OCR 리스트(여러건)

for r in text_recs:
    if not r.get("doc_id"): continue
    doc_text[r["doc_id"]] = r

for r in ocr_recs:
    if not r.get("doc_id"): continue
    if r.get("source_type","").startswith("asset_ocr_hwpx"):
        ocr_by_doc[r["doc_id"]].append(r)

# ---- 3) OCR 전처리: 너무 짧거나(노이즈) 중복 텍스트 제거 ----
MIN_CHARS = 12                      # 이보다 짧으면 스킵(개행/잡음 방지)
seen_hash = set()

def is_valid_ocr_text(t: str) -> bool:
    t = (t or "").strip()
    if len(t) < MIN_CHARS: return False
    # 영문/한글/숫자 비율이 너무 낮으면 제거(선택)
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(len(t),1) < 0.15: return False
    return True

def uniq_filter(texts):
    out=[]
    for t in texts:
        h = md5(normalize(t))
        if h in seen_hash: 
            continue
        seen_hash.add(h)
        out.append(t)
    return out

# ---- 4) 병합 레코드 생성 ----
OUT_MERGE.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_MERGE, "w", encoding="utf-8") as wf:
    for doc_id, base in doc_text.items():
        # 4-1) 문서 본문(원문 텍스트)
        base_text = normalize(base.get("text",""))
        wf.write(json.dumps({
            "source": "hwpx",
            "doc_id": doc_id,
            "block_index": 0,                     # 첫 블록
            "type": "doc_text",                   # 본문 전체
            "text": base_text,
            "meta": {
                "source_path": base.get("source_path"),
                "ts_merged": now_kst_iso(),
                "source_type": "hwpx_text"
            }
        }, ensure_ascii=False) + "\n")

        # 4-2) OCR(이미지별) 블록
        idx = 1
        ocr_list = ocr_by_doc.get(doc_id, [])
        ocr_texts = []
        for o in ocr_list:
            t = o.get("text","")
            if not is_valid_ocr_text(t): 
                continue
            t_norm = normalize(t)
            # 개별 이미지 OCR 블록
            wf.write(json.dumps({
                "source": "hwpx",
                "doc_id": doc_id,
                "block_index": idx,
                "type": "ocr_image",
                "text": t_norm,
                "meta": {
                    "image_path": o.get("source_path"),
                    "model": o.get("model"),
                    "escalation_steps": o.get("escalation_steps", 0),
                    "avg_conf": o.get("avg_conf", -1.0),
                    "ts_merged": now_kst_iso(),
                    "source_type": o.get("source_type")
                }
            }, ensure_ascii=False) + "\n")
            idx += 1
            ocr_texts.append(t_norm)

        # 4-3) OCR 전체 합본(검색/후처리 편의용)
        ocr_texts = uniq_filter(ocr_texts)        # 중복 제거
        if ocr_texts:
            wf.write(json.dumps({
                "source": "hwpx",
                "doc_id": doc_id,
                "block_index": idx,
                "type": "ocr_concat",
                "text": "\n\n".join(ocr_texts)[:200000],  # 너무 길면 컷
                "meta": {
                    "image_count": len(ocr_list),
                    "ocr_kept": len(ocr_texts),
                    "ts_merged": now_kst_iso(),
                    "note": "문서 내 모든 이미지 OCR 합본(중복/노이즈 제거 후)"
                }
            }, ensure_ascii=False) + "\n")

print(f"[OK] HWPX merged → {OUT_MERGE}")


# ##### **1-4-1) 품질점검**

# 1. **품질 점검(초간단 EDA)**(hwpx 병합파일 EDA)
#     - `pages_hwpx_merged.jsonl`의
#         - 문서 수(`doc_id` unique), 블록 타입 분포(`doc_text/ocr_image/ocr_concat`),
#         - 평균/최대 텍스트 길이, 빈(짧은) 블록 비율을 확인.
#     - 문서 수/ 타입 분포/ 길이  통계/ 짧은 비율

# In[ ]:


# -*- coding: utf-8 -*-
import json, re, math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

# ===== 경로 설정 =====
HOME = Path.home()                                   # 사용자 홈 디렉토리
# 설명: 실행 환경의 홈 디렉토리를 가져옵니다.

INTERIM = HOME / "data" / "interim"                  # 중간 산출물 폴더
# 설명: 프로젝트에서 사용 중인 공통 중간 경로입니다.

IN_MERGED = INTERIM / "pages_hwpx_merged.jsonl"      # HWPX 병합 결과(JSONL)
# 설명: 방금 만들었던 HWPX 전용 병합 파일 경로입니다.


# ===== 유틸 =====
def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
# 설명: JSONL 파일을 한 줄씩 제너레이터로 읽어 메모리를 아낍니다.

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()
# 설명: 공백/개행을 하나의 공백으로 정리하여 길이·중복 계산의 노이즈를 줄입니다.


# ===== 파라미터(조절 가능) =====
SHORT_LEN = 30                                       # "짧은 블록"으로 간주할 기준(문자 수)
# 설명: 30자 미만은 소제목/캡션/노이즈일 가능성이 높아 비율을 따로 봅니다.

LENGTH_BINS = [0, 50, 100, 200, 400, 800, 1600, 3200, 10000]
# 설명: 길이 분포 히스토그램 구간(문자 수 기준)입니다.


# ===== 코어 로직 =====
def run_eda():
    if not IN_MERGED.exists():
        raise FileNotFoundError(f"파일이 없습니다: {IN_MERGED}")

    doc_ids = set()                                   # 문서 고유 개수 집계
    # 설명: 문서 수(고유 doc_id)를 세기 위한 집합입니다.

    type_counter = Counter()                          # 블록 타입 분포
    # 설명: doc_text / ocr_image / ocr_concat 개수를 셉니다.

    lengths = []                                      # 전체 텍스트 길이 리스트
    # 설명: 전체 블록의 텍스트 길이를 모아 통계(평균/최대/중앙/백분위)를 계산합니다.

    lengths_by_type = defaultdict(list)               # 타입별 텍스트 길이
    # 설명: 타입별로도 길이 통계를 보고 싶기 때문에 따로 모읍니다.

    short_count = 0                                   # 짧은 블록(문자 수 < SHORT_LEN) 개수
    # 설명: 짧은 블록 비율을 계산합니다.

    total = 0                                         # 총 레코드 수
    # 설명: 파일 전체 레코드(블록) 수를 셉니다.

    ocr_per_doc = Counter()                           # 문서별 OCR(ocr_image) 블록 수
    # 설명: 어떤 문서가 이미지(OCR)가 많은지 확인할 때 씁니다.

    # 길이 히스토그램(전역)
    hist_bins = [0]*(len(LENGTH_BINS)-1)
    # 설명: 구간 개수만큼 0으로 초기화합니다.

    # 파일 순회
    for rec in load_jsonl(IN_MERGED):
        total += 1

        doc_id = rec.get("doc_id")
        if doc_id:
            doc_ids.add(doc_id)

        btype = rec.get("type", "?")
        type_counter[btype] += 1

        text = normalize(rec.get("text", ""))
        L = len(text)
        lengths.append(L)
        lengths_by_type[btype].append(L)

        if L < SHORT_LEN:
            short_count += 1

        # 히스토그램 집계
        for i in range(len(LENGTH_BINS)-1):
            if LENGTH_BINS[i] <= L < LENGTH_BINS[i+1]:
                hist_bins[i] += 1
                break

        # 문서별 OCR 카운트
        if btype == "ocr_image" and doc_id:
            ocr_per_doc[doc_id] += 1

    # 안전 가드: 데이터가 하나도 없을 때
    if total == 0:
        print("[EDA] 파일에 레코드가 없습니다.")
        return

    # ===== 전체 통계 =====
    def pct(p):
        # p백분위수(간이): 정렬 후 인덱스로 근사
        arr = sorted(lengths)
        idx = min(max(int(round(p/100 * (len(arr)-1))), 0), len(arr)-1)
        return arr[idx]

    print("===== HWPX EDA (pages_hwpx_merged.jsonl) =====")
    # 설명: 보고서 헤더를 출력합니다.

    print(f"- 총 블록 수: {total:,}")
    # 설명: 병합 파일 내 전체 레코드(블록) 개수입니다.

    print(f"- 문서 수(doc_id unique): {len(doc_ids):,}")
    # 설명: 고유 문서 개수입니다.

    print(f"- 블록 타입 분포: {dict(type_counter)}")
    # 설명: {'doc_text': x, 'ocr_image': y, 'ocr_concat': z} 형식의 분포입니다.

    print(f"- 전체 텍스트 길이 (문자수) 통계")
    # 설명: 전체 블록의 텍스트 길이에 대한 요약 통계입니다.
    print(f"  · 평균: {mean(lengths):.1f}")
    print(f"  · 중앙값: {median(lengths):.1f}")
    print(f"  · p95: {pct(95)}")
    print(f"  · 최대: {max(lengths)}")

    short_ratio = short_count / total
    print(f"- 짧은 블록(<{SHORT_LEN}자) 비율: {short_ratio:.2%}  ({short_count}/{total})")
    # 설명: 제목/캡션/노이즈 비중을 가늠하는 지표입니다. 너무 높으면 청킹/클리닝 재검토 필요.

    # ===== 타입별 통계 =====
    print("- 타입별 길이 통계")
    # 설명: 타입(doc_text/ocr_image/ocr_concat) 각각의 길이 분포 요약입니다.
    for t, arr in lengths_by_type.items():
        if not arr: 
            continue
        arr_sorted = sorted(arr)
        def tpct(p):
            idx = min(max(int(round(p/100 * (len(arr_sorted)-1))), 0), len(arr_sorted)-1)
            return arr_sorted[idx]
        print(f"  [{t}] n={len(arr)}  mean={mean(arr):.1f}  median={median(arr):.1f}  p95={tpct(95)}  max={max(arr)}")

    # ===== 길이 히스토그램 =====
    print("- 길이 분포(히스토그램, 문자수 구간)")
    # 설명: 길이대별 블록 개수를 출력합니다.
    for i in range(len(LENGTH_BINS)-1):
        lo, hi = LENGTH_BINS[i], LENGTH_BINS[i+1]
        print(f"  {lo:>4} ~ {hi:<4}: {hist_bins[i]:>6}")

    # ===== OCR 많은 문서 Top-N =====
    TOP_N = 10
    print(f"- OCR(ocr_image) 블록이 많은 문서 TOP{TOP_N}")
    # 설명: 이미지 기반 정보가 많은 문서를 빠르게 파악할 수 있습니다.
    for doc, cnt in ocr_per_doc.most_common(TOP_N):
        print(f"  {doc}: {cnt}")

    print("===============================================")
    # 설명: 보고서 끝 표시입니다.


if __name__ == "__main__":
    run_eda()
# 설명: 스크립트를 직접 실행했을 때만 EDA를 수행합니다.


# **[종합 해석]**
# 
# 1. **문서 구조는 정상**: 문서당 `doc_text + 여러 OCR + OCR 합본` 구조가 잘 잡혔음.
# 2. **본문(doc_text)과 OCR 합본은 길이가 너무 길어 → 청킹 필요**.
# 3. **OCR 개별 블록은 편차가 심함**:
#     - 짧은 건 그냥 쓰면 되고,
#     - 긴 건 청킹 필요 (특히 표 OCR).
# 4. **철도공사 문서는 OCR 의존도가 높음** → “표/그림 질의”에서 OCR 가중치를 반드시 주는 게 중요.
# 5. **짧은 블록 노이즈는 적음** → 클리닝 부담은 크지 않음.

# # 청킹

# In[ ]:


# import json, re, math
# from pathlib import Path
# from collections import defaultdict
# from datetime import datetime, timezone, timedelta

# # ====== 경로 ======
# HOME = Path.home()                                                 # 사용자 홈 디렉터리
# # (주석) Path.home()는 현재 사용자 홈 경로를 돌려줍니다.

# INTERIM = HOME / "data" / "interim"                                # 중간 산출물 폴더
# # (주석) 프로젝트에서 사용 중인 공통 중간 산출물 경로입니다.

# IN_MERGED = INTERIM / "pages_hwpx_merged.jsonl"                    # hwpx 병합 결과 (입력)
# # (주석) 앞 단계에서 만든 HWPX 전용 병합 파일입니다.

# OUT_CHUNKS = INTERIM / "chunks_hwpx_800_200.jsonl"                 # 청킹 결과 (출력)
# # (주석) 800/200 규칙으로 만든 청크를 저장할 파일입니다.

# # ====== 청킹 파라미터 ======
# WIN = 800                                                          # 윈도우 크기(문자 수)
# # (주석) 한 청크의 최대 길이(문자 기준)입니다.

# STRIDE = 200                                                       # 오버랩 크기(문자 수)
# # (주석) 슬라이딩 윈도우로 겹치는 길이입니다. 경계 손실을 줄입니다.

# # ====== 로더 ======
# def load_jsonl(path: Path):
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line=line.strip()
#             if line:
#                 yield json.loads(line)
# # (주석) JSONL을 제너레이터로 읽어 메모리를 절약합니다.

# def normalize(s: str) -> str:
#     return re.sub(r"\s+", " ", (s or "")).strip()
# # (주석) 공백/개행을 일관화해 임베딩 품질과 중복 필터에 도움을 줍니다.

# # ====== 간단 EDA(선택) ======
# def quick_eda():
#     cnt = 0
#     types = defaultdict(int)
#     for r in load_jsonl(IN_MERGED):
#         cnt += 1
#         types[r.get("type","?")] += 1
#     print("[EDA] records:", cnt, "types:", dict(types))
# # (주석) 블록 타입 분포와 총 레코드 수를 빠르게 확인합니다.

# # ====== 청킹 ======
# def chunk_text(text: str, win=WIN, stride=STRIDE):
#     t = normalize(text)
#     if not t:
#         return []

#     chunks = []
#     start = 0
#     n = len(t)
#     while start < n:
#         end = min(start + win, n)
#         chunks.append(t[start:end])
#         if end == n:
#             break
#         start = end - stride
#     return chunks
# # (주석) 문자 기반 슬라이딩 윈도우 청킹 함수입니다.
# # (주석) 마지막 청크는 남은 길이만큼 생성됩니다.

# def run_chunk():
#     OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)

#     with open(OUT_CHUNKS, "w", encoding="utf-8") as wf:
#         for r in load_jsonl(IN_MERGED):
#             doc_id = r["doc_id"]
#             btype  = r.get("type")              # doc_text / ocr_image / ocr_concat
#             text   = r.get("text","")

#             # 타입별 윈도우/스트라이드 미세 조정(옵션)
#             win, stride = WIN, STRIDE
#             if btype == "ocr_image":
#                 win, stride = 600, 150          # OCR은 노이즈 가능성 → 조금 더 작은 윈도우 권장
#             elif btype == "ocr_concat":
#                 win, stride = 700, 200          # 합본은 중복 가능 → 다소 보수적으로

#             for i, ch in enumerate(chunk_text(text, win, stride)):
#                 out = {
#                     "source": "hwpx",
#                     "doc_id": doc_id,
#                     "block_type": btype,
#                     "parent_block_index": r.get("block_index", -1),
#                     "chunk_index": i,
#                     "text": ch,
#                     "meta": r.get("meta", {})
#                 }
#                 wf.write(json.dumps(out, ensure_ascii=False) + "\n")
#     print(f"[OK] chunked → {OUT_CHUNKS}")
# # (주석) 타입별로 윈도우를 미묘하게 달리해 안정적인 품질을 노립니다.

# if __name__ == "__main__":
#     quick_eda()            # (선택) 분포 확인
#     run_chunk()            # 청킹 실행


# In[ ]:


# -*- coding: utf-8 -*-
import json, re
from pathlib import Path

# ===== 경로 =====
HOME = Path.home()
INTERIM = HOME / "data" / "interim"

IN_MERGED = INTERIM / "pages_hwpx_merged.jsonl"          # 병합 입력
OUT_CHUNKS = INTERIM / "chunks_hwpx_800_200.jsonl"       # 청킹 출력

# ===== 청킹 파라미터 =====
WIN = 800         # 한 청크 최대 길이
STRIDE = 200      # 겹치는 길이

def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_text(text: str, win=WIN, stride=STRIDE):
    t = normalize(text)
    if not t:
        return []
    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + win, n)
        chunks.append(t[start:end])
        if end == n:
            break
        start = end - stride
    return chunks

def run_chunk():
    OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CHUNKS, "w", encoding="utf-8") as wf:
        for r in load_jsonl(IN_MERGED):
            doc_id = r["doc_id"]
            btype  = r.get("type")             # doc_text / ocr_image / ocr_concat
            text   = r.get("text","")

            # OCR은 조금 더 보수적으로 (짧으면 그대로, 길면 쪼갬)
            win, stride = WIN, STRIDE
            if btype == "ocr_image":
                win, stride = 600, 150
            elif btype == "ocr_concat":
                win, stride = 700, 200

            chunks = chunk_text(text, win, stride)
            for i, ch in enumerate(chunks):
                out = {
                    "source": "hwpx",
                    "doc_id": doc_id,
                    "block_type": btype,
                    "parent_block_index": r.get("block_index", -1),
                    "chunk_index": i,
                    "text": ch,
                    "meta": r.get("meta", {})
                }
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] chunked → {OUT_CHUNKS}")

if __name__ == "__main__":
    run_chunk()


# # 임베딩 & 인덱스 구축

# - OpenAI 임베딩(`text-embedding-3-small`)을 써서 FAISS 인덱스 생성
# - 결과물: `faiss_index_hwpx.bin` + `chunks_hwpx_800_200.jsonl` (메타데이터 연결)

# In[ ]:


# 터미널 : faiss-cpu 설치
# pip install faiss-cpu numpy tqdm


# In[ ]:


# -*- coding: utf-8 -*-
import os, json, math
# 설명: 표준 라이브러리 모듈 임포트

from pathlib import Path
# 설명: 경로 처리를 위한 Path 객체 사용

import numpy as np
# 설명: 벡터/배열 연산을 위한 numpy

from tqdm import tqdm
# 설명: 진행률 표시용

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss-cpu가 필요합니다. `pip install faiss-cpu` 후 다시 실행하세요.") from e
# 설명: 벡터 인덱싱 라이브러리 FAISS 임포트. 설치 안 되어 있으면 안내.

from openai import OpenAI
# 설명: OpenAI SDK (>=1.0) 사용

# ===== 경로/설정 =====
HOME = Path.home()
# 설명: 사용자 홈 경로

INTERIM = HOME / "data" / "interim"
# 설명: 프로젝트 중간 산출물 경로

IN_CHUNKS = INTERIM / "chunks_hwpx_800_200.jsonl"
# 설명: 앞 단계에서 만든 청킹 결과 파일

INDEX_BIN = INTERIM / "faiss_index_hwpx.bin"
# 설명: FAISS 인덱스 저장 경로

META_NPY  = INTERIM / "faiss_meta_hwpx.npy"
# 설명: 메타데이터(매핑용) 저장 경로 (numpy object 배열)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# 설명: 임베딩 모델 이름 (환경변수로 바꿀 수 있음)

client = OpenAI()
# 설명: OpenAI 클라이언트 생성 (환경변수 OPENAI_API_KEY 사용)

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    # 설명: 벡터의 L2 노름(길이)을 계산하고 0으로 나누는 것을 방지하기 위해 작은 값을 더함
    return v / norm
    # 설명: 코사인 유사도를 쓰기 위해 벡터를 단위 벡터로 정규화

def embed_texts(batch_texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
    # 설명: OpenAI 임베딩 API 호출 (배치 입력)
    vecs = [d.embedding for d in resp.data]
    # 설명: 응답에서 임베딩 벡터를 추출
    return np.array(vecs, dtype="float32")
    # 설명: numpy float32 배열로 변환 (FAISS 호환)

def build_faiss():
    if not IN_CHUNKS.exists():
        raise FileNotFoundError(f"청크 파일이 없습니다: {IN_CHUNKS}")
    # 설명: 입력 파일 존재 체크

    texts, metas = [], []
    # 설명: 임베딩 대상 텍스트와 메타데이터를 보관할 리스트

    with open(IN_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # 설명: 한 줄씩 JSON 파싱
            t = rec.get("text", "").strip()
            # 설명: 필수 필드 text
            if not t:
                continue
            texts.append(t)
            # 설명: 임베딩 대상 텍스트 수집
            metas.append({
                "doc_id": rec.get("doc_id"),
                "block_type": rec.get("block_type"),
                "parent_block_index": rec.get("parent_block_index"),
                "chunk_index": rec.get("chunk_index"),
                "source": rec.get("source"),
                "meta": rec.get("meta", {})
            })
            # 설명: 검색 결과 표시/후처리에 필요한 메타데이터 저장

    if not texts:
        raise RuntimeError("임베딩할 텍스트가 없습니다. 청킹 결과를 확인하세요.")
    # 설명: 방어 코드

    # ---- 배치 임베딩 ----
    B = 256
    # 설명: 배치 크기 (환경/쿼터에 맞춰 조절 가능)

    vec_list = []
    for i in tqdm(range(0, len(texts), B), desc="embedding"):
        batch = texts[i:i+B]
        # 설명: 배치 슬라이싱
        v = embed_texts(batch)
        # 설명: OpenAI 임베딩 호출
        vec_list.append(v)
        # 설명: 결과를 리스트에 누적

    X = np.vstack(vec_list).astype("float32")
    # 설명: 배치 결과를 수직 결합해 (N, D) 배열 생성

    X = normalize(X)
    # 설명: 코사인 유사도를 내적으로 쓰기 위해 정규화

    d = X.shape[1]
    # 설명: 임베딩 차원 수

    index = faiss.IndexFlatIP(d)
    # 설명: 내적(Inner Product) 기반 인덱스 생성 (정규화했으니 코사인과 동일)

    index.add(X)
    # 설명: 모든 벡터를 인덱스에 추가

    faiss.write_index(index, str(INDEX_BIN))
    # 설명: FAISS 인덱스를 파일로 저장

    # 메타데이터도 별도 저장 (numpy object array)
    np.save(META_NPY, np.array(metas, dtype=object), allow_pickle=True)
    # 설명: 인덱스와 같은 순서로 매핑된 메타 배열 저장

    print(f"[OK] FAISS index saved → {INDEX_BIN} (n={len(texts)}, dim={d})")
    # 설명: 완료 로그 출력
    print(f"[OK] Metadata saved   → {META_NPY}")
    # 설명: 메타 저장 로그 출력

if __name__ == "__main__":
    build_faiss()
    # 설명: 스크립트 직접 실행 시 인덱스 구축 수행


# In[ ]:


# 사용할 수 있는 임베딩 모델 확인

from openai import OpenAI; import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
for m in ["text-embedding-3-small","text-embedding-3-large"]:
    try:
        client.embeddings.create(model=m, input=["ping"])
        print("OK:", m); CHOSEN=m; break
    except Exception as e:
        print("FAIL:", m, e)


# In[ ]:


import os
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_PROJECT:", os.getenv("OPENAI_PROJECT"))        # 이게 잡혀 있으면 의심 ↑
print("OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE") or "default")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) 채팅은 되는가? (키 자체 생존 확인)
client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user","content":"ping"}])
print("chat ok")

# 어제와 오늘 사용한 키가 같은지
print(os.getenv("OPENAI_API_KEY")[:12])

# 2) 임베딩만 막히는가? (403 재현)
client.embeddings.create(model="text-embedding-3-small", input=["ping"])




# ## 검색함수 +  hwpx 타입별 가중치 리랭크

# In[ ]:


# 검색합수 & 리랭크 수정

# -*- coding: utf-8 -*-
import os, re, json
# 표준 모듈
import numpy as np
# 수치 연산용
from pathlib import Path
# 경로 관리
import faiss
# 검색용 인덱스 로딩
from openai import OpenAI
# 쿼리 임베딩 계산

HOME = Path.home()
# 홈 경로
INTERIM = HOME / "data" / "interim"
# 산출물 경로
INDEX_BIN = INTERIM / "faiss_index_hwpx.bin"
# 인덱스 파일 경로
META_NPY  = INTERIM / "faiss_meta_hwpx.npy"
# 메타 배열 경로
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# 임베딩 모델 이름

client = OpenAI()
# OpenAI 클라이언트

# ---- 타입별 가중치 (기본 prior) ----
TYPE_PRIOR = {
    "doc_text": 1.00,     # 본문은 기본값 1.0
    "ocr_image": 0.90,    # OCR 개별은 노이즈 가능성으로 소폭 감점
    "ocr_concat": 0.80    # 합본은 중복/주제 섞임 가능성으로 더 감점
}

# ---- 질의 키워드에 따른 보정 규칙 ----
OCR_KEYWORDS = ["그림", "도표", "표", "양식", "사진", "캡션", "Figure", "Table"]
# 표/그림 관련 질의 키워드 목록

def has_ocr_intent(q: str) -> bool:
    qn = q.lower()
    # 소문자 변환으로 일치율 향상
    if any(k.lower() in qn for k in OCR_KEYWORDS):
        return True
    # 키워드가 쿼리에 포함되면 True
    return False

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    # 쿼리 임베딩 계산
    v = np.array([resp.data[0].embedding], dtype="float32")
    # numpy float32 배열로 변환
    # 정규화 (코사인 내적)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    # 코사인 내적을 위해 단위 벡터화
    return v

def load_resources():
    if not INDEX_BIN.exists() or not META_NPY.exists():
        raise FileNotFoundError("인덱스/메타 파일이 없습니다. 먼저 인덱스를 빌드하세요.")
    # 리소스 존재 확인

    index = faiss.read_index(str(INDEX_BIN))
    # FAISS 인덱스 로드

    metas = np.load(META_NPY, allow_pickle=True)
    # 메타데이터 로드 (검색 결과 매핑용)

    return index, metas

def rerank(scores: np.ndarray, metas, query: str) -> np.ndarray:
    boosted = scores.copy()
    # 점수를 복사해 보정

    ocr_intent = has_ocr_intent(query)
    # 질의가 표/그림 의도인지 감지

    for i in range(len(metas)):
        mt = metas[i]
        t = (mt.get("block_type") or "").strip()
        # 블록 타입 추출

        prior = TYPE_PRIOR.get(t, 1.0)
        # 기본 타입별 가중치

        if ocr_intent:
            if t == "ocr_image":
                prior *= 1.15   # 표/그림 질의 시 OCR 이미지 상향
            elif t == "ocr_concat":
                prior *= 1.05   # 합본도 약간 상향
        # 질의 유형에 따른 보정

        boosted[i] = boosted[i] * prior + (0.01 if t == "doc_text" else 0.0)
        # 기본적으로 doc_text에 아주 작은 상수 가산으로 동률 시 본문 우선

    return boosted

def search(query: str, topk=10):
    index, metas = load_resources()
    # 인덱스와 메타 로드

    qv = embed_query(query)
    # 쿼리 임베딩

    sims, idxs = index.search(qv, topk)
    # 코사인 유사도(내적) 기반 상위 topk 검색

    sims = sims[0]
    idxs = idxs[0]
    # 배치 차원 제거

    # 1차 점수/메타 수집
    raw = []
    for score, ix in zip(sims, idxs):
        mt = metas[ix].item() if isinstance(metas[ix], np.ndarray) else metas[ix]
        raw.append((score, ix, mt))
    # (점수, 인덱스, 메타) 튜플로 수집

    # 2차 리랭크 (타입별 가중치/질의 의도 반영)
    scores = np.array([r[0] for r in raw], dtype="float32")
    # 원 점수 배열
    re_scores = rerank(scores, [r[2] for r in raw], query)
    # 보정 점수 계산
    order = np.argsort(-re_scores)
    # 내림차순 정렬 인덱스

    results = []
    for rank in order:
        score, ix, mt = raw[rank]
        results.append({
            "rank": len(results)+1,
            "score_raw": float(score),
            "score_re": float(re_scores[rank]),
            "index_id": int(ix),
            "doc_id": mt.get("doc_id"),
            "block_type": mt.get("block_type"),
            "chunk_index": mt.get("chunk_index"),
            "parent_block_index": mt.get("parent_block_index"),
            "source": mt.get("source"),
            "meta": mt.get("meta", {})
        })
    # 정렬된 결과를 딕셔너리로 구성

    return results
    # 최종 결과 반환



# --- (검색 함수 + 리랭크 코드 아래에 추가) ---
if __name__ == "__main__":                                   # ← 스크립트처럼 직접 실행할 때만 동작
    try:
        test_q = "사업 기간"                                  # 간단한 테스트 질의
        out = search(test_q, topk=3)                          # 상위 3개만 간단 확인
        for r in out:
            print(r["rank"], r["doc_id"], r["block_type"], f'{r["score_re"]:.3f}')
        print("[OK] search() self-test passed")
    except Exception as e:
        import traceback
        print("[ERR] search() self-test failed:", repr(e))    # 에러 메시지 확인
        traceback.print_exc()                                 # 스택트레이스까지 출력


# In[ ]:


# 자가 진단

res = search("사업 기간", topk=3)                      # 검색 함수 직접 호출
for r in res:
    print(r["rank"], r["doc_id"], r["block_type"], f'{r["score_re"]:.3f}')


# ## 스모크 테스트: 질의 셋 + Hit@3 / MRR@10 간이 평가

# In[ ]:


# -*- coding: utf-8 -*-
import json, re                                      # 표준 모듈
from pathlib import Path                              # 경로 처리
from math import inf                                  # 무한대 상수(정답 랭크 없을 때 사용)

HOME = Path.home()                                    # 홈 폴더
INTERIM = HOME / "data" / "interim"                   # 중간 산출물 폴더
CHUNKS_FILE = INTERIM / "chunks_hwpx_800_200.jsonl"   # 검색 결과 인덱스→텍스트 매핑용 원본 청크 파일

def load_chunks():
    rows=[]
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:  # 설명: 청크 파일 읽기
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows                                         # 설명: 청크 리스트 반환

# 설명: 테스트용 질의/키워드(간이 정답 판정). 문서에 맞게 자유롭게 수정 가능
QUERIES = [
    {"q": "과업 범위", "keywords": ["과업 범위", "업무 범위", "주요 과업"]},
    {"q": "사업 기간", "keywords": ["사업기간", "계약기간", "착수", "완료"]},
    {"q": "평가 기준", "keywords": ["평가 기준", "정량평가", "정성평가", "배점"]},
    {"q": "예산",     "keywords": ["금액", "원", "예산", "부가세"]},
    {"q": "도표",     "keywords": ["도표", "그림", "Figure", "Table", "표 "]},
]

def contains_any(text: str, keywords) -> bool:
    t = (text or "").lower()                           # 소문자 변환
    for k in keywords:
        if k.lower() in t:                             # 키워드 포함 여부
            return True
    return False

def resolve_search_fn(search_fn=None):
    """search 함수를 안전하게 확보한다."""
    if callable(search_fn):
        return search_fn                               # 주입된 함수가 있으면 그대로 사용

    # 1) 모듈에서 가져오기를 시도
    try:
        from search_hwpx import search as imported_search  # 파일로 분리해 둔 경우
        return imported_search
    except Exception:
        pass

    # 2) 같은 노트북/세션 전역에 정의된 search 찾기
    g = globals().get("search")                        # 전역에 search가 정의되어 있나 확인
    if callable(g):
        return g

    # 3) 여기까지 못 찾았으면 명확히 알리고 중단
    raise RuntimeError(
        "search() 함수를 찾지 못했습니다.\n"
        "- 같은 노트북 셀에 `search()`가 정의되어 있거나,\n"
        "- `search_hwpx.py` 파일에 `def search(...):`가 있고, 동일한 경로에서 실행해야 합니다.\n"
        "먼저 '검색 함수 + 가중치 리랭크' 코드 셀을 실행했는지 확인해 주세요."
    )

def eval_smoke(topk=10, search_fn=None):
    search_fn = resolve_search_fn(search_fn)           # search 함수를 안전하게 확보
    chunks = load_chunks()                             # index_id → 텍스트 매핑용 로드

    def get_text_by_index_id(index_id: int) -> str:
        if 0 <= index_id < len(chunks):                # 유효 인덱스인지 확인
            return chunks[index_id].get("text") or ""  # 청크 텍스트 반환
        return ""

    hits_at_3, rr_at_10 = [], []                       # 지표 누적 리스트

    for item in QUERIES:
        q   = item["q"]
        kws = item["keywords"]

        results = search_fn(q, topk=topk)              # 검색 실행 (리랭크 포함)
        rank_of_first_hit = inf                        # 기본은 미발견

        for i, r in enumerate(results[:topk], start=1):# 상위 topk만 검사
            txt = get_text_by_index_id(r["index_id"])  # 결과 인덱스로 청크 텍스트 조회
            if contains_any(txt, kws):                 # 키워드 중 하나라도 포함되면 정답
                rank_of_first_hit = i
                break

        hits_at_3.append(1.0 if rank_of_first_hit <= 3 else 0.0)            # Hit@3 계산
        rr_at_10.append(1.0 / rank_of_first_hit if rank_of_first_hit != inf else 0.0)  # MRR@10 계산

        print(f"[Q] {q} → rank@hit: {rank_of_first_hit if rank_of_first_hit!=inf else 'None'}")  # 설명: 개별 로그

    hit3  = sum(hits_at_3) / len(hits_at_3)           # Hit@3 평균
    mrr10 = sum(rr_at_10) / len(rr_at_10)             # MRR@10 평균
    print(f"[SMOKE] Hit@3 = {hit3:.2f}  |  MRR@10 = {mrr10:.2f}")  # 최종 지표 출력

if __name__ == "__main__":
    eval_smoke(topk=10)                                # 스모크 테스트 실행


# 1. Hit@3 / MRR@10 평가 지표
# 
# - **Hit@3**: 쿼리마다 **상위 3개 결과 안에 ‘정답’이 있으면 1, 없으면 0**으로 본 뒤 평균.
#     - 이번엔 5개 쿼리 중 3개가 Top-3 안에서 맞았으니 **3/5 = 0.60**.
# - **MRR@10**: 쿼리마다 **첫 번째 정답의 순위의 역수(1/rank)** 를 쓰고 평균. Top-10에 정답이 없으면 0.
#     - 로그 기준(순서대로) `None, 1, 2, 4, 2` → 역수는 `0, 1, 0.5, 0.25, 0.5`
#     - 평균은 `(0 + 1 + 0.5 + 0.25 + 0.5) / 5 = 0.45`.

# # 미세튜닝 : 리트리브 검색성능 향상

# In[ ]:


# 검색합수 & 리랭크 수정
# doc_test 1.10 up , ocr_image 0.85로 down --> 질의가 본문 청크로 더 잘 올라오도록

# -*- coding: utf-8 -*-
import os, re, json
# 표준 모듈
import numpy as np
# 수치 연산용
from pathlib import Path
# 경로 관리
import faiss
# 검색용 인덱스 로딩
from openai import OpenAI
# 쿼리 임베딩 계산

HOME = Path.home()
# 홈 경로
INTERIM = HOME / "data" / "interim"
# 산출물 경로
INDEX_BIN = INTERIM / "faiss_index_hwpx.bin"
# 인덱스 파일 경로
META_NPY  = INTERIM / "faiss_meta_hwpx.npy"
# 메타 배열 경로
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# 임베딩 모델 이름

client = OpenAI()
# OpenAI 클라이언트

# ---- 타입별 가중치 (기본 prior) ----
TYPE_PRIOR = {
    "doc_text": 1.00,     # 본문은 기본값 1.0   # 가중치 up
    "ocr_image": 0.90,    # OCR 개별은 노이즈 가능성으로 소폭 감점 # 0.85로 가중치 down
    "ocr_concat": 0.80    # 합본은 중복/주제 섞임 가능성으로 더 감점
}

# ---- 질의 키워드에 따른 보정 규칙 ----
OCR_KEYWORDS = ["그림", "도표", "표", "양식", "사진", "캡션", "Figure", "Table"]
# 표/그림 관련 질의 키워드 목록

def has_ocr_intent(q: str) -> bool:
    qn = q.lower()
    # 소문자 변환으로 일치율 향상
    if any(k.lower() in qn for k in OCR_KEYWORDS):
        return True
    # 키워드가 쿼리에 포함되면 True
    return False

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    # 쿼리 임베딩 계산
    v = np.array([resp.data[0].embedding], dtype="float32")
    # numpy float32 배열로 변환
    # 정규화 (코사인 내적)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    # 코사인 내적을 위해 단위 벡터화
    return v

def load_resources():
    if not INDEX_BIN.exists() or not META_NPY.exists():
        raise FileNotFoundError("인덱스/메타 파일이 없습니다. 먼저 인덱스를 빌드하세요.")
    # 리소스 존재 확인

    index = faiss.read_index(str(INDEX_BIN))
    # FAISS 인덱스 로드

    metas = np.load(META_NPY, allow_pickle=True)
    # 메타데이터 로드 (검색 결과 매핑용)

    return index, metas

def rerank(scores: np.ndarray, metas, query: str) -> np.ndarray:
    boosted = scores.copy()
    # 점수를 복사해 보정

    ocr_intent = has_ocr_intent(query)
    # 질의가 표/그림 의도인지 감지

    for i in range(len(metas)):
        mt = metas[i]
        t = (mt.get("block_type") or "").strip()
        # 블록 타입 추출

        prior = TYPE_PRIOR.get(t, 1.0)
        # 기본 타입별 가중치

        if ocr_intent:
            if t == "ocr_image":
                prior *= 1.15   # 표/그림 질의 시 OCR 이미지 상향
            elif t == "ocr_concat":
                prior *= 1.05   # 합본도 약간 상향
        # 질의 유형에 따른 보정
        else:
            if t == "doc_text":
                prior *= 1.10     # 일반  텍스트 질의에서 본문 0.10만큼 강화
            elif t == "ocr_image":
                prior *= 0.85         # ocr은 0.05 억제

        boosted[i] = boosted[i] * prior + (0.01 if t == "doc_text" else 0.0)
        # 기본적으로 doc_text에 아주 작은 상수 가산으로 동률 시 본문 우선

    return boosted

def search(query: str, topk=10):
    index, metas = load_resources()
    # 인덱스와 메타 로드

    qv = embed_query(query)
    # 쿼리 임베딩

    sims, idxs = index.search(qv, topk)
    # 코사인 유사도(내적) 기반 상위 topk 검색

    sims = sims[0]
    idxs = idxs[0]
    # 배치 차원 제거

    # 1차 점수/메타 수집
    raw = []
    for score, ix in zip(sims, idxs):
        mt = metas[ix].item() if isinstance(metas[ix], np.ndarray) else metas[ix]
        raw.append((score, ix, mt))
    # (점수, 인덱스, 메타) 튜플로 수집

    # 2차 리랭크 (타입별 가중치/질의 의도 반영)
    scores = np.array([r[0] for r in raw], dtype="float32")
    # 원 점수 배열
    re_scores = rerank(scores, [r[2] for r in raw], query)
    # 보정 점수 계산
    order = np.argsort(-re_scores)
    # 내림차순 정렬 인덱스

    results = []
    for rank in order:
        score, ix, mt = raw[rank]
        results.append({
            "rank": len(results)+1,
            "score_raw": float(score),
            "score_re": float(re_scores[rank]),
            "index_id": int(ix),
            "doc_id": mt.get("doc_id"),
            "block_type": mt.get("block_type"),
            "chunk_index": mt.get("chunk_index"),
            "parent_block_index": mt.get("parent_block_index"),
            "source": mt.get("source"),
            "meta": mt.get("meta", {})
        })
    # 정렬된 결과를 딕셔너리로 구성

    return results
    # 최종 결과 반환



# --- (검색 함수 + 리랭크 코드 아래에 추가) ---
if __name__ == "__main__":                                   # ← 스크립트처럼 직접 실행할 때만 동작
    try:
        test_q = "사업 기간"                                  # 간단한 테스트 질의
        out = search(test_q, topk=3)                          # 상위 3개만 간단 확인
        for r in out:
            print(r["rank"], r["doc_id"], r["block_type"], f'{r["score_re"]:.3f}')
        print("[OK] search() self-test passed")
    except Exception as e:
        import traceback
        print("[ERR] search() self-test failed:", repr(e))    # 에러 메시지 확인
        traceback.print_exc()                                 # 스택트레이스까지 출력


# In[ ]:


# 스모크 테스트 수정 : 예산 영역 키워드 추가(총사업비~원 단위)

# -*- coding: utf-8 -*-
import json, re                                      # 설명: 표준 모듈
from pathlib import Path                              # 설명: 경로 처리
from math import inf                                  # 설명: 무한대 상수(정답 랭크 없을 때 사용)

HOME = Path.home()                                    # 설명: 홈 폴더
INTERIM = HOME / "data" / "interim"                   # 설명: 중간 산출물 폴더
CHUNKS_FILE = INTERIM / "chunks_hwpx_800_200.jsonl"   # 설명: 검색 결과 인덱스→텍스트 매핑용 원본 청크 파일

def load_chunks():
    rows=[]
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:  # 설명: 청크 파일 읽기
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows                                         # 설명: 청크 리스트 반환

# 설명: 테스트용 질의/키워드(간이 정답 판정). 문서에 맞게 자유롭게 수정 가능
QUERIES = [
    {"q": "과업 범위", "keywords": ["과업 범위", "업무 범위", "주요 과업"]},
    {"q": "사업 기간", "keywords": ["사업기간", "계약기간", "착수", "완료"]},
    {"q": "평가 기준", "keywords": ["평가 기준", "정량평가", "정성평가", "배점"]},
    {"q": "예산",     "keywords": ["금액", "원", "예산", "부가세", "총사업비", "추정가격", "기초금액",
                                 "계약금액", "VAT", "부가세", "억/천만", "₩"]},
    {"q": "도표",     "keywords": ["도표", "그림", "Figure", "Table", "표 "]},
]

def contains_any(text: str, keywords) -> bool:
    t = (text or "").lower()                           # 설명: 소문자 변환
    for k in keywords:
        if k.lower() in t:                             # 설명: 키워드 포함 여부
            return True
    return False

def resolve_search_fn(search_fn=None):
    """search 함수를 안전하게 확보한다."""
    if callable(search_fn):
        return search_fn                               # 설명: 주입된 함수가 있으면 그대로 사용

    # 1) 모듈에서 가져오기를 시도
    try:
        from search_hwpx import search as imported_search  # 설명: 파일로 분리해 둔 경우
        return imported_search
    except Exception:
        pass

    # 2) 같은 노트북/세션 전역에 정의된 search 찾기
    g = globals().get("search")                        # 설명: 전역에 search가 정의되어 있나 확인
    if callable(g):
        return g

    # 3) 여기까지 못 찾았으면 명확히 알리고 중단
    raise RuntimeError(
        "search() 함수를 찾지 못했습니다.\n"
        "- 같은 노트북 셀에 `search()`가 정의되어 있거나,\n"
        "- `search_hwpx.py` 파일에 `def search(...):`가 있고, 동일한 경로에서 실행해야 합니다.\n"
        "먼저 '검색 함수 + 가중치 리랭크' 코드 셀을 실행했는지 확인해 주세요."
    )

def eval_smoke(topk=10, search_fn=None):
    search_fn = resolve_search_fn(search_fn)           # 설명: search 함수를 안전하게 확보
    chunks = load_chunks()                             # 설명: index_id → 텍스트 매핑용 로드

    def get_text_by_index_id(index_id: int) -> str:
        if 0 <= index_id < len(chunks):                # 설명: 유효 인덱스인지 확인
            return chunks[index_id].get("text") or ""  # 설명: 청크 텍스트 반환
        return ""

    hits_at_3, rr_at_10 = [], []                       # 설명: 지표 누적 리스트

    for item in QUERIES:
        q   = item["q"]
        kws = item["keywords"]

        results = search_fn(q, topk=topk)              # 설명: 검색 실행 (리랭크 포함)
        rank_of_first_hit = inf                        # 설명: 기본은 미발견

        for i, r in enumerate(results[:topk], start=1):# 설명: 상위 topk만 검사
            txt = get_text_by_index_id(r["index_id"])  # 설명: 결과 인덱스로 청크 텍스트 조회
            if contains_any(txt, kws):                 # 설명: 키워드 중 하나라도 포함되면 정답
                rank_of_first_hit = i
                break

        hits_at_3.append(1.0 if rank_of_first_hit <= 3 else 0.0)            # 설명: Hit@3 계산
        rr_at_10.append(1.0 / rank_of_first_hit if rank_of_first_hit != inf else 0.0)  # 설명: MRR@10 계산

        print(f"[Q] {q} → rank@hit: {rank_of_first_hit if rank_of_first_hit!=inf else 'None'}")  # 설명: 개별 로그

    hit3  = sum(hits_at_3) / len(hits_at_3)           # 설명: Hit@3 평균
    mrr10 = sum(rr_at_10) / len(rr_at_10)             # 설명: MRR@10 평균
    print(f"[SMOKE] Hit@3 = {hit3:.2f}  |  MRR@10 = {mrr10:.2f}")  # 설명: 최종 지표 출력

if __name__ == "__main__":
    eval_smoke(topk=10)                                # 설명: 스모크 테스트 실행


# In[ ]:


# 스모크 테스트 호출 (search 함수를 직접 주입해서 호출)
eval_smoke(topk=10, search_fn=search)                 # 설명: 상위 10개 검색 → Hit@3/MRR@10 출력
# ↑ 출력 예: [Q] 과업 범위 → rank@hit: 2  / [SMOKE] Hit@3 = 0.60 | MRR@10 = 0.52


# ## 검색함수 & 리랭크 수정

# In[ ]:


# 미세튜닝 2

# 검색합수 & 리랭크 수정
# doc_text 1.12 up , ocr_image 0.80로 down --> 질의가 본문 청크로 더 잘 올라오도록
# OCR_KEYWORDS에 키워드 추가 : 캡처, 스캔, 이미지, 첨부


# -*- coding: utf-8 -*-
import os, re, json
# 표준 모듈
import numpy as np
# 수치 연산용
from pathlib import Path
# 경로 관리
import faiss
# 검색용 인덱스 로딩
from openai import OpenAI
# 쿼리 임베딩 계산

HOME = Path.home()
# 홈 경로
INTERIM = HOME / "data" / "interim"
# 산출물 경로
INDEX_BIN = INTERIM / "faiss_index_hwpx.bin"
# 인덱스 파일 경로
META_NPY  = INTERIM / "faiss_meta_hwpx.npy"
# 메타 배열 경로
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# 임베딩 모델 이름

client = OpenAI()
# OpenAI 클라이언트

# ---- 타입별 가중치 (기본 prior) ----
TYPE_PRIOR = {
    "doc_text": 1.00,     # 본문은 기본값 1.0   # 가중치 up
    "ocr_image": 0.90,    # OCR 개별은 노이즈 가능성으로 소폭 감점 # 0.85로 가중치 down
    "ocr_concat": 0.80    # 합본은 중복/주제 섞임 가능성으로 더 감점
}

# ---- 질의 키워드에 따른 보정 규칙 ----   # 질의 키워드 추가
OCR_KEYWORDS = ["그림", "도표", "표", "양식", "사진", "캡션", "Figure", "Table",
               "캡처", "스캔", "이미지", "첨부"]
# 표/그림 관련 질의 키워드 목록

def has_ocr_intent(q: str) -> bool:
    qn = q.lower()
    # 소문자 변환으로 일치율 향상
    if any(k.lower() in qn for k in OCR_KEYWORDS):
        return True
    # 키워드가 쿼리에 포함되면 True
    return False

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    # 쿼리 임베딩 계산
    v = np.array([resp.data[0].embedding], dtype="float32")
    # numpy float32 배열로 변환
    # 정규화 (코사인 내적)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    # 코사인 내적을 위해 단위 벡터화
    return v

def load_resources():
    if not INDEX_BIN.exists() or not META_NPY.exists():
        raise FileNotFoundError("인덱스/메타 파일이 없습니다. 먼저 인덱스를 빌드하세요.")
    # 리소스 존재 확인

    index = faiss.read_index(str(INDEX_BIN))
    # FAISS 인덱스 로드

    metas = np.load(META_NPY, allow_pickle=True)
    # 메타데이터 로드 (검색 결과 매핑용)

    return index, metas

def rerank(scores: np.ndarray, metas, query: str) -> np.ndarray:
    boosted = scores.copy()
    # 점수를 복사해 보정

    ocr_intent = has_ocr_intent(query)
    # 질의가 표/그림 의도인지 감지

    for i in range(len(metas)):
        mt = metas[i]
        t = (mt.get("block_type") or "").strip()
        # 블록 타입 추출

        prior = TYPE_PRIOR.get(t, 1.0)
        # 기본 타입별 가중치

        if ocr_intent:
            if t == "ocr_image":
                prior *= 1.15   # 표/그림 질의 시 OCR 이미지 상향
            elif t == "ocr_concat":
                prior *= 1.05   # 합본도 약간 상향
        # 질의 유형에 따른 보정
        else:
            if t == "doc_text":
                prior *= 1.15     # 일반  텍스트 질의에서 본문 0.10만큼 강화
            elif t == "ocr_image":
                prior *= 0.75         # ocr은 0.05 억제

        boosted[i] = boosted[i] * prior + (0.01 if t == "doc_text" else 0.0)
        # 기본적으로 doc_text에 아주 작은 상수 가산으로 동률 시 본문 우선

    return boosted

def search(query: str, topk=10):
    index, metas = load_resources()
    # 인덱스와 메타 로드

    qv = embed_query(query)
    # 쿼리 임베딩

    sims, idxs = index.search(qv, topk)
    # 코사인 유사도(내적) 기반 상위 topk 검색

    sims = sims[0]
    idxs = idxs[0]
    # 배치 차원 제거

    # 1차 점수/메타 수집
    raw = []
    for score, ix in zip(sims, idxs):
        mt = metas[ix].item() if isinstance(metas[ix], np.ndarray) else metas[ix]
        raw.append((score, ix, mt))
    # (점수, 인덱스, 메타) 튜플로 수집

    # 2차 리랭크 (타입별 가중치/질의 의도 반영)
    scores = np.array([r[0] for r in raw], dtype="float32")
    # 원 점수 배열
    re_scores = rerank(scores, [r[2] for r in raw], query)
    # 보정 점수 계산
    order = np.argsort(-re_scores)
    # 내림차순 정렬 인덱스

    results = []
    for rank in order:
        score, ix, mt = raw[rank]
        results.append({
            "rank": len(results)+1,
            "score_raw": float(score),
            "score_re": float(re_scores[rank]),
            "index_id": int(ix),
            "doc_id": mt.get("doc_id"),
            "block_type": mt.get("block_type"),
            "chunk_index": mt.get("chunk_index"),
            "parent_block_index": mt.get("parent_block_index"),
            "source": mt.get("source"),
            "meta": mt.get("meta", {})
        })
    # 정렬된 결과를 딕셔너리로 구성

    return results
    # 최종 결과 반환



# --- (검색 함수 + 리랭크 코드 아래에 추가) ---
if __name__ == "__main__":                                   # ← 스크립트처럼 직접 실행할 때만 동작
    try:
        test_q = "사업 기간"                                  # 간단한 테스트 질의
        out = search(test_q, topk=3)                          # 상위 3개만 간단 확인
        for r in out:
            print(r["rank"], r["doc_id"], r["block_type"], f'{r["score_re"]:.3f}')
        print("[OK] search() self-test passed")
    except Exception as e:
        import traceback
        print("[ERR] search() self-test failed:", repr(e))    # 에러 메시지 확인
        traceback.print_exc()                                 # 스택트레이스까지 출력



# In[ ]:


# 스모크 테스트 수정 : 예산 영역 키워드 추가(총사업비~원 단위)

# -*- coding: utf-8 -*-
import json, re                                      # 설명: 표준 모듈
from pathlib import Path                              # 설명: 경로 처리
from math import inf                                  # 설명: 무한대 상수(정답 랭크 없을 때 사용)

HOME = Path.home()                                    # 설명: 홈 폴더
INTERIM = HOME / "data" / "interim"                   # 설명: 중간 산출물 폴더
CHUNKS_FILE = INTERIM / "chunks_hwpx_800_200.jsonl"   # 설명: 검색 결과 인덱스→텍스트 매핑용 원본 청크 파일

def load_chunks():
    rows=[]
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:  # 설명: 청크 파일 읽기
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows                                         # 설명: 청크 리스트 반환

# 설명: 테스트용 질의/키워드(간이 정답 판정). 문서에 맞게 자유롭게 수정 가능
QUERIES = [
    {"q": "과업 범위", "keywords": ["과업 범위", "업무 범위", "주요 과업"]},
    {"q": "사업 기간", "keywords": ["사업기간", "계약기간", "착수", "완료"]},
    {"q": "평가 기준", "keywords": ["평가 기준", "정량평가", "정성평가", "배점"]},
    {"q": "예산",     "keywords": ["금액", "원", "예산", "부가세", "총사업비", "추정가격", "기초금액",
                                 "계약금액", "VAT", "부가세", "억/천만", "₩"]},
    {"q": "도표",     "keywords": ["도표", "그림", "Figure", "Table", "표 "]},
]

def contains_any(text: str, keywords) -> bool:
    t = (text or "").lower()                           # 설명: 소문자 변환
    for k in keywords:
        if k.lower() in t:                             # 설명: 키워드 포함 여부
            return True
    return False

def resolve_search_fn(search_fn=None):
    """search 함수를 안전하게 확보한다."""
    if callable(search_fn):
        return search_fn                               # 설명: 주입된 함수가 있으면 그대로 사용

    # 1) 모듈에서 가져오기를 시도
    try:
        from search_hwpx import search as imported_search  # 설명: 파일로 분리해 둔 경우
        return imported_search
    except Exception:
        pass

    # 2) 같은 노트북/세션 전역에 정의된 search 찾기
    g = globals().get("search")                        # 설명: 전역에 search가 정의되어 있나 확인
    if callable(g):
        return g

    # 3) 여기까지 못 찾았으면 명확히 알리고 중단
    raise RuntimeError(
        "search() 함수를 찾지 못했습니다.\n"
        "- 같은 노트북 셀에 `search()`가 정의되어 있거나,\n"
        "- `search_hwpx.py` 파일에 `def search(...):`가 있고, 동일한 경로에서 실행해야 합니다.\n"
        "먼저 '검색 함수 + 가중치 리랭크' 코드 셀을 실행했는지 확인해 주세요."
    )

def eval_smoke(topk=10, search_fn=None):
    search_fn = resolve_search_fn(search_fn)           # 설명: search 함수를 안전하게 확보
    chunks = load_chunks()                             # 설명: index_id → 텍스트 매핑용 로드

    def get_text_by_index_id(index_id: int) -> str:
        if 0 <= index_id < len(chunks):                # 설명: 유효 인덱스인지 확인
            return chunks[index_id].get("text") or ""  # 설명: 청크 텍스트 반환
        return ""

    hits_at_3, rr_at_10 = [], []                       # 설명: 지표 누적 리스트

    for item in QUERIES:
        q   = item["q"]
        kws = item["keywords"]

        results = search_fn(q, topk=topk)              # 설명: 검색 실행 (리랭크 포함)
        rank_of_first_hit = inf                        # 설명: 기본은 미발견

        for i, r in enumerate(results[:topk], start=1):# 설명: 상위 topk만 검사
            txt = get_text_by_index_id(r["index_id"])  # 설명: 결과 인덱스로 청크 텍스트 조회
            if contains_any(txt, kws):                 # 설명: 키워드 중 하나라도 포함되면 정답
                rank_of_first_hit = i
                break

        hits_at_3.append(1.0 if rank_of_first_hit <= 3 else 0.0)            # 설명: Hit@3 계산
        rr_at_10.append(1.0 / rank_of_first_hit if rank_of_first_hit != inf else 0.0)  # 설명: MRR@10 계산

        print(f"[Q] {q} → rank@hit: {rank_of_first_hit if rank_of_first_hit!=inf else 'None'}")  # 설명: 개별 로그

    hit3  = sum(hits_at_3) / len(hits_at_3)           # 설명: Hit@3 평균
    mrr10 = sum(rr_at_10) / len(rr_at_10)             # 설명: MRR@10 평균
    print(f"[SMOKE] Hit@3 = {hit3:.2f}  |  MRR@10 = {mrr10:.2f}")  # 설명: 최종 지표 출력

if __name__ == "__main__":
    eval_smoke(topk=10)                                # 설명: 스모크 테스트 실행


# In[ ]:


# 스모크 테스트 호출 (search 함수를 직접 주입해서 호출)
eval_smoke(topk=10, search_fn=search)                 # 설명: 상위 10개 검색 → Hit@3/MRR@10 출력
# ↑ 출력 예: [Q] 과업 범위 → rank@hit: 2  / [SMOKE] Hit@3 = 0.60 | MRR@10 = 0.52


# [결과]
# 
# - doc_text와 doc_image의 가중치를 조정했으나, 결과는 동일함

# ## **문자 그대로 매칭 추가 : 보너스 lexical boost 추가**

# **[패치 포인트]** - 검색 모듈에 추가
# 
# - 상단에 청크 원문을 읽어오는 헬퍼와 도메인 키워드, 보너스 함수 추가
# - ``search()`` 안에서 초기 topK에 대해 보너스를 더한 뒤 최종 정렬

# **[패치 포인트의 효과]**
# 
# 
# 
# - 지금 검색은 “의미가 비슷한 내용”을 잘 찾지만, **정확히 같은 단어**(예: “과업 범위”, “평가 기준”, “예산”, “도표”)가 있는 문장을 **더 위로** 올리는 힘은 약함
# 
# 
# - 그래서, 상위 후보들에 한해 **문자 그대로 매칭 보너스**를 조금 얹는다.
#     - 질문과 **정확히 같은 말**이나 **동의어**가 본문에 있으면 ``+점수``
#     - 본문 **처음 부분(표제/머리)** 에 핵심어가 나오면 ``+점수``
#     - “예산/금액” 질문인데 **숫자+단위(원, 억, ₩, VAT 등)** 가 같이 있으면 ``+점수``
#     - “기간/착수 ~ 완료” 질문인데 **날짜 범위**(예: 2024.01.01~2024.12.31)가 보이면 ``+점수``
#     - “도표/그림” 질문인데 **표/그림 키워드**와 **숫자가 많은 텍스트**면 ``+점수``
# 
# 
# > 이렇게 하면 **Top-3 안에 들던 결과**가 **Top-1~2**로 더 자주 올라와서 **MRR(정답의 평균 순위)** 가 상승하는 효과가 나타남
# 

# In[ ]:


# --- 상단 유틸(한 번만 선언) ---
import re, json
from pathlib import Path
import numpy as np  # ✅ 꼭 필요!

HOME = Path.home()
INTERIM = HOME / "data" / "interim"
CHUNKS_FILE = INTERIM / "chunks_hwpx_800_200.jsonl"

# 도메인 동의어 세트(필요시 추가/수정)
DOMAIN_KEYWORDS = {
    "과업 범위": ["과업 범위","과업범위","업무 범위","수행 범위","주요 과업","Scope of Work","SOW","과업의 범위"],
    "사업 기간": ["사업기간","계약기간","수행기간","착수","완료"],
    "평가 기준": ["평가 기준","평가기준","평가항목","배점","정량평가","정성평가"],
    "예산":     ["예산","금액","기초금액","추정가격","총사업비","계약금액","VAT","부가세","원","억","천만"],
    "도표":     ["도표","그림","Figure","Table","표 "],
}

_CH_TEXTS = None
def _load_chunk_texts():
    """인덱스 빌드에 쓴 청크 텍스트(빈 텍스트 제외)를 인덱스 순서대로 로드"""
    global _CH_TEXTS
    if _CH_TEXTS is None:
        _CH_TEXTS = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = (rec.get("text") or "").strip()
                if t:  # 인덱스 빌드시 빈 텍스트 제외했으므로 동일하게
                    _CH_TEXTS.append(t)
    return _CH_TEXTS

def _get_chunk_text(ix: int) -> str:
    texts = _load_chunk_texts()
    return texts[ix] if 0 <= ix < len(texts) else ""

def _lexical_bonus(query: str, text: str) -> float:
    """정확/동의어/토큰 매치에 따른 보너스 (최대 0.20)"""
    q_norm = (query or "").lower().replace(" ", "")
    t_norm = (text or "").lower().replace(" ", "")
    bonus = 0.0

    # 1) 도메인 동의어 그룹 보너스
    syns = []
    for key, lst in DOMAIN_KEYWORDS.items():
        if key.replace(" ","") in q_norm or any(s.replace(" ","") in q_norm for s in lst):
            syns = list(set([key] + lst))
            break
    for kw in syns:
        if kw.lower().replace(" ","") in t_norm:
            bonus += 0.05   # 동의어 1개 매치당 +0.05

    # 2) 토큰 중첩(간단 오버랩) 보너스
    toks = [tok for tok in re.split(r"[\s/():·,\-]+", (query or "").lower()) if len(tok) >= 2]
    match_cnt = sum(1 for tok in toks if tok in (text or "").lower())
    bonus += min(0.02 * match_cnt, 0.08)

    # 3) 헤딩/앵커 형태 가산 (문서 표제/캡션에 자주 등장)
    if any(h in (text or "") for h in ["과업 범위","평가 기준","사업 기간","예산","금액"]):
        bonus += 0.05

    return min(bonus, 0.20)

# --- 통째로 교체용 search() ---
def search(query: str, topk=10):
    index, metas = load_resources()                        # 인덱스/메타 로드
    qv = embed_query(query)                                # 쿼리 임베딩
    sims, idxs = index.search(qv, topk)                    # 1차 검색
    sims, idxs = sims[0], idxs[0]

    # 후보 수집
    raw = []
    for score, ix in zip(sims, idxs):
        mt = metas[ix].item() if isinstance(metas[ix], np.ndarray) else metas[ix]
        raw.append((float(score), int(ix), mt))            # (원점수, 인덱스, 메타)

    # 타입별 prior 리랭크 (임베딩 점수 보정)
    scores = np.array([r[0] for r in raw], dtype="float32")
    re_scores = rerank(scores, [r[2] for r in raw], query) # 여기서 re_scores 생성

    # === lexical bonus 적용 (topK 후보 텍스트에 대한 소량 가산점) ===
    lex = np.zeros_like(re_scores)
    for j, (_, ix, _) in enumerate(raw):
        txt = _get_chunk_text(ix)
        lex[j] = _lexical_bonus(query, txt)

    final_scores = re_scores + lex                         # 보너스 합산
    order = np.argsort(-final_scores)                      # 최종 정렬

    # 결과 포맷
    results = []
    for out_rank, j in enumerate(order, start=1):
        score_raw, ix, mt = raw[j]
        results.append({
            "rank": out_rank,
            "score_raw": float(score_raw),
            "score_re": float(final_scores[j]),           # 최종 점수
            "index_id": int(ix),
            "doc_id": mt.get("doc_id"),
            "block_type": mt.get("block_type"),
            "chunk_index": mt.get("chunk_index"),
            "parent_block_index": mt.get("parent_block_index"),
            "source": mt.get("source"),
            "meta": mt.get("meta", {})
        })
    return results


# In[ ]:


# 스모크 테스트 호출 (search 함수를 직접 주입해서 호출)
eval_smoke(topk=10, search_fn=search)                 # 설명: 상위 10개 검색 → Hit@3/MRR@10 출력
# ↑ 출력 예: [Q] 과업 범위 → rank@hit: 2  / [SMOKE] Hit@3 = 0.60 | MRR@10 = 0.52


# In[ ]:


eval_smoke(topk=10, search_fn=search)


# In[ ]:


# 여기 나온 임베딩 모델만 바로 쓸 수 있음
from openai import OpenAI; import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
avail = [m.id for m in client.models.list() if "embedding" in m.id]
print(avail)  # 여기 나온 것만 바로 쓸 수 있음ㄴ


# # RAG 응답 (Search 결과로 실제 답변 생성)

# - ``search()`` 결과 상위에서 문서 다양한 보장
#     - 3 ~ 4개만 뽑아서 gpt-4.1.mini로 컨텍스트 기반 답변 생성 
# - 출처 태그를 달아서 검증 쉬움

# ## RAG 최소 러너 버전

# In[ ]:


# -*- coding: utf-8 -*-
# HWPX RAG 러너: 검색결과 → 컨텍스트 구성 → gpt-4.1-mini로 답변 생성(+출처 태그)

import os, json
from pathlib import Path
from openai import OpenAI

# 0) 준비
HOME = Path.home()
INTERIM = HOME / "data" / "interim"
CHUNKS_FILE = INTERIM / "chunks_hwpx_800_200.jsonl"
OUT_JSONL   = INTERIM / "rag_hwpx_results.jsonl"

# search() 확보 (이미 노트북에 있다면 이 try 블록은 건너뜀)
try:
    _ = search  # noqa: F821
except NameError:
    from search_hwpx import search  # 파일로 분리했다면 여기서 import

# 1) 청크 로드
def _load_chunks(path=CHUNKS_FILE):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("청크가 비었습니다. chunks_hwpx_800_200.jsonl 확인 필요.")
    return rows

CHUNKS = _load_chunks()

# 2) OpenAI 클라이언트 (모델명 명시! 기본값에 의존 X)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3) RAG 생성 함수
def rag_answer(question: str, topk=10, max_ctx=4, per_ctx_chars=500, model="gpt-4.1-mini"):
    # (a) 검색 (A안 적용된 search() 사용)
    results = search(question, topk=topk)

    # (b) 문서 다양성 보장: 서로 다른 doc_id 위주로 상위 컨텍스트 선택
    picked, seen = [], set()
    for r in results:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        picked.append(r)
        if len(picked) >= max_ctx:
            break

    # (c) 컨텍스트 블록 만들기(+출처 태그)
    ctx_blocks = []
    for r in picked:
        idx = r["index_id"]
        text = CHUNKS[idx]["text"] if 0 <= idx < len(CHUNKS) else ""
        text = (text or "")[:per_ctx_chars]
        tag  = f"[{r['doc_id']}:{r['block_type']}:{r['chunk_index']}]"
        ctx_blocks.append(f"{tag}\n{text}")

    system = ("You are a Korean RAG assistant. Answer ONLY from the provided context. "
              "If uncertain, say you don't know. Include source tags like [doc:block:chunk].")
    user = f"질문: {question}\n\n컨텍스트:\n" + "\n\n---\n\n".join(ctx_blocks)

    resp = client.chat.completions.create(
        model=model,  # 모델 명시! (기본 gpt-3.5로 가지 않도록)
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.0,
    )
    answer = (resp.choices[0].message.content or "").strip()

    # (d) 결과 레코드 구성(파일 저장용)
    rec = {
        "question": question,
        "answer": answer,
        "contexts": [{
            "doc_id": r["doc_id"],
            "block_type": r["block_type"],
            "chunk_index": r["chunk_index"],
            "index_id": r["index_id"],
            "score": r["score_re"]
        } for r in picked]
    }
    return rec

# 4) 스모크용 질문으로 RAG 실행 + 저장
SMOKE = [
    {"question":"과업 범위"},
    {"question":"사업 기간"},
    {"question":"평가 기준"},
    {"question":"예산"},
    {"question":"도표"},
]

with open(OUT_JSONL, "w", encoding="utf-8") as out:
    for it in SMOKE:
        q = it["question"]
        rec = rag_answer(q, topk=10, max_ctx=4, per_ctx_chars=500, model="gpt-4.1-mini")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n[Q] {q}\n{rec['answer']}\n")
        print(f"[Saved] → {OUT_JSONL}")


# ## 생성품질 체크(가벼운 버전)

# - 생성 결과에 정답 키워드가 답변에 들어갔는지 확인

# In[ ]:


# -*- coding: utf-8 -*-
# RAG 생성 결과의 초간단 정답 감지(키워드 기반)

import json
from pathlib import Path

INTERIM = Path.home() / "data" / "interim"
OUT_JSONL = INTERIM / "rag_hwpx_results.jsonl"

# 질문별 키워드(스모크와 동일/확장 가능)
KEYS = {
    "과업 범위": ["과업 범위","업무 범위","주요 과업","수행 범위","SOW"],
    "사업 기간": ["사업기간","계약기간","착수","완료","기간"],
    "평가 기준": ["평가 기준","평가기준","배점","정량평가","정성평가"],
    "예산":     ["예산","금액","총사업비","부가세","VAT","원","억","천만"],
    "도표":     ["도표","그림","Figure","Table","표 "],
}

ok, total = 0, 0
with open(OUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        q, a = rec["question"], (rec["answer"] or "")
        total += 1
        hit = any(k.lower() in a.lower() for k in KEYS.get(q, []))
        ok += 1 if hit else 0
        print(f"[{q}] {'OK' if hit else 'MISS'}")

print(f"\n[GEN-EVAL] keyword-hit rate = {ok}/{total} = {ok/total:.2f}")


# In[ ]:





# In[ ]:




