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


# 개인폴 넣고 실행
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


# In[13]:


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


# In[15]:


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

# In[16]:


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

# In[17]:


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

# ### 1-1. hwp -> hwpt

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


# In[54]:


# 포맷 버전별 파일 생성
# hwpx, html, xml, doc, pdf

get_ipython().system('mkdir ~/data/raw/converted/{hwpx,html,xml,docs,pdf}')
# {}안에 꼭 붙여써야 4가지 디렉토리가 각각 생성됨


# In[1]:


# 로컬에서 포맷 변환한 파일들 업로드 : 총 4개


# In[56]:


# html, xml로 변환된 파일의 이미지 파일들의 텍스트를
# 읽어와야 하기에 디렉터리 구조 추가


# In[ ]:


get_ipython().system('ls ~/data/raw/converted/')


# In[60]:


get_ipython().system('mkdir -p ~/data/raw/converted/html/{한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보,한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축,한국한의학연구원_통합정보시스템_고도화_용역,한국철도공사_용역_예약발매시스템_개량_ISMP_용역}')
get_ipython().system('mkdir -p ~/data/raw/converted/xml/{한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보,한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축,한국한의학연구원_통합정보시스템_고도화_용역,한국철도공사_용역_예약발매시스템_개량_ISMP_용역}')


# In[ ]:


get_ipython().system('ls ~/data/raw/converted/html')
print("="*5)
get_ipython().system('ls ~/data/raw/converted/xml')


# In[68]:


get_ipython().system('mkdir -p ~/data/raw/converted/html/한영대학_한영대학교_특성화_맞춤형_교육환경_구축_트랙운영_학사정보/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국해양조사협회_2024년_항해용_간행물_품질관리_업무보조_시스템_구축/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국한의학연구원_통합정보시스템_고도화_용역/assets')
get_ipython().system('mkdir -p ~/data/raw/converted/html/한국철도공사_용역_예약발매시스템_개량_ISMP_용역/assets')


# In[69]:


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


# #### 1) HWPX -> page_hwpx.jsonl

# In[27]:


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


# #### 1-1) jupyterhub에서 홈 디렉토리에 API KEY를 환경변수 저장

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
                "ts": now_ks_iso(),
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

# ##### HWPX 이미지 추출

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


# In[72]:


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


# In[88]:


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


# ### 1-2. HWP → HTML/XML

# #### HTML이랑 XML을 같이 돌리는 이유
# 
# - **상호보완**
#     - 같은 HWP라도 변환기/옵션에 따라 **HTML에는 보이는 내용이 XML엔 빠지거나**
#     - 반대로 **XML엔 남는데 HTML 변환에서 누락**되는 케이스가 실제로 있음(표 캡션, 각주, 특수기호, 목차 앵커 등).
# 
# 
# 
# - **자산(assets) 차이**
# 
#     - 어떤 문서는 **이미지가 HTML 쪽 assets에만** 제대로 비트맵으로 나와 있고
#     - 어떤 건 **XML 쪽 assets에만** 살아있는 경우가 있음
#        > 둘 다 훑어야 **이미지 기반 정보(도표/스샷/양식)**를 놓치지 않음
# 
# 
# - **구조 vs 편의**
# 
#     - XML은 구조(태그)가 더 엄격해서 **세밀한 청킹/후처리**에 유리하고
#     -  HTML은 파싱이 쉬워 **대량처리**에 유리함
#        > 둘 다 뽑아두면 이후 단계에서 **품질/속도 트레이드오프**를 유연하게 선택 가능.

# #### 1) HTML 텍스트 추출 -> page_html.jsonl로 저장

# - 문서별 폴더의 *.html 전체 -> 한 레코드
# - beautifulsoup 사전 설치 필수

# In[ ]:


# -*- coding: utf-8 -*-
"""
/raw/converted/html/<doc_id>/*.html 에서 본문 텍스트를 뽑아 JSONL 저장
출력: /home/spai0308/data/interim/pages_html.jsonl
"""
import json, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup  # pip install beautifulsoup4

DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"
OUT = DATA / "interim" / "pages_html.jsonl"

KST = timezone(timedelta(hours=9))
def now_iso(): return datetime.now(KST).isoformat()

def extract_html_dir(doc_dir: Path) -> str:
    """해당 문서 폴더의 모든 *.html 텍스트를 합쳐서 반환"""
    texts = []
    for html in sorted(doc_dir.glob("*.html")):
        try:
            # 기본 파서(html.parser)는 내장이라 설치 불필요
            soup = BeautifulSoup(html.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            # 스크립트/스타일 제거
            for tag in soup(["script","style"]): tag.decompose()
            # 의미 있는 줄바꿈 보존
            text = soup.get_text("\n", strip=True)
            if text:
                # 너무 연속된 빈 줄 줄이기
                text = "\n".join([line for line in text.splitlines() if line.strip()!=""])
                texts.append(text)
        except Exception as e:
            texts.append(f"[HTML_PARSE_ERROR] {html.name}: {e}")
    return "\n".join(texts)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for doc_dir in sorted(HTML_ROOT.glob("*")):
            if not doc_dir.is_dir(): continue
            doc_id = doc_dir.name
            text = extract_html_dir(doc_dir)
            rec = {
                "doc_id": doc_id,
                "source_path": str(doc_dir),
                "text": text,
                "ts": now_iso(),
                "source_type": "html_text"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] {OUT}")

if __name__ == "__main__":
    main()


# In[ ]:


# 확인
# 줄 수 확인
get_ipython().system('wc -l /home/spai0308/data/interim/pages_html.jsonl')
print("="*100)

# 샘플 보기 : 첫번째 줄만
get_ipython().system('head -n 1 ~/data/interim/pages_html.jsonl | jq .')
# jq . : 사람이 읽기 좋게 줄바꿈해서 출력(\n로 구분)

# 샘플 각 줄 씩 확인
get_ipython().system("sed -n '4p' /home/spai0308/data/interim/pages_html.jsonl | jq .")

# sed -n 'Np' file : 파일에서 N번째 줄만 출력


# #### 2) XML 텍스트 추출  → pages_xml.jsonl

# - 문서별 폴더의 *.xml 전체 -> 한 레코드로 추출

# In[ ]:


# -*- coding: utf-8 -*-
"""
/raw/converted/xml/<doc_id>/*.xml 에서 텍스트 노드만 추출 → JSONL
출력: /home/spai0308/data/interim/pages_xml.jsonl
"""
import json, glob, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA = Path("/home/spai0308/data")
XML_ROOT = DATA / "raw" / "converted" / "xml"
OUT = DATA / "interim" / "pages_xml.jsonl"

KST = timezone(timedelta(hours=9))
def now_iso(): return datetime.now(KST).isoformat()

def extract_xml_dir(doc_dir: Path) -> str:
    texts = []
    for xmlf in sorted(doc_dir.glob("*.xml")):
        try:
            root = ET.fromstring(xmlf.read_bytes())
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
        except Exception as e:
            texts.append(f"[XML_PARSE_ERROR] {xmlf.name}: {e}")
    return "\n".join(texts)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for doc_dir in sorted(XML_ROOT.glob("*")):
            if not doc_dir.is_dir(): continue
            doc_id = doc_dir.name
            text = extract_xml_dir(doc_dir)
            rec = {
                "doc_id": doc_id,
                "source_path": str(doc_dir),
                "text": text,
                "ts": now_iso(),
                "source_type": "xml_text"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] {OUT}")

if __name__ == "__main__":
    main()


# In[ ]:


# 대시보드 확인

# 확인
# 줄 수 확인
get_ipython().system('wc -l /home/spai0308/data/interim/pages_xml.jsonl')
print("="*100)

# 샘플 보기 : 첫번째 줄만
get_ipython().system('head -n 1 ~/data/interim/pages_xml.jsonl | jq .')

# jq . : 사람이 읽기 좋게 줄바꿈해서 출력(\n로 구분)
print("="*100)

# 샘플 각 줄 씩 확인
get_ipython().system("sed -n '1p' /home/spai0308/data/interim/pages_xml.jsonl | jq .")

# sed -n 'Np' file : 파일에서 N번째 줄만 출력


# #### 3) HTML 자산 OCR로 돌리기

# In[ ]:


# -*- coding: utf-8 -*-
"""
목적:
- /home/spai0308/data/raw/converted/html/**/assets/* 아래의 이미지 파일들에서
  OpenAI 비전 모델로 텍스트를 읽어 JSONL로 저장합니다.

왜 이렇게 하나요?
- 많은 RFP/제안서에서 표나 스크린샷 등 '이미지 안 글자'가 중요합니다.
- HTML 변환 시 'assets' 폴더에 이미지가 모여 있으니 그걸 한 장씩 OCR 합니다.
- 중복 방지(캐시)로 이미 처리한 이미지는 건너뛰어 시간/비용을 절약합니다.

출력:
- /home/spai0308/data/interim/assets_html_ocr.jsonl  ← 이미지별 OCR 결과가 줄 단위로 저장됨
- /home/spai0308/data/interim/assets_openai_ocr.global.cache.json ← 이미 처리한 이미지 해시 캐시
"""

# ---- 기본 라이브러리 불러오기 (파일경로, 시간, 이미지 핸들링 등) ----
import os, io, json, glob, base64, time, random, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from openai import APITimeoutError, APIError, RateLimitError

# ---- 이미지 처리 라이브러리: Pillow(PIL) ----
# Image: 이미지 열기/다루기, ImageSequence: GIF 같은 멀티프레임 처리, UnidentifiedImageError: 이미지가 아닐 때 예외
from PIL import Image, ImageSequence, UnidentifiedImageError

# ---- 진행률 바 표시(진행 상황 눈에 보이게) ----
from tqdm import tqdm

# ---- OpenAI Python SDK (새 클라이언트) ----
# OPENAI_API_KEY는 사전에 환경변수로 넣어두어야 합니다.
from openai import OpenAI

# ========================= 경로 설정 =========================
# 데이터 루트 (/home/spai0308/data) 기준으로 HTML 변환본 폴더와 출력 폴더 지정
DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"  # HTML 문서별 폴더들
INTERIM = DATA / "interim"                       # 중간 산출물 저장소

# 최종 OCR 결과가 들어갈 JSONL 파일 경로
OUT_JSONL = INTERIM / "assets_html_ocr.jsonl"

# HTML과 XML이 섞여도 재사용할 수 있게 '전역 캐시' 파일 하나만 사용
# (같은 이미지가 HTML, XML 양쪽에 있더라도 한 번만 처리)
CACHE_JSON = INTERIM / "assets_openai_ocr.global.cache.json"

# ========================= 시간대/타임스탬프 =========================
# 한국 시간(KST)으로 타임스탬프를 찍습니다. (나중에 디버깅/정렬이 쉬움)
KST = timezone(timedelta(hours=9))
def now_iso(): 
    """현재 시간을 KST로 ISO8601 문자열로 반환 (예: 2025-09-19T14:03:12+09:00)"""
    return datetime.now(KST).isoformat()

# ========================= OpenAI 클라이언트 =========================
# 환경변수 OPENAI_API_KEY가 설정되어 있어야 정상 작동합니다.
client = OpenAI()

# ========================= OCR 동작 설정(필요 시만 변경) =========================
# 모델 우선순위(가벼운 → 무거운). 1차 패스는 비용/속도 절약을 위해 주로 nano/mini만 사용 권장
MODEL_LADDER = [
    "gpt-4.1-nano",  # 가장 빠르고 저렴 (기본)
    "gpt-4.1-mini",  # 조금 더 정확
    "gpt-4.1",       # 필요시만 승급
    "gpt-4o",        # 필요시만 승급
]

MAX_ESCALATION_STEPS = 2   # 품질이 낮으면 상위 모델로 '승급'하는 단계 수. 1차는 0(승급 OFF) 추천
DOC_UPGRADE_BUDGET   = 3   # 문서(doc_id)당 상향 허용 이미지 수 (승급 켰을 때만 의미)
MAX_REQ_PER_MIN      = 20  # 분당 요청 수(간단한 레이트리밋). 너무 높이면 API가 제한 걸 수 있음

# 품질 체크 기준: 너무 짧거나(20자 미만), 문자비(영/한글 비율)가 너무 낮으면 "부실"로 판단
MIN_CHARS       = 20        # 최소 글자 수
MIN_ALPHA_RATIO = 0.30      # 알파벳/한글 비율 임계값
HANGUL_BONUS    = 0.10      # 한글이 조금이라도 포함되면 가산점

# 처리 대상 이미지 확장자(대소문자 무관)
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

# ========================= 유틸리티 함수들 =========================
def file_md5(p: Path) -> str:
    """
    파일 내용을 기준으로 md5 해시 생성.
    - 해시가 같으면 '완전히 같은 이미지'로 간주 → 캐시에 기록해 중복 요청 방지
    """
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):  # 1MB 단위로 읽기
            h.update(chunk)
    return h.hexdigest()

def extract_doc_id(img_path: Path) -> str:
    """
    경로에서 문서 폴더 이름(doc_id) 뽑기.
    - 예: /.../converted/html/<doc_id>/assets/파일
    - 위 구조에서 <doc_id> 추출
    """
    parts = img_path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i + 3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
    return "unknown"

def compress_image(pil_img: Image.Image, max_side=1024, fmt="JPEG", quality=75) -> bytes:
    """
    이미지 용량 줄이기(속도/비용 절약):
    - 긴 변 기준 1024px로 리사이즈 + JPEG 품질 75로 압축
    - OCR에는 충분한 품질이며 전송/처리 시간 크게 절약
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w,h) > max_side:
        s = max_side / float(max(w,h))
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(b: bytes, mime="image/jpeg") -> str:
    """
    OpenAI에 이미지를 'data URL'로 인라인 전송:
    - "data:<MIME>;base64,<인코딩된데이터>" 형태
    - 쉼표 뒤 공백 있으면 깨질 수 있어, 공백 없이 만듭니다.
    """
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def text_quality_is_poor(text: str) -> bool:
    """
    OCR 결과가 '부실'한지 간단 체크:
    - 20자 미만이면 부실
    - 알파벳/한글 비율이 너무 낮으면 부실 (한글 포함되면 보너스)
    """
    t = (text or "").strip()
    if len(t) < MIN_CHARS:
        return True
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha / total if total else 0.0) + (HANGUL_BONUS if hangul > 0 else 0.0)
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win: list):
    """
    간단한 분당 요청수 제한(RPM) 제어:
    - 최근 60초 내 요청 기록을 보고, 너무 많으면 잠깐 대기
    """
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))  # 가장 오래된 요청과의 차만큼 대기
    win.append(time.time())

def call_vision(model: str, pil_img: Image.Image, timeout=75, max_retries=3) -> str:
    """
    OpenAI 비전 모델 호출(안정성 강화):
    - 이미지 압축 → data URL로 전송
    - 타임아웃/재시도(지수백오프)로 일시적 지연/오류에 견고
    """
    img_bytes = compress_image(pil_img, max_side=768, fmt="JPEG", quality=75)
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
                timeout=timeout
            )
            # 모델 응답에서 텍스트 본문만 꺼내기
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # 에러가 나면 잠깐 기다렸다가 재시도 (최대 max_retries번)
            if attempt >= max_retries:
                raise
            sleep = min(2**attempt + random.uniform(0,1), 12)
            tqdm.write(f"[retry] {type(e).__name__}: {e} → {sleep:.1f}s 대기 (시도 {attempt+1}/{max_retries})")
            time.sleep(sleep)
    return ""

def ocr_with_escalation(pil_img: Image.Image, doc_id: str, doc_budget_used: dict):
    """
    '모델 승급' 로직:
    - 1) 가장 가벼운 모델로 먼저 시도
    - 2) 결과가 '부실'하면 상위 모델로 승급 (단, MAX_ESCALATION_STEPS와 DOC_UPGRADE_BUDGET 제한)
    - 1차 패스는 MAX_ESCALATION_STEPS=0으로 꺼두는 걸 권장 (비용/속도 절약)
    """
    used_model = MODEL_LADDER[0]
    text = call_vision(used_model, pil_img)
    steps = 0
    while text_quality_is_poor(text) and steps < MAX_ESCALATION_STEPS:
        # 문서별 승급 예산 초과 시 중단
        if doc_budget_used.get(doc_id, 0) >= DOC_UPGRADE_BUDGET:
            break
        steps += 1
        next_idx = min(steps, len(MODEL_LADDER)-1)
        used_model = MODEL_LADDER[next_idx]
        text = call_vision(used_model, pil_img)
        doc_budget_used[doc_id] = doc_budget_used.get(doc_id, 0) + 1
    return text, used_model, steps

def iter_html_assets():
    """
    HTML 변환본 폴더 아래 모든 문서를 훑어
    - .../html/<doc_id>/assets/* 경로의 이미지들을 찾아서
    - (이미지경로, 멀티프레임 여부, PIL 이미지 객체)를 하나씩 yield 합니다.
    """
    pattern = str(HTML_ROOT / "**" / "assets" / "*")
    for p in glob.glob(pattern, recursive=True):
        path = Path(p)
        if path.suffix.lower() not in VALID_EXT:
            continue
        try:
            pil = Image.open(str(path))
        except UnidentifiedImageError:
            # 이미지가 아니거나 손상된 파일은 건너뜀
            continue
        yield path, getattr(pil, "is_animated", False), pil

# ========================= 실행(노트북 한 셀에서 그대로) =========================
# 출력 폴더 만들기(없으면 생성)
INTERIM.mkdir(parents=True, exist_ok=True)
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# 캐시 로드: 이전에 처리했던 이미지 md5 목록
cache = {}
if CACHE_JSON.exists():
    try:
        cache = json.load(open(CACHE_JSON, "r", encoding="utf-8"))
    except Exception:
        cache = {}

rpm_win = []        # 분당 요청 제한 창
doc_budget_used = {}# 문서별 승급 예산 사용량
total = 0           # 이번 실행에서 새로 처리한 이미지 수

with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
    # iter_html_assets() 로 모든 이미지 순회
    for img_path, is_multi, pil in tqdm(iter_html_assets(), desc="OpenAI OCR (html)"):
        try:            
            md5 = file_md5(img_path)
            if md5 in cache:
                # 같은 이미지(내용 동일)는 이미 처리함 → 건너뜀
                continue

            # 문서 식별자(폴더 이름) 뽑기
            doc_id = extract_doc_id(img_path)
            # 이미지 처리 중 어떤 파일에서 지연발생하는지 확인
            tqdm.write(f"→ {extract_doc_id(img_path)} / {img_path.name}")

            if is_multi:
                # GIF 같은 멀티프레임 이미지 처리
                for frame_index, frame in enumerate(ImageSequence.Iterator(pil)):
                    ratelimit(rpm_win)  # RPM 제어
                    text, used_model, steps = ocr_with_escalation(frame.convert("RGB"), doc_id, doc_budget_used)
                    rec = {
                        "doc_id": doc_id,
                        "source_path": str(img_path),
                        "frame_index": int(frame_index),
                        "text": text,
                        "avg_conf": -1.0,  # 평균 신뢰도(미사용) 자리
                        "lang": "ko+en",
                        "preprocess": {"resize_max_side":1024,"format":"jpeg","quality":75},
                        "ts": now_iso(),
                        "source_type": "asset_ocr_html",  # HTML 자산 OCR이라는 표시
                        "provider": "openai",
                        "model": used_model,
                        "escalation_steps": steps
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                # 일반 단일 이미지 처리
                ratelimit(rpm_win)
                text, used_model, steps = ocr_with_escalation(pil.convert("RGB"), doc_id, doc_budget_used)
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(img_path),
                    "frame_index": 0,
                    "text": text,
                    "avg_conf": -1.0,
                    "lang": "ko+en",
                    "preprocess": {"resize_max_side":1024,"format":"jpeg","quality":75},
                    "ts": now_iso(),
                    "source_type": "asset_ocr_html",
                    "provider": "openai",
                    "model": used_model,
                    "escalation_steps": steps
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 처리 완료한 이미지는 캐시에 등록
            cache[md5] = True
            total += 1

            # 100장마다 캐시를 파일에 저장(중간 저장: 중단돼도 재개 쉬움)
            if total % 100 == 0:
                with open(CACHE_JSON, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)

        except Exception as e:
            # 실패도 한 줄로 남겨두면 나중에 원인 파악이 쉬움
            err = {
                "doc_id": "unknown",
                "source_path": str(img_path),
                "error": repr(e),
                "ts": now_iso(),
                "source_type": "asset_ocr_html_error",
                "provider": "openai",
            }
            out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
            out_f.flush()   # 줄마다 오류 디스크에 바로 기록

# 마지막에 캐시 저장(중간 저장 못 했더라도 최종 반영)
with open(CACHE_JSON, "w", encoding="utf-8") as cf:
    json.dump(cache, cf, ensure_ascii=False, indent=2)

print(f"[OK] OpenAI OCR complete (html) → {OUT_JSONL} (이번에 새로 처리한 이미지: {total}장)")


# **[문제 발생]**
# - hd8.png 같은 빡빡한 다이어그램은 한장 통째 ocr 시 타임아웃이 잦음
# - 예외로 떨어지고(에러 레코드만 기록) -> total 성공 건만 세서 0장 됨
# 
# 
# ------------------------------------------------------
# 
# 
# 
# **[ 해결 방안]**
# - 자동 타일링 폴백
#     - 통짜 ocr이 타임아웃-에러 나면 -> 즉시 이미지 격자(타일)로 잘라서 타일별로 ocr --> 합쳐서 기록 
# - 성공 처리로 간주
#     - 타일링 경괄르 성공 레코드로 저장 + 캐시에 체크해서 재실행 시 재시도 안하게 
# - 로그/플러시 강화
#     - 어떤 파일에 폴백이 발동했는지 보이고, 줄마다 flush()해서 진행사황이 즉시 파일에 남도록 기록

# In[20]:


# -*- coding: utf-8 -*-
"""
HTML assets 전체 OCR:
- 통짜 OCR 실패(타임아웃/에러) 시 자동으로 '타일 OCR' 폴백 → 결과 합쳐 기록
- 성공 시 캐시에 체크하여 재실행 시 스킵
출력:
  /home/spai0308/data/interim/assets_html_ocr.jsonl
캐시:
  /home/spai0308/data/interim/assets_openai_ocr.global.cache.json
"""

import os, io, json, glob, base64, time, random, math, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageSequence, UnidentifiedImageError, ImageEnhance, ImageFilter
from tqdm import tqdm
from openai import OpenAI, APITimeoutError, APIError, RateLimitError

# ----- 경로 -----
DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"
INTERIM = DATA / "interim"
OUT_JSONL = INTERIM / "assets_html_ocr.jsonl"
CACHE_JSON = INTERIM / "assets_openai_ocr.global.cache.json"

# ----- 시간 -----
KST = timezone(timedelta(hours=9))
now_iso = lambda: datetime.now(KST).isoformat()

# ----- OpenAI -----
client = OpenAI()  # OPENAI_API_KEY 필요

# ----- 설정 -----
MODEL_PRIMARY = "gpt-4.1-nano"   # 기본
MODEL_TIMEOUT_UP = "gpt-4.1-mini" # 타임아웃 회피용 1회 업그레이드
USE_QUALITY_ESCALATION = False    # 1차 패스 비용 절약
MAX_REQ_PER_MIN = 12              # RPM 완화
MIN_CHARS, MIN_ALPHA_RATIO, HANGUL_BONUS = 20, 0.30, 0.10
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

# 타일링 파라미터(폴백용)
TARGET_TILE_LONG_SIDE = 900
TILE_OVERLAP = 0.12
UPSCALE_BEFORE_OCR = 1.2
JPEG_QUALITY = 74
MAX_SEND_SIDE = 1024

# ----- 유틸 -----
def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_doc_id(img_path: Path) -> str:
    parts = img_path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
    return "unknown"

def compress_image(pil_img: Image.Image, max_side=MAX_SEND_SIDE, quality=JPEG_QUALITY) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w,h) > max_side:
        s = max_side / float(max(w,h))
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(b, mime="image/jpeg"): 
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def text_quality_is_poor(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CHARS: return True
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha/total if total else 0.0) + (HANGUL_BONUS if hangul>0 else 0.0)
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())

def enhance_for_text(pil):
    img = pil.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.12)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    return img

# ----- OpenAI 호출 -----
def call_once(model: str, pil_img: Image.Image, timeout=75) -> str:
    # 얇은 글자 가독성 위해 살짝 업스케일
    if UPSCALE_BEFORE_OCR and UPSCALE_BEFORE_OCR != 1.0:
        w, h = pil_img.size
        pil_img = pil_img.resize((int(w*UPSCALE_BEFORE_OCR), int(h*UPSCALE_BEFORE_OCR)), Image.LANCZOS)
    data_url = to_data_url(compress_image(pil_img))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
            {"role":"user","content":[
                {"type":"text","text":"Extract all text from the image/tile. If it's a diagram, read labels left-to-right, top-to-bottom. Return text only."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ],
        temperature=0.0,
        timeout=timeout
    )
    return (resp.choices[0].message.content or "").strip()

def call_with_timeout_upgrade(pil_img: Image.Image, timeout=75, max_retries=3):
    model, upgraded = MODEL_PRIMARY, False
    for attempt in range(max_retries+1):
        try:
            return call_once(model, pil_img, timeout=timeout)
        except (APITimeoutError, RateLimitError, APIError) as e:
            if isinstance(e, APITimeoutError) and not upgraded and model == MODEL_PRIMARY:
                tqdm.write("[timeout] temporary model upgrade: nano → mini")
                model, upgraded = MODEL_TIMEOUT_UP, True
            sleep = min(2**attempt + random.uniform(0,1), 15)
            tqdm.write(f"[retry] {type(e).__name__}: {e} → {sleep:.1f}s wait")
            time.sleep(sleep)
        except Exception as e:
            if attempt >= max_retries: raise
            sleep = min(2**attempt + random.uniform(0,1), 10)
            tqdm.write(f"[retry] {type(e).__name__}: {e} → {sleep:.1f}s wait")
            time.sleep(sleep)
    return ""

# ----- 타일링 -----
def make_tiles(img: Image.Image, target_long=TARGET_TILE_LONG_SIDE, overlap=TILE_OVERLAP):
    W, H = img.size
    long_side = max(W, H)
    tiles_on_long = max(1, math.ceil(long_side / float(target_long)))
    if W >= H:
        cols = tiles_on_long
        rows = max(1, round(H / (W/cols)))
    else:
        rows = tiles_on_long
        cols = max(1, round(W / (H/rows)))
    tile_w = math.ceil(W / cols)
    tile_h = math.ceil(H / rows)
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))
    bboxes, y = [], 0
    while True:
        x = 0
        bottom = min(y + tile_h, H)
        top = max(0, bottom - tile_h)
        row = []
        while True:
            right = min(x + tile_w, W)
            left = max(0, right - tile_w)
            row.append((left, top, right, bottom))
            if right >= W: break
            x += step_x
        bboxes.append(row)
        if bottom >= H: break
        y += step_y
    return bboxes

def tiled_ocr_fallback(orig: Image.Image, doc_id: str, img_path: Path, out_f, rpm_win: list) -> bool:
    """타일 OCR 실행 후 합친 레코드 1건 저장. 성공 시 True 반환."""
    bboxes = make_tiles(orig)
    texts = []
    tile_count = 0
    for r, row in enumerate(bboxes):
        for c, (x0,y0,x1,y1) in enumerate(row):
            tile = enhance_for_text(orig.crop((x0,y0,x1,y1)))
            ratelimit(rpm_win)
            t = call_with_timeout_upgrade(tile)  # 타임아웃 회피 로직 포함
            texts.append(t.strip())
            tile_count += 1
            # (선택) 타일별 상세 레코드도 남기고 싶으면 아래 주석 해제
            # rec_tile = {"doc_id":doc_id,"source_path":str(img_path),"tile_rc":[r,c],"bbox":[x0,y0,x1,y1],
            #            "text":t,"ts":now_iso(),"source_type":"asset_ocr_html_tiled","provider":"openai","model":"auto"}
            # out_f.write(json.dumps(rec_tile, ensure_ascii=False)+"\n"); out_f.flush()
    combined = "\n".join([t for t in texts if t])
    rec = {
        "doc_id": doc_id,
        "source_path": str(img_path),
        "frame_index": 0,
        "text": combined,
        "avg_conf": -1.0,
        "lang": "ko+en",
        "preprocess": {"mode":"tiled","tile_long":TARGET_TILE_LONG_SIDE,"overlap":TILE_OVERLAP,
                       "upscale":UPSCALE_BEFORE_OCR,"send_side":MAX_SEND_SIDE,"jpeg_q":JPEG_QUALITY},
        "ts": now_iso(),
        "source_type": "asset_ocr_html_tiled_agg",
        "provider": "openai",
        "model": "auto(nano→mini on timeout)",
        "tile_count": tile_count
    }
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
    # 텍스트가 거의 없으면 실패 간주
    return len(combined.strip()) >= MIN_CHARS

# ----- 자산 이미지 이터레이터 -----
def iter_html_assets():
    pattern = str(HTML_ROOT / "**" / "assets" / "*")
    for p in glob.glob(pattern, recursive=True):
        path = Path(p)
        if path.suffix.lower() not in VALID_EXT:
            continue
        try:
            pil = Image.open(str(path))
        except UnidentifiedImageError:
            continue
        yield path, getattr(pil, "is_animated", False), pil

# ================= 실행 =================
INTERIM.mkdir(parents=True, exist_ok=True)
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# 캐시 로드
cache = {}
if CACHE_JSON.exists():
    try:
        cache = json.load(open(CACHE_JSON, "r", encoding="utf-8"))
    except Exception:
        cache = {}

rpm_win, total = [], 0

with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
    for img_path, is_multi, pil in tqdm(iter_html_assets(), desc="OpenAI OCR (html)"):
        md5 = file_md5(img_path)
        if md5 in cache:
            continue

        doc_id = extract_doc_id(img_path)
        tqdm.write(f"→ {doc_id} / {img_path.name}")

        try:
            # 1) 통짜 OCR 먼저 시도
            if is_multi:
                texts=[]
                for frame in ImageSequence.Iterator(pil):
                    ratelimit(rpm_win)
                    t = call_with_timeout_upgrade(frame.convert("RGB"))
                    texts.append(t)
                text = "\n".join(texts)
            else:
                ratelimit(rpm_win)
                text = call_with_timeout_upgrade(pil.convert("RGB"))

            # (선택) 품질 승급 — 1차 패스에선 보통 끔
            if USE_QUALITY_ESCALATION and text_quality_is_poor(text):
                ratelimit(rpm_win)
                text = call_once(MODEL_TIMEOUT_UP, pil.convert("RGB"))

            # 성공 저장
            rec = {
                "doc_id": doc_id, "source_path": str(img_path), "frame_index": 0,
                "text": text, "avg_conf": -1.0, "lang":"ko+en",
                "preprocess":{"mode":"single","send_side":MAX_SEND_SIDE,"jpeg_q":JPEG_QUALITY},
                "ts": now_iso(), "source_type":"asset_ocr_html", "provider":"openai", "model":"auto(nano→mini on timeout)"
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
            cache[md5] = True; total += 1

        except Exception as e:
            # 2) 통짜 실패(타임아웃 등) → 타일링 폴백
            tqdm.write(f"[fallback] tiled OCR for {img_path.name} because: {type(e).__name__} {e}")
            ok = tiled_ocr_fallback(pil.convert("RGB"), doc_id, img_path, out_f, rpm_win)
            if ok:
                cache[md5] = True; total += 1
            else:
                # 폴백도 부실하면 에러 레코드 남김
                err = {"doc_id":doc_id,"source_path":str(img_path),"error":repr(e),
                       "ts":now_iso(),"source_type":"asset_ocr_html_error","provider":"openai"}
                out_f.write(json.dumps(err, ensure_ascii=False)+"\n"); out_f.flush()

# 캐시 저장
with open(CACHE_JSON, "w", encoding="utf-8") as cf:
    json.dump(cache, cf, ensure_ascii=False, indent=2)

print(f"[OK] HTML OCR complete → {OUT_JSONL} (newly processed: {total})")


# [확인 결과]
# 
# 
# - 계속 ocr 되지 않았던 hd8.png 해결
# - 나머지 이미지는 md5 때문에 스킵되어서 다시 ocr 하지 않음
# - 마지막줄 ``OpenAI OCR (html) : 174it`` 의미 --> 총 174장을 훑고, 그 중 1장만 새로 기록함

# ##### 빠른 확인

# In[ ]:


get_ipython().system('wc -l /home/spai0308/data/interim/assets_html_ocr.jsonl')
print("="*50)

get_ipython().system('head -n 1 /home/spai0308/data/interim/assets_html_ocr.jsonl | jq .')


# In[ ]:


# hd8 결과 확인(마지막 한 줄 보기)
get_ipython().system('grep -F "hd8.png" /home/spai0308/data/interim/assets_html_ocr.jsonl | tail -n 1 | jq .')
print("="*50)


# 어떤 타입으로 저장됐는지 개수 확인
get_ipython().system('grep -c \'"source_type":"asset_ocr_html"\' /home/spai0308/data/interim/assets_html_ocr.jsonl')
print("="*50)

get_ipython().system('grep -c \'"source_type":"asset_ocr_html_tiled_agg"\' /home/spai0308/data/interim/assets_html_ocr.jsonl')
print("="*50)

get_ipython().system('grep -c \'"source_type":"asset_ocr_html_error"\' /home/spai0308/data/interim/assets_html_ocr.jsonl')


# - assets_html_ocr.jsonl에 html의 모든 이미지가 전부 다 기록되었는지 확인

# In[ ]:


from pathlib import Path
import json, hashlib, os
from collections import Counter, defaultdict

HTML_ROOT = Path("/home/spai0308/data/raw/converted/html")
OUT_JSONL = Path("/home/spai0308/data/interim/assets_html_ocr.jsonl")
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# 1) 디스크 이미지 목록 수집
img_paths = [p for p in HTML_ROOT.rglob("*") if (p.is_file() and p.suffix.lower() in VALID_EXT)]
img_paths_set = set(str(p) for p in img_paths)

# 2) JSONL 로드
jsonl_records = []
if OUT_JSONL.exists():
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                jsonl_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

# 3) JSONL에서 유효한 source_path와 타입 집계
jsonl_paths = [r.get("source_path") for r in jsonl_records if r.get("source_path")]
jsonl_paths_set = set(jsonl_paths)
source_types = Counter(r.get("source_type","") for r in jsonl_records)

# 4) 누락/초과 계산
missing_on_jsonl = sorted(img_paths_set - jsonl_paths_set)   # 디스크엔 있는데 JSONL엔 없음
orphan_in_jsonl  = sorted(jsonl_paths_set - img_paths_set)   # JSONL엔 있는데 디스크엔 없음(경로 변동/삭제)

# 5) 중복 이미지(md5 기준) 진단
#    - 동일 md5가 여러 경로에 존재 → 현재 캐시 설계(md5 단일키)면 뒤 경로들은 스킵되어 JSONL에 기록이 안될 수 있음
md5_to_paths = defaultdict(list)
for p in img_paths:
    try:
        md5 = file_md5(p)
        md5_to_paths[md5].append(str(p))
    except Exception:
        pass

duplicate_md5 = {md5:paths for md5,paths in md5_to_paths.items() if len(paths)>1}

print("=== SUMMARY ===")
print(f" 디스크 이미지 수: {len(img_paths_set)}")
print(f" JSONL 기록(고유 source_path) 수: {len(jsonl_paths_set)}")
print(" 타입별 개수:", dict(source_types))
print(f" JSONL에 누락된 경로 수: {len(missing_on_jsonl)}")
print(f" 디스크엔 없는데 JSONL에만 있는 경로 수: {len(orphan_in_jsonl)}")
print(f" md5 중복 그룹 수: {len(duplicate_md5)}  (같은 그림이 여러 경로에 복제된 경우)")

# 샘플 보여주기
print("\n--- JSONL에 누락된 이미지 예시(최대 10) ---")
for p in missing_on_jsonl[:10]:
    print("  -", p)

print("\n--- md5 중복 예시(최대 5 그룹) ---")
for i, (md5, paths) in enumerate(duplicate_md5.items()):
    print(f"  * md5={md5} (경로 {len(paths)}개)")
    for sp in paths[:3]:
        print("     -", sp)
    if i>=4: break


# **[결과]**
# 
# 
# 
# - 처리는 대부분 캐시에 남고, jsonl엔 제대로 안 적힘
# - 원인 & 바로잡는 순서
#     - 1) 백필로 jsonl 채우기
#       2) 메인 스크립트 패이
#       3) 재검증

# In[ ]:


# 백필(backfill) - 캐시에서 텍스트 꺼내 jsonl에 "경로별 레코드" 생성


# --- backfill_from_cache.py ---
from pathlib import Path
import json, hashlib

HTML_ROOT = Path("/home/spai0308/data/raw/converted/html")     # 원본 이미지 루트
OUT_JSONL = Path("/home/spai0308/data/interim/assets_html_ocr.jsonl")  # JSONL 결과 파일
CACHE_JSON = Path("/home/spai0308/data/interim/assets_openai_ocr.global.cache.json")  # 전역 캐시
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()
# (설명) 이미지 파일 md5 계산

def extract_doc_id(img_path: Path) -> str:
    parts = img_path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
    return "unknown"
# (설명) /converted/html/<doc_id>/assets/<file> 에서 doc_id 추출

def should_ignore(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts:
        return True
    if any(part.startswith(".") for part in p.parts):
        return True
    return False
# (설명) 체크포인트/숨김 경로는 스캔 제외

# 1) 디스크 이미지 수집(불필요 경로 제외)
img_paths = [p for p in HTML_ROOT.rglob("*")
             if p.is_file() and p.suffix.lower() in VALID_EXT and not should_ignore(p)]
img_paths_set = set(str(p) for p in img_paths)
# (설명) 실제 OCR 대상이 되는 이미지 경로만 모음

# 2) 기존 JSONL의 source_path 집합
jsonl_paths_set = set()
if OUT_JSONL.exists():
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                rec = json.loads(line)
                sp = rec.get("source_path")
                if sp: jsonl_paths_set.add(sp)
            except json.JSONDecodeError:
                pass
# (설명) 이미 기록된 경로는 중복으로 만들지 않기 위해 수집

# 3) 캐시 로드
cache = {}
if CACHE_JSON.exists():
    try:
        cache = json.load(open(CACHE_JSON, encoding="utf-8"))
    except Exception:
        cache = {}
# (설명) 기존 OCR 결과(텍스트 등)가 들어있는 전역 캐시

# 4) 누락 대상 중, 캐시에 텍스트가 있는 것만 백필
added = 0
with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
    for p in img_paths:
        sp = str(p)
        if sp in jsonl_paths_set:
            continue  # 이미 JSONL에 있으면 skip
        md5 = file_md5(p)
        c = cache.get(md5)
        if isinstance(c, dict) and c.get("text"):  # 캐시에 텍스트가 있어야 백필 가능
            doc_id = extract_doc_id(p)
            rec = {
                "doc_id": doc_id,
                "source_path": sp,
                "frame_index": 0,
                "text": c["text"],
                "avg_conf": -1.0,
                "lang": c.get("lang","ko+en"),
                "preprocess": c.get("preprocess", {"mode":"single"}),
                "ts": c.get("ts",""),
                "source_type": "asset_ocr_html_from_cache",
                "provider": "openai",
                "model": c.get("model","cache")
            }
            out_f.write(json.dumps(rec, ensure_ascii=False)+"\n"); out_f.flush()
            added += 1

print(f"[backfill] JSONL에 새로 추가된 from_cache 레코드: {added}")
# (설명) 캐시 보유 텍스트로만 안전하게 채움. 이후 진짜 미처리본은 메인 스크립트로 처리.




# [문제 발생]
# - ``from_cache 0``은 캐시에 텍스트가 없어서 백필 조건에 걸린 게 하나도 없었다는 뜻
# → 다음 스텝은 **캐시 스키마 업그레이드 + 기존 불완전 캐시는 비워내고 재실행**이 필요
# 

# In[35]:


# 0-1) 현재 결과/캐시 백업
get_ipython().system('cp /home/spai0308/data/interim/assets_html_ocr.jsonl /home/spai0308/data/interim/assets_html_ocr.jsonl.bak.$(date +%Y%m%d-%H%M%S)')
get_ipython().system('cp /home/spai0308/data/interim/assets_openai_ocr.global.cache.json /home/spai0308/data/interim/assets_openai_ocr.global.cache.json.bak.$(date +%Y%m%d-%H%M%S)')


# In[ ]:


# 확인
get_ipython().system('ls /home/spai0308/data/interim/')
get_ipython().system('ls /home/spai0308/data/interim/')


# In[ ]:


# 현재 캐시 상태 빠르게 점검
from pathlib import Path
import json

CACHE = Path("/home/spai0308/data/interim/assets_openai_ocr.global.cache.json")

data = json.load(open(CACHE, encoding="utf-8"))
bool_count = sum(1 for v in data.values() if isinstance(v, bool))    # (주석) True/False 형태(구캐시)
dict_count = sum(1 for v in data.values() if isinstance(v, dict))    # (주석) 새 스키마(dict) 형태
with_text = sum(1 for v in data.values() if isinstance(v, dict) and v.get("text"))
print("총 엔트리:", len(data))
print("불리언 캐시:", bool_count)
print("딕셔너리 캐시:", dict_count)
print("  └ 텍스트 포함:", with_text)


# **[문제 확인 및 해결 방법]**
# 
# - 총 엔트리 26, 전부 불리언(True/False), 텍스트 포함 0 →
# - 백필이 안되는 이유
# > 캐시에 OCR 텍스트가 저장돼 있지 않아서(from_cache로 옮길 재료가 없음).

# In[ ]:


# 불리언 엔트리 전부 제거 (캐시정리)

# prune_cache_remove_booleans.py
from pathlib import Path, PurePath
import json

CACHE_JSON = Path("/home/spai0308/data/interim/assets_openai_ocr.global.cache.json")

cache = {}
if CACHE_JSON.exists():
    cache = json.load(open(CACHE_JSON, encoding="utf-8"))

removed = 0
for k in list(cache.keys()):
    if isinstance(cache[k], bool):   # 불리언(구캐시) → 삭제
        cache.pop(k); removed += 1

json.dump(cache, open(CACHE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[prune] 불리언 엔트리 삭제: {removed}건")


# **[메인: ocr 스크립치 핵심 패치]**
# 
# - 숨김/체크포인트 폴더 제외
# - 성공 시 캐시에 텍스트 저장(dict 스키마)
# - 같은 md5 잽라견 시 ocr 생략 + jsonl에 ``from_cache`` 레코드 기록
# - 부실컷이면 타일링 풀백 -> 여전히 부실하면 상위모델 1회 재시도
# - 마지막 요약 리포트 출력

# In[ ]:


# 초기화 셀: 기존 결과/캐시 백업한 뒤 비우기

from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil, json, os

KST = timezone(timedelta(hours=9))
ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")

DATA = Path("/home/spai0308/data")
INTERIM = DATA / "interim"
OUT_JSONL = INTERIM / "assets_html_ocr.jsonl"
CACHE_JSON = INTERIM / "assets_openai_ocr.global.cache.json"

INTERIM.mkdir(parents=True, exist_ok=True)

# 1) 백업
if OUT_JSONL.exists():
    shutil.copy2(OUT_JSONL, OUT_JSONL.with_suffix(OUT_JSONL.suffix + f".bak.{ts}"))
if CACHE_JSON.exists():
    shutil.copy2(CACHE_JSON, CACHE_JSON.with_suffix(CACHE_JSON.suffix + f".bak.{ts}"))

# 2) 비우기(존재하면 0바이트로, 없으면 생성)
OUT_JSONL.write_text("", encoding="utf-8")
CACHE_JSON.write_text("{}", encoding="utf-8")

print("[reset] backed up & cleared:")
print(" -", OUT_JSONL)
print(" -", CACHE_JSON)


# **[새 파이프라인]** --> 완벽버전
# - 모델 제한 준수: nano/mini 기본
# - 선택적 gpt-4.1/gpt-4.o만 테스트용

# In[ ]:


# -*- coding: utf-8 -*-
"""
HTML assets OCR (모델 제한 준수 버전)
- 기본 모델: gpt-4.1-nano / 타임아웃/품질 보강: gpt-4.1-mini
- (옵션) 테스트가 필요할 때만 제한적으로 gpt-4.1, gpt-4o 사용 가능 (기본 False)
- 숨김/.ipynb_checkpoints 제외
- 캐시(dict)로 텍스트 저장 → 같은 md5 재발견 시 OCR 생략 + JSONL에 from_cache 기록
- '부실컷만' 타일링 폴백, 여전히 부실하면 (기본) mini 재시도 / (옵션) gpt-4.1→gpt-4o 순 테스트
"""

import os, io, json, glob, base64, time, random, math, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from PIL import Image, ImageSequence, UnidentifiedImageError, ImageEnhance, ImageFilter
from tqdm import tqdm
from openai import OpenAI, APITimeoutError, APIError, RateLimitError

# ====== 경로 / 환경 ======
DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"
INTERIM = DATA / "interim"
OUT_JSONL = INTERIM / "assets_html_ocr.jsonl"
CACHE_JSON = INTERIM / "assets_openai_ocr.global.cache.json"

KST = timezone(timedelta(hours=9))
now_iso = lambda: datetime.now(KST).isoformat(timespec="seconds")

client = OpenAI()  # OPENAI_API_KEY 필요

# ====== 모델 / 정책 ======
MODEL_PRIMARY = "gpt-4.1-nano"      # 1차
MODEL_TIMEOUT_UP = "gpt-4.1-mini"    # 타임아웃/품질 보강
# 테스트가 "정말" 필요할 때만 아래를 True로 (기본 False)
USE_TEST_MODELS = False
TEST_MODEL_ORDER = ["gpt-4.1", "gpt-4o"]  # 이 순서로 1회씩만 시도 (USE_TEST_MODELS=True일 때만)

MAX_REQ_PER_MIN = 12
VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

# 품질 판정(부실컷)
MIN_CHARS = 20
MIN_ALPHA_RATIO = 0.30
HANGUL_BONUS = 0.10

# 타일링 파라미터
TARGET_TILE_LONG_SIDE = 900
TILE_OVERLAP = 0.12
UPSCALE_BEFORE_OCR = 1.2
JPEG_QUALITY = 74
MAX_SEND_SIDE = 1024

# ====== 유틸 ======
def should_ignore(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts:
        return True
    if any(part.startswith(".") for part in p.parts):
        return True
    return False

def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_doc_id(img_path: Path) -> str:
    parts = img_path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
    return "unknown"

def compress_image(pil_img: Image.Image, max_side=MAX_SEND_SIDE, quality=JPEG_QUALITY) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w,h) > max_side:
        s = max_side / float(max(w,h))
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(b, mime="image/jpeg"):
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def enhance_for_text(pil: Image.Image) -> Image.Image:
    img = pil.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.12)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    return img

def text_quality_is_poor(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CHARS: 
        return True
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha/total if total else 0.0) + (HANGUL_BONUS if hangul>0 else 0.0)
    return ratio < MIN_ALPHA_RATIO

def ratelimit(win:list):
    now = time.time()
    win[:] = [t for t in win if now - t < 60.0]
    if len(win) >= MAX_REQ_PER_MIN:
        time.sleep(60.0 - (now - win[0]))
    win.append(time.time())

# ====== OpenAI 호출 ======
def call_once(model: str, pil_img: Image.Image, timeout=75) -> str:
    if UPSCALE_BEFORE_OCR and UPSCALE_BEFORE_OCR != 1.0:
        w, h = pil_img.size
        pil_img = pil_img.resize((int(w*UPSCALE_BEFORE_OCR), int(h*UPSCALE_BEFORE_OCR)), Image.LANCZOS)
    data_url = to_data_url(compress_image(pil_img))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an OCR assistant for Korean & English. Extract only the text, preserving meaningful line breaks."},
            {"role":"user","content":[
                {"type":"text","text":"Extract all text from the image/tile. If it's a diagram, read labels left-to-right, top-to-bottom. Return text only."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ],
        temperature=0.0,
        timeout=timeout
    )
    return (resp.choices[0].message.content or "").strip()

def call_with_timeout_upgrade(pil_img: Image.Image, timeout=75, max_retries=3):
    model, upgraded = MODEL_PRIMARY, False
    for attempt in range(max_retries+1):
        try:
            return call_once(model, pil_img, timeout=timeout)
        except (APITimeoutError, RateLimitError, APIError) as e:
            if isinstance(e, APITimeoutError) and not upgraded and model == MODEL_PRIMARY:
                tqdm.write("[timeout] temporary model upgrade: nano → mini")
                model, upgraded = MODEL_TIMEOUT_UP, True
            sleep = min(2**attempt + random.uniform(0,1), 15)
            tqdm.write(f"[retry] {type(e).__name__}: {e} → {sleep:.1f}s wait")
            time.sleep(sleep)
        except Exception as e:
            if attempt >= max_retries:
                raise
            sleep = min(2**attempt + random.uniform(0,1), 10)
            tqdm.write(f"[retry] {type(e).__name__}: {e} → {sleep:.1f}s wait")
            time.sleep(sleep)
    return ""

# ====== 타일링 ======
def make_tiles(img: Image.Image, target_long=TARGET_TILE_LONG_SIDE, overlap=TILE_OVERLAP):
    W, H = img.size
    long_side = max(W, H)
    tiles_on_long = max(1, math.ceil(long_side / float(target_long)))
    if W >= H:
        cols = tiles_on_long
        rows = max(1, round(H / (W/cols)))
    else:
        rows = tiles_on_long
        cols = max(1, round(W / (H/rows)))
    tile_w = math.ceil(W / cols)
    tile_h = math.ceil(H / rows)
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))
    bboxes, y = [], 0
    while True:
        x = 0
        bottom = min(y + tile_h, H)
        top = max(0, bottom - tile_h)
        row = []
        while True:
            right = min(x + tile_w, W)
            left = max(0, right - tile_w)
            row.append((left, top, right, bottom))
            if right >= W: break
            x += step_x
        bboxes.append(row)
        if bottom >= H: break
        y += step_y
    return bboxes

def tiled_ocr_fallback(orig: Image.Image, doc_id: str, img_path: Path, out_f, rpm_win: list):
    """타일 OCR → 합본 텍스트와 성공여부 반환 (combined, ok). 동시에 합본 레코드 1건 기록."""
    bboxes = make_tiles(orig)
    texts = []
    tile_count = 0
    for r, row in enumerate(bboxes):
        for c, (x0,y0,x1,y1) in enumerate(row):
            tile = enhance_for_text(orig.crop((x0,y0,x1,y1)))
            ratelimit(rpm_win)
            t = call_with_timeout_upgrade(tile)
            texts.append((t or "").strip())
            tile_count += 1

    combined = "\n".join([t for t in texts if t])
    rec = {
        "doc_id": doc_id,
        "source_path": str(img_path),
        "frame_index": 0,
        "text": combined,
        "avg_conf": -1.0,
        "lang": "ko+en",
        "preprocess": {"mode":"tiled","tile_long":TARGET_TILE_LONG_SIDE,"overlap":TILE_OVERLAP,
                       "upscale":UPSCALE_BEFORE_OCR,"send_side":MAX_SEND_SIDE,"jpeg_q":JPEG_QUALITY},
        "ts": now_iso(),
        "source_type": "asset_ocr_html_tiled_agg",
        "provider": "openai",
        "model": "auto(nano→mini on timeout)",
        "tile_count": tile_count
    }
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
    ok = not text_quality_is_poor(combined)
    return combined, ok

# ====== 스캔 ======
def iter_html_assets():
    pattern = str(HTML_ROOT / "**" / "assets" / "*")
    for p in glob.glob(pattern, recursive=True):
        path = Path(p)
        if path.suffix.lower() not in VALID_EXT:
            continue
        if should_ignore(path):
            continue
        try:
            pil = Image.open(str(path))
        except UnidentifiedImageError:
            continue
        yield path, getattr(pil, "is_animated", False), pil

# ====== 실행 ======
INTERIM.mkdir(parents=True, exist_ok=True)
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# 캐시 로드
cache = {}
if CACHE_JSON.exists():
    try:
        cache = json.load(open(CACHE_JSON, "r", encoding="utf-8"))
    except Exception:
        cache = {}

rpm_win, total = [], 0

with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
    for img_path, is_multi, pil in tqdm(iter_html_assets(), desc="OpenAI OCR (html)"):
        md5 = file_md5(img_path)
        c = cache.get(md5)

        # ---- 캐시에 텍스트가 있으면 → OCR 생략 + from_cache 기록 ----
        if isinstance(c, dict) and c.get("text"):
            doc_id = extract_doc_id(img_path)
            rec = {
                "doc_id": doc_id, "source_path": str(img_path), "frame_index": 0,
                "text": c["text"], "avg_conf": -1.0, "lang": c.get("lang","ko+en"),
                "preprocess": c.get("preprocess", {"mode":"single"}),
                "ts": now_iso(), "source_type":"asset_ocr_html_from_cache",
                "provider":"openai","model": c.get("model","cache")
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
            total += 1
            continue

        # ---- 그 외(불리언/미존재) → OCR 시도 ----
        doc_id = extract_doc_id(img_path)
        tqdm.write(f"→ {doc_id} / {img_path.name}")

        try:
            # 1) 단일컷
            if is_multi:
                texts=[]
                for frame in ImageSequence.Iterator(pil):
                    ratelimit(rpm_win)
                    texts.append(call_with_timeout_upgrade(frame.convert("RGB")))
                text = "\n".join(t or "" for t in texts)
            else:
                ratelimit(rpm_win)
                text = call_with_timeout_upgrade(pil.convert("RGB"))

            # 2) 부실컷이면 타일링 폴백
            if text_quality_is_poor(text):
                combined, ok = tiled_ocr_fallback(pil.convert("RGB"), doc_id, img_path, out_f, rpm_win)
                if ok:
                    text = combined
                else:
                    # 3) 여전히 부실하면 (기본) mini 1회 재시도
                    ratelimit(rpm_win)
                    text = call_once(MODEL_TIMEOUT_UP, pil.convert("RGB"))
                    # (옵션) 테스트 허용시 gpt-4.1 → gpt-4o 순서로 1회씩만 추가 시도
                    if USE_TEST_MODELS and text_quality_is_poor(text):
                        for tm in TEST_MODEL_ORDER:
                            ratelimit(rpm_win)
                            text = call_once(tm, pil.convert("RGB"))
                            if not text_quality_is_poor(text):
                                break

            # 4) 성공 저장: JSONL + 캐시(dict)
            rec = {
                "doc_id": doc_id, "source_path": str(img_path), "frame_index": 0,
                "text": text, "avg_conf": -1.0, "lang":"ko+en",
                "preprocess":{"mode":"single","send_side":MAX_SEND_SIDE,"jpeg_q":JPEG_QUALITY},
                "ts": now_iso(), "source_type":"asset_ocr_html",
                "provider":"openai", "model": f"auto({MODEL_PRIMARY}→{MODEL_TIMEOUT_UP} on timeout)"
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
            cache[md5] = {
                "text": text,
                "model": rec["model"],
                "preprocess": rec["preprocess"],
                "lang": "ko+en",
                "ts": rec["ts"]
            }
            total += 1

        except Exception as e:
            tqdm.write(f"[fallback] tiled OCR for {img_path.name} because: {type(e).__name__} {e}")
            combined, ok = tiled_ocr_fallback(pil.convert("RGB"), doc_id, img_path, out_f, rpm_win)
            if ok:
                cache[md5] = {
                    "text": combined,
                    "model": f"auto({MODEL_PRIMARY}→{MODEL_TIMEOUT_UP} on timeout)",
                    "preprocess": {"mode":"tiled","tile_long":TARGET_TILE_LONG_SIDE,"overlap":TILE_OVERLAP},
                    "lang": "ko+en",
                    "ts": now_iso()
                }
                total += 1
            else:
                err = {"doc_id":doc_id,"source_path":str(img_path),"error":repr(e),
                       "ts":now_iso(),"source_type":"asset_ocr_html_error","provider":"openai"}
                out_f.write(json.dumps(err, ensure_ascii=False)+"\n"); out_f.flush()

# 캐시 저장
with open(CACHE_JSON, "w", encoding="utf-8") as cf:
    json.dump(cache, cf, ensure_ascii=False, indent=2)

print(f"[OK] HTML OCR complete → {OUT_JSONL} (processed or recorded: {total})")

# ====== 요약 리포트 ======
def summary_report():
    def img_list():
        return [p for p in HTML_ROOT.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_EXT and not should_ignore(p)]
    img_paths = img_list()
    img_paths_set = set(str(p) for p in img_paths)

    jsonl_records = []
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: jsonl_records.append(json.loads(line))
                except json.JSONDecodeError: pass

    jsonl_paths = [r.get("source_path") for r in jsonl_records if r.get("source_path")]
    jsonl_paths_set = set(jsonl_paths)
    source_types = Counter(r.get("source_type","") for r in jsonl_records)

    missing_on_jsonl = sorted(img_paths_set - jsonl_paths_set)
    orphan_in_jsonl  = sorted(jsonl_paths_set - img_paths_set)

    print("\n=== SUMMARY ===")
    print(f" 디스크 이미지 수(숨김/체크포인트 제외): {len(img_paths_set)}")
    print(f" JSONL 기록(고유 source_path) 수: {len(jsonl_paths_set)}")
    print(" 타입별 개수:", dict(source_types))
    print(f" JSONL에 누락된 경로 수: {len(missing_on_jsonl)}")
    print(f" 디스크엔 없는데 JSONL에만 있는 경로 수: {len(orphan_in_jsonl)}")
    if missing_on_jsonl[:10]:
        print("\n--- JSONL에 누락된 이미지 예시(최대 10) ---")
        for p in missing_on_jsonl[:10]:
            print("  -", p)

summary_report()


# In[ ]:


# # 부실컷 목록 확인 : 삭제 안하고 있었을 경우

# jq -r 'select(.source_type=="asset_ocr_html" and ((.text // "") | length) < 20) | .source_path' \
#   /home/spai0308/data/interim/assets_html_ocr.jsonl > /home/spai0308/data/interim/retry_list_html.txt

# jq -r 'select(.source_type=="asset_ocr_html_error") | .source_path' \
#   /home/spai0308/data/interim/assets_html_ocr.jsonl >> /home/spai0308/data/interim/retry_list_html.txt

# sort -u -o /home/spai0308/data/interim/retry_list_html.txt /home/spai0308/data/interim/retry_list_html.txt


# #### 4) XML 자산 OCR로 돌리기

# - 상단의 html 자산과 merge
# - 중복 제거(deddup)
# -> 최종 jsonl 생성 

# **[전략]**
# 
# 
# 
# - **텍스트 추출**:
#     - **XML에서 텍스트** 추출(문단/블록) + 필요하면 HTML의 순수 텍스트도 추가.
# - **이미지 OCR**:
#     - **한 쪽만 택해서** 돌린다 → 지금은 **HTML 자산(/assets)** 을 이미 OCR 완료했으니 **XML 이미지 OCR은 끈다**.
#     - 이렇게 해야 **중복 OCR/비용/중복 레코드**를 막을 수 있음
# 
# 3) 병합(merge) 단계에서 일어나는 일
# 
# - **XML 텍스트 블록** + **HTML 이미지 OCR 텍스트**를 **같은 스키마**로 모은 뒤,
# - 정규화한 문자열의 **md5(텍스트 기반)** 를 키로 **(doc_id, text_md5)** 중복 제거 →
#     - **같은 내용**이 두 소스에 있으면 **우선순위(XML > OCR)** 로 하나만 남기고,
#     - 내용이 조금이라도 다르면 **둘 다 남음**(과도 삭제 방지).
# 
# 4) 예외적으로 OCR을 추가해야 하는 상황
# 
# - **XML에만 존재하는 이미지가 있고**, 그 안에 **중요 텍스트**(예: 표의 숫자, 제출양식 필드)가 들어있는데
#     
#     HTML 자산에는 **동일 이미지가 없을 때** → **그 “누락된 이미지”만 선별적으로 OCR**하면 됨.
#     
#     (즉, 기본은 한쪽만, **누락이 확인된 경우에만 예외적으로 보강**)

# In[ ]:


# -*- coding: utf-8 -*-
"""
[무엇을 하나요?]
- /raw/converted/html/**/assets/* 와 /raw/converted/xml/**/assets/* 안의 이미지에서
  OpenAI 비전으로 텍스트를 읽어 JSONL로 저장합니다.
- HWPX가 아닌 'HTML/XML 변환본의 첨부 이미지'용입니다.
- 모델 자동 승급 정책은 타이트하게 조정: 문서당 정말 필요한 몇 장만 상위 모델로 실시
    - OEPNAI_OCR_MAX_ESCALATION_STEPS=1
    - OPENAI_OCR_DOC_UPGRADE_BUDGET=3
    - 승급 기준 완화 : OPENAI_OCR_RPM = 20으로 수정(15~30 사이 설정이 좋음)

[출력]
- /home/spai0308/data/interim/assets_html_ocr.jsonl
- /home/spai0308/data/interim/assets_xml_ocr.jsonl
"""

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

OUT_HTML  = INTERIM / "assets_html_ocr.jsonl"              # HTML 자산 OCR 결과
OUT_XML   = INTERIM / "assets_xml_ocr.jsonl"               # XML  자산 OCR 결과
CACHE_HTML = INTERIM / "assets_html_ocr.openai.cache.json" # 이미 처리한 이미지(해시) 캐시
CACHE_XML  = INTERIM / "assets_xml_ocr.openai.cache.json"

# ====== 시간대 ======
KST = timezone(timedelta(hours=9))
def now_kst_iso(): return datetime.now(KST).isoformat()

# ====== OpenAI ======
client = OpenAI()

# ====== 모델 정책(다름) ======
MODEL_LADDER = [
    os.getenv("OPENAI_OCR_MODEL_PRIMARY",   "gpt-4.1-nano"),
    os.getenv("OPENAI_OCR_MODEL_SECONDARY", "gpt-4.1-mini"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK1", "gpt-4.1"),
    os.getenv("OPENAI_OCR_MODEL_FALLBACK2", "gpt-4o"),
    os.getenv("ASSETS_KIND", "xml")
]
MAX_ESCALATION_STEPS = int(os.getenv("OPENAI_OCR_MAX_ESCALATION_STEPS", "1"))
DOC_UPGRADE_BUDGET   = int(os.getenv("OPENAI_OCR_DOC_UPGRADE_BUDGET", "3"))
MAX_REQ_PER_MIN      = int(os.getenv("OPENAI_OCR_RPM", "20"))

MIN_CHARS       = int(os.getenv("OPENAI_OCR_MIN_CHARS", "20"))
MIN_ALPHA_RATIO = float(os.getenv("OPENAI_OCR_MIN_ALPHA_RATIO", "0.3"))
HANGUL_BONUS    = float(os.getenv("OPENAI_OCR_HANGUL_BONUS", "0.1"))
                  

VALID_EXT = {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp"}

def file_md5(p: Path) -> str:
    """
    - 이미지 내용을 md5 해시로 바꿔, '이미 처리한 파일'은 건너뜁니다.
      (캐시 파일에 기록)
    """
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_doc_id(img_path: Path) -> str:
    """
    - 경로 예: /converted/html/<doc_id>/assets/파일명
    - 위 구조에서 <doc_id>를 뽑아내 문서 식별자로 사용합니다.
    """
    parts = img_path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i + 3 < len(parts) and parts[i+3] == "assets":
            return parts[i+2]
    return "unknown"

# 품질
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
    - kind = 'html' 또는 'xml'
    - /converted/<kind>/<doc_id>/assets/* 아래의 이미지들을 재귀적으로 찾습니다.
    - yield (이미지경로, 멀티프레임여부, PIL.Image)
    """
    pattern = str(CONVERTED / kind / "**" / "assets" / "*")
    for p in glob.glob(pattern, recursive=True):
        path = Path(p)
        if path.suffix.lower() not in VALID_EXT:
            continue
        try:
            pil = Image.open(str(path))
        except UnidentifiedImageError:
            continue
        yield path, getattr(pil, "is_animated", False), pil

def run_openai_assets_ocr(kind: str, out_jsonl: Path, cache_json: Path):
    """
    - HTML 또는 XML의 assets 이미지에 대해 OCR을 실행하고,
      중복(이미 처리한 md5)은 캐시로 건너뜁니다.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 캐시 불러오기(이미 처리한 파일의 md5 목록)
    cache = {}
    if cache_json.exists():
        try:
            cache = json.load(open(cache_json, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    rpm_win = []           # 레이트리밋 창
    doc_budget_used = {}   # 문서별 승급 사용량
    total = 0

    with open(out_jsonl, "a", encoding="utf-8") as out_f:
        for img_path, is_multi, pil in tqdm(iter_asset_images(kind), desc=f"OpenAI OCR ({kind})"):
            try:
                md5 = file_md5(img_path)
                if md5 in cache:
                    continue                       # 이미 처리함 → 스킵
                doc_id = extract_doc_id(img_path)

                if is_multi:
                    # GIF 등 멀티프레임 처리
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
                    # 일반 단일 이미지
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

                cache[md5] = True   # 캐시에 처리 완료 표시
                total += 1

            except Exception as e:
                # 실패 로그도 한 줄로 남겨두기
                err = {
                    "doc_id": "unknown",
                    "source_path": str(img_path),
                    "error": repr(e),
                    "ts": now_kst_iso(),
                    "source_type": f"asset_ocr_{kind}_error",
                    "provider": "openai",
                }
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

    # 캐시 파일 업데이트
    with open(cache_json, "w", encoding="utf-8") as cf:
        json.dump(cache, cf, ensure_ascii=False, indent=2)

    print(f"[OK] OpenAI OCR complete ({kind}) → {out_jsonl} (total {total} images)")

if __name__ == "__main__":
    import os
    KIND = os.getenv("ASSETS_KIND") # "html" 또는 "xml" 지정 시 해당만 실행
    if KIND in ("html", "xml"):
        run_openai_assets_ocr(KIND, OUT_HTML if KIND=="html" else OUT_XML,
                              CACHE_HTML if KIND=="html" else CACHE_XML)
    else:
        #지정 안 하면 두개 다 실행 기본)
        # HTML 자산 이미지 OCR
        run_openai_assets_ocr("html", OUT_HTML, CACHE_HTML)
        # XML  자산 이미지 OCR
        run_openai_assets_ocr("xml",  OUT_XML,  CACHE_XML)


# **[결과]**
# 
# 
# **1) HTML 패스**
# 
# ```
# OpenAI OCR (html): 174it [04:36,  1.59s/it]
# [OK] OpenAI OCR complete (html) → /home/spai0308/data/interim/assets_html_ocr.jsonl (total 26 images)
# 
# ```
# 
# - **174it**: HTML 쪽 자산 폴더에서 **후보 이미지 174장**을 순회(스캔)했다는 뜻.
# - **total 26 images**: 그중 **진짜로 API에 OCR을 보낸 건 26장**뿐이라는 뜻.
#     
#     나머지는 **캐시에서 텍스트를 찾아 재사용**했고, 그래서 전체 시간도 4분대로 비교적 짧은 시간내에 완료
#     
# - 결과는 **`assets_html_ocr.jsonl`*에 기록됨. (캐시 재사용분도 레코드로 남도록 짜놨다면 174개 전부가 경로별로 들어가 있음)
# 
# 
# **2) XML 패스**
# 
# ```
# OpenAI OCR (xml): 46it [10:30, 13.72s/it]
# [OK] OpenAI OCR complete (xml) → /home/spai0308/data/interim/assets_xml_ocr.jsonl (total 44 images)
# 
# ```
# 
# - **46it**: XML에서 **이미지로 분류된 항목 46개**를 순회했다는 뜻.
# - **total 44 images**: 그중 **44개는 실제로 OCR을 새로 호출**했다는 뜻(캐시 히트는 2개 수준이었던 셈).
#     
#     → HTML과 달리 **처음 보는 이미지가 많아서** 실제 호출이 거의 전부였고, 그래서 **시간이 길어졌음.**
#     
#     - 결과는 **`assets_xml_ocr.jsonl`*에 기록됨.
# 

# In[ ]:


# 확인
get_ipython().system('ls ~/data/interim')


# In[ ]:


get_ipython().system('wc -l /home/spai0308/data/interim/assets_html_ocr.jsonl')
print("="*50)
get_ipython().system('head -n 10 /home/spai0308/data/interim/assets_html_ocr.jsonl | jq .')
print("="*50)
get_ipython().system('wc -l /home/spai0308/data/interim/assets_xml_ocr.jsonl')
print("="*50)
get_ipython().system('head -n 10 /home/spai0308/data/interim/assets_xml_ocr.jsonl | jq .')


# #### 5) html VS xml 텍스트 비교

# - XML에서 순수 텍스트 블록을 뽑음
# - HTML에서 순수 텍스트 블록을 뽑음(BeautifulSoup 있으면 사용, 없으면 표준 html.parser로 폴백)
# - HTML 자산 이미지 OCR(이미 만든 assets_html_ocr.jsonl)도 함께 비교
# - 블록 정규화 → (doc_id, md5(normalized_text)) 기준으로 겹침/차이를 집계
# - **문서별 커버리지 표를 CSV로 저장** + **예시 미스매치 샘플** 출력

# In[ ]:


# -*- coding: utf-8 -*-
"""
XML 텍스트 vs HTML 텍스트(+ HTML 이미지 OCR) 비교 리포트
- 비교 축: XML_text, HTML_text, HTML_OCR(assets_html_ocr.jsonl)
- 산출물: /home/spai0308/data/interim/compare_xml_html_coverage.csv (문서별 커버리지)
- 콘솔: 총괄 요약 + 문서별 미스매치 샘플
"""

import re, json, hashlib, io
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm

# ========== 경로 ==========
DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"
XML_ROOT  = DATA / "raw" / "converted" / "xml"
INTERIM   = DATA / "interim"
OCR_HTML_JSONL = INTERIM / "assets_html_ocr.jsonl"
OUT_CSV = INTERIM / "compare_xml_html_coverage.csv"

KST = timezone(timedelta(hours=9))
now_iso = lambda: datetime.now(KST).isoformat(timespec="seconds")

# ========== 유틸 ==========
def should_ignore(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts: return True
    if any(part.startswith(".") for part in p.parts): return True
    return False

def extract_doc_id_from_path(path: Path) -> str:
    parts = path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+2 < len(parts):
            return parts[i+2]
    return "unknown"

_ZWS = "".join(chr(c) for c in [0x200B,0x200C,0x200D,0x2060,0xFEFF])
_WS_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\xa0", " ").replace("\u00A0", " ")
    for ch in _ZWS: s = s.replace(ch, "")
    s = _WS_RE.sub(" ", s).strip()
    return s

MIN_CHARS = 20
MIN_ALPHA_RATIO = 0.20
HANGUL_BONUS = 0.10
def text_quality_ok(s: str) -> bool:
    t = normalize_text(s)
    if len(t) < MIN_CHARS: return False
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha/total if total else 0.0) + (HANGUL_BONUS if hangul>0 else 0.0)
    return ratio >= MIN_ALPHA_RATIO

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# ========== HTML 텍스트 추출 ==========
def iter_html_files(root: Path):
    for p in root.rglob("*.html"):
        if p.is_file() and not should_ignore(p):
            yield p

def html_to_text_blocks(html_path: Path):
    """BeautifulSoup이 있으면 사용, 없으면 html.parser 폴백."""
    text = ""
    try:
        from bs4 import BeautifulSoup  # 선택 의존성
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        # script/style 제거
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        # 표준 html.parser 기반 간단 스트리퍼
        from html.parser import HTMLParser
        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.buf = []
            def handle_data(self, data):
                if data: self.buf.append(data)
        s = _Stripper()
        s.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(s.buf)

    text = text.replace("\r\n","\n").replace("\r","\n")
    # 문단 분리: 빈 줄 기준
    paras = re.split(r"\n{2,}", text)
    blocks = []
    for para in paras:
        para = normalize_text(para.replace("\n"," "))
        if text_quality_ok(para):
            blocks.append(para)
    return blocks

# ========== XML 텍스트 추출 ==========
def iter_xml_files(root: Path):
    for p in root.rglob("*.xml"):
        if p.is_file() and not should_ignore(p):
            yield p

def xml_to_text_blocks(xml_path: Path):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    raw = "".join(root.itertext()).replace("\r\n","\n").replace("\r","\n")
    paras = re.split(r"\n{2,}", raw)
    blocks = []
    for para in paras:
        para = normalize_text(para.replace("\n"," "))
        if text_quality_ok(para):
            blocks.append(para)
    return blocks

# ========== HTML 이미지 OCR(JSONL) 로더 ==========
def iter_ocr_records(jsonl_path: Path):
    if not jsonl_path.exists(): return
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = normalize_text(r.get("text",""))
            if not text_quality_ok(text):
                continue
            yield {
                "doc_id": r.get("doc_id","unknown"),
                "text": text,
                "source_path": r.get("source_path",""),
                "source_type": r.get("source_type","asset_ocr_html"),
            }

# ========== 스캔 & 집계 ==========
# 각 소스별: doc_id -> set(md5) 와 md5->원문 매핑(샘플 출력용)
xml_set = defaultdict(set); xml_text_by_md5 = {}
html_set = defaultdict(set); html_text_by_md5 = {}
ocr_set = defaultdict(set);  ocr_text_by_md5  = {}

# HTML 텍스트
for hp in tqdm(list(iter_html_files(HTML_ROOT)), desc="HTML text"):
    doc_id = extract_doc_id_from_path(hp)
    for blk in html_to_text_blocks(hp):
        m = md5_text(blk)
        html_set[doc_id].add(m)
        if m not in html_text_by_md5:
            html_text_by_md5[m] = blk

# XML 텍스트
for xp in tqdm(list(iter_xml_files(XML_ROOT)), desc="XML text"):
    doc_id = extract_doc_id_from_path(xp)
    for blk in xml_to_text_blocks(xp):
        m = md5_text(blk)
        xml_set[doc_id].add(m)
        if m not in xml_text_by_md5:
            xml_text_by_md5[m] = blk

# HTML 이미지 OCR
for r in tqdm(list(iter_ocr_records(OCR_HTML_JSONL)), desc="HTML OCR(jsonl)"):
    doc_id = r["doc_id"]; blk = r["text"]; m = md5_text(blk)
    ocr_set[doc_id].add(m)
    if m not in ocr_text_by_md5:
        ocr_text_by_md5[m] = blk

# ========== 문서별 커버리지 계산 ==========
doc_ids = sorted(set(xml_set.keys()) | set(html_set.keys()) | set(ocr_set.keys()))

rows = []
totals = Counter()

def safe_pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0

for doc in doc_ids:
    X = xml_set.get(doc, set())
    H = html_set.get(doc, set())
    O = ocr_set.get(doc, set())
    xi, hi, oi = len(X), len(H), len(O)

    X_and_H  = len(X & H)
    X_and_O  = len(X & O)
    H_and_O  = len(H & O)

    # 커버리지 (A가 B를 커버): |A∩B| / |B|
    cov_xml_covers_html = safe_pct(X_and_H, hi)
    cov_xml_covers_ocr  = safe_pct(X_and_O, oi)
    cov_html_covers_xml = safe_pct(X_and_H, xi)

    only_xml  = len(X - (H | O))
    only_html = len(H - (X | O))
    only_ocr  = len(O - (X | H))

    rows.append({
        "doc_id": doc,
        "xml_blocks": xi,
        "html_blocks": hi,
        "htmlocr_blocks": oi,
        "xml∩html": X_and_H,
        "xml∩ocr": X_and_O,
        "html∩ocr": H_and_O,
        "cov_xml→html(%)": cov_xml_covers_html,
        "cov_xml→ocr(%)":  cov_xml_covers_ocr,
        "cov_html→xml(%)": cov_html_covers_xml,
        "only_xml": only_xml,
        "only_html": only_html,
        "only_htmlocr": only_ocr,
    })

    totals["xml"] += xi
    totals["html"] += hi
    totals["ocr"] += oi
    totals["X∩H"] += X_and_H
    totals["X∩O"] += X_and_O
    totals["H∩O"] += H_and_O
    totals["only_xml"]  += only_xml
    totals["only_html"] += only_html
    totals["only_ocr"]  += only_ocr

# ========== CSV 저장 ==========
import csv
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["doc_id"])
    w.writeheader()
    for r in rows: w.writerow(r)

print(f"\n[OK] 문서별 커버리지 저장 → {OUT_CSV}")

# ========== 총괄 요약 ==========
def pct(n, d): 
    return f"{(100*n/d):.2f}%" if d else "0%"
tot_xml, tot_html, tot_ocr = totals["xml"], totals["html"], totals["ocr"]
print("\n===== OVERALL =====")
print(f"XML text blocks      : {tot_xml}")
print(f"HTML text blocks     : {tot_html}")
print(f"HTML OCR blocks      : {tot_ocr}")
print(f"Overlap XML∩HTML     : {totals['X∩H']}  (HTML covered by XML: {pct(totals['X∩H'], tot_html)})")
print(f"Overlap XML∩HTML_OCR : {totals['X∩O']}  (HTML_OCR covered by XML: {pct(totals['X∩O'], tot_ocr)})")
print(f"Only XML             : {totals['only_xml']}")
print(f"Only HTML            : {totals['only_html']}")
print(f"Only HTML_OCR        : {totals['only_ocr']}")

# ========== 미스매치 샘플 출력 ==========
def show_samples(doc_id, n_each=2):
    X = xml_set.get(doc_id, set())
    H = html_set.get(doc_id, set())
    O = ocr_set.get(doc_id, set())

    print(f"\n--- SAMPLES for doc_id={doc_id} ---")
    only_xml  = list(X - (H | O))[:n_each]
    only_html = list(H - (X | O))[:n_each]
    only_ocr  = list(O - (X | H))[:n_each]

    if only_xml:
        print("\n[only in XML text]")
        for m in only_xml:
            t = xml_text_by_md5.get(m,"")[:240]
            print(" •", t + ("..." if len(t)==240 else ""))

    if only_html:
        print("\n[only in HTML text]")
        for m in only_html:
            t = html_text_by_md5.get(m,"")[:240]
            print(" •", t + ("..." if len(t)==240 else ""))

    if only_ocr:
        print("\n[only in HTML OCR]")
        for m in only_ocr:
            t = ocr_text_by_md5.get(m,"")[:240]
            print(" •", t + ("..." if len(t)==240 else ""))

# 문서 3개 정도 샘플
for doc in doc_ids[:3]:
    show_samples(doc)


# #### 5-1) 의미 유사성 기반으로 xml이 커버하는 범위비교

# In[ ]:


# -*- coding: utf-8 -*-
"""
의미 유사도(퍼지) 기반 커버리지:
- XML 텍스트가 HTML 텍스트/HTML OCR을 '의미적으로' 얼마나 커버하는지 측정
- 외부 API 없이 로컬에서 동작 (토큰 Jaccard, 문자 3그램 Jaccard, difflib 시퀀스 유사도)
- 출력:
    /home/spai0308/data/interim/semantic_coverage.csv
    /home/spai0308/data/interim/semantic_samples.txt
"""

import re, json, unicodedata, difflib, csv, math, io
from pathlib import Path
from collections import defaultdict, Counter

# ================== 경로 ==================
DATA = Path("/home/spai0308/data")
HTML_ROOT = DATA / "raw" / "converted" / "html"
XML_ROOT  = DATA / "raw" / "converted" / "xml"
INTERIM   = DATA / "interim"

OCR_HTML_JSONL = INTERIM / "assets_html_ocr.jsonl"   # 이미 생성한 파일
OUT_CSV  = INTERIM / "semantic_coverage.csv"
OUT_TXT  = INTERIM / "semantic_samples.txt"

# ================== 파라미터 ==================
MIN_CHARS = 20
MIN_ALPHA_RATIO = 0.20
HANGUL_BONUS = 0.10

# 의미 유사도 판정 임계값(경험치): 0.72 이상이면 '의미적 동일'로 간주
THRESH = 0.72

# 후보 압축: 토큰이 2개 이상 겹치는 블록만 상세 비교(성능)
MIN_SHARED_TOKENS = 2

# 문서별 샘플 출력 개수
SAMPLES_PER_DOC = 2

# ================== 유틸 ==================
def should_ignore(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts: return True
    if any(part.startswith(".") for part in p.parts): return True
    return False

def extract_doc_id_from_path(path: Path) -> str:
    parts = path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+2 < len(parts):
            return parts[i+2]
    return "unknown"

_ZWS = "".join(chr(c) for c in [0x200B,0x200C,0x200D,0x2060,0xFEFF])
_WS_RE = re.compile(r"\s+")

def normalize_base(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0"," ").replace("\u00A0"," ")
    for ch in _ZWS: s = s.replace(ch, "")
    s = _WS_RE.sub(" ", s).strip()
    return s

def text_quality_ok(s: str) -> bool:
    t = normalize_base(s)
    if len(t) < MIN_CHARS: return False
    tot = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hang = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha/tot if tot else 0.0) + (HANGUL_BONUS if hang>0 else 0.0)
    return ratio >= MIN_ALPHA_RATIO

# 토큰: 한글/영문/숫자만 추출, 길이>=2
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
def tokens_of(s: str):
    s = normalize_base(s).lower()
    return set(_TOKEN_RE.findall(s))

# 문자 3그램(공백 제거 후)
def char_ngrams(s: str, n=3):
    s = normalize_base(s).lower().replace(" ", "")
    if len(s) < n: return set()
    return set(s[i:i+n] for i in range(len(s)-n+1))

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

def seq_ratio(a: str, b: str) -> float:
    a = normalize_base(a).lower()
    b = normalize_base(b).lower()
    return difflib.SequenceMatcher(None, a, b).ratio()

def best_semantic_score(textA: str, cand_list):
    """cand_list: [(textB, tokensB, c3B)]"""
    tokA = tokens_of(textA)
    c3A  = char_ngrams(textA, 3)
    best = 0.0
    best_b = None
    # 후보 축소: 토큰 2개 이상 겹치는 것만
    narrowed = []
    for (tB, tokB, c3B) in cand_list:
        if len(tokA & tokB) >= MIN_SHARED_TOKENS:
            narrowed.append((tB, tokB, c3B))
    # 후보가 없으면 전수(소량일 때) 혹은 char3 유사도만 쓰기
    base_list = narrowed if narrowed else cand_list
    for (tB, tokB, c3B) in base_list:
        s1 = jaccard(tokA, tokB)
        s2 = jaccard(c3A, c3B)
        s3 = seq_ratio(textA, tB)
        score = max(s1, s2, s3)
        if score > best:
            best = score
            best_b = tB
    return best, best_b

# ================== 로더 ==================
def iter_html_files(root: Path):
    for p in root.rglob("*.html"):
        if p.is_file() and not should_ignore(p):
            yield p

def html_to_text_blocks(html_path: Path):
    text = ""
    try:
        from bs4 import BeautifulSoup
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        from html.parser import HTMLParser
        class _Stripper(HTMLParser):
            def __init__(self): super().__init__(); self.buf=[]
            def handle_data(self, d): 
                if d: self.buf.append(d)
        s = _Stripper()
        s.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(s.buf)
    text = text.replace("\r\n","\n").replace("\r","\n")
    paras = re.split(r"\n{2,}", text)
    out=[]
    for para in paras:
        para = normalize_base(para.replace("\n"," "))
        if text_quality_ok(para): out.append(para)
    return out

def iter_xml_files(root: Path):
    for p in root.rglob("*.xml"):
        if p.is_file() and not should_ignore(p):
            yield p

def xml_to_text_blocks(xml_path: Path):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    raw = "".join(root.itertext()).replace("\r\n","\n").replace("\r","\n")
    paras = re.split(r"\n{2,}", raw)
    out=[]
    for para in paras:
        para = normalize_base(para.replace("\n"," "))
        if text_quality_ok(para): out.append(para)
    return out

def iter_ocr_records(jsonl_path: Path):
    if not jsonl_path.exists(): return
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = normalize_base(r.get("text",""))
            if not text_quality_ok(text): 
                continue
            yield r.get("doc_id","unknown"), text

# ================== 수집 ==================
from tqdm import tqdm

xml_blocks   = defaultdict(list)
html_blocks  = defaultdict(list)
ocrh_blocks  = defaultdict(list)  # HTML OCR

for xp in tqdm(list(iter_xml_files(XML_ROOT)), desc="XML text"):
    doc = extract_doc_id_from_path(xp)
    xml_blocks[doc].extend(xml_to_text_blocks(xp))

for hp in tqdm(list(iter_html_files(HTML_ROOT)), desc="HTML text"):
    doc = extract_doc_id_from_path(hp)
    html_blocks[doc].extend(html_to_text_blocks(hp))

for doc_id, txt in tqdm(list(iter_ocr_records(OCR_HTML_JSONL)), desc="HTML OCR"):
    ocrh_blocks[doc_id].append(txt)

# ================== 인덱싱(성능용) ==================
def index_blocks(blocks: list):
    """블록 -> (원문, 토큰셋, char3셋) 목록과 토큰 인덱스 반환"""
    triples = []
    inv = defaultdict(list)
    for i, t in enumerate(blocks):
        tok = tokens_of(t)
        c3  = char_ngrams(t, 3)
        triples.append((t, tok, c3))
        for w in tok:
            inv[w].append(i)
    return triples, inv

# ================== 의미 커버리지 계산 ==================
rows=[]
with open(OUT_TXT, "w", encoding="utf-8") as dbg:
    for doc in sorted(set(xml_blocks) | set(html_blocks) | set(ocrh_blocks)):
        X = xml_blocks.get(doc, [])
        H = html_blocks.get(doc, [])
        O = ocrh_blocks.get(doc, [])

        Xtriples, Xinv = index_blocks(X)

        def candidates_for(text):
            # 토큰 교집합 기반 후보 추출
            toks = tokens_of(text)
            cand_idx = set()
            for w in toks:
                for idx in Xinv.get(w, []):
                    cand_idx.add(idx)
            # 후보가 없으면 전체(소량 가정) — 너무 많으면 상위 일부 제한 가능
            if not cand_idx:
                cand_idx = set(range(len(Xtriples)))
                # 큰 문서라면 아래처럼 제한 두는 게 안전
                # cand_idx = set(list(range(min(500, len(Xtriples)))))
            return [Xtriples[i] for i in cand_idx]

        # HTML 텍스트 커버
        covered_h = 0
        only_html_samples = []
        for t in H:
            score, pair = best_semantic_score(t, candidates_for(t)) if Xtriples else (0.0, None)
            if score >= THRESH:
                covered_h += 1
            else:
                if len(only_html_samples) < SAMPLES_PER_DOC:
                    only_html_samples.append((t, score, pair))

        # HTML OCR 커버
        covered_o = 0
        only_ocr_samples = []
        for t in O:
            score, pair = best_semantic_score(t, candidates_for(t)) if Xtriples else (0.0, None)
            if score >= THRESH:
                covered_o += 1
            else:
                if len(only_ocr_samples) < SAMPLES_PER_DOC:
                    only_ocr_samples.append((t, score, pair))

        cov_h = round(100.0*covered_h/len(H), 2) if H else 0.0
        cov_o = round(100.0*covered_o/len(O), 2) if O else 0.0

        rows.append({
            "doc_id": doc,
            "xml_blocks": len(X),
            "html_blocks": len(H),
            "htmlocr_blocks": len(O),
            "sem_cov_xml→html(%)": cov_h,
            "sem_cov_xml→htmlocr(%)": cov_o,
            "html_not_covered_cnt": len(H)-covered_h,
            "htmlocr_not_covered_cnt": len(O)-covered_o,
        })

        # 샘플 덤프 (증거용)
        print(f"\n=== {doc} ===", file=dbg)
        print(f"[XML blocks: {len(X)} | HTML: {len(H)} | HTML_OCR: {len(O)}]", file=dbg)
        if only_html_samples:
            print("\n[Only in HTML text (not semantically covered by XML)]", file=dbg)
            for t, sc, pair in only_html_samples:
                print(f" • score={sc:.3f} | HTML: {t[:300]}", file=dbg)
                if pair:
                    print(f"          closest XML: {pair[:300]}", file=dbg)
        if only_ocr_samples:
            print("\n[Only in HTML OCR (not semantically covered by XML)]", file=dbg)
            for t, sc, pair in only_ocr_samples:
                print(f" • score={sc:.3f} | OCR: {t[:300]}", file=dbg)
                if pair:
                    print(f"          closest XML: {pair[:300]}", file=dbg)

# ================== 저장 & 총괄 ==================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["doc_id"])
    w.writeheader()
    for r in rows: w.writerow(r)

print(f"[OK] 의미 커버리지 저장 → {OUT_CSV}")
print(f"[OK] 샘플 덤프 → {OUT_TXT}")


# In[ ]:


get_ipython().system('cat /home/spai0308/data/interim/semantic_samples.txt')


# **[의미 커버리지 & 샘플 덤프 확인]**
# 
# - semantic_coverage.csv를 판다스로 로드해서 통계/요약/분포를 확인하는 코드
# - matplotlib 터미널에서 사전설치

# In[ ]:


# -*- coding: utf-8 -*-
# semantic_coverage.csv를 판다스로 로드해서 통계/요약/분포를 확인하는 코드 (단일 셀)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 경로 =====
PATH = Path("/home/spai0308/data/interim/semantic_coverage.csv")

# ===== 로드 =====
df = pd.read_csv(PATH)

# 컬럼 존재 확인 (이름이 다르면 여기서 확인)
expected = [
    "doc_id","xml_blocks","html_blocks","htmlocr_blocks",
    "sem_cov_xml→html(%)","sem_cov_xml→htmlocr(%)",
    "html_not_covered_cnt","htmlocr_not_covered_cnt"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    print("[WARN] 누락된 컬럼:", missing)

# 숫자형 변환 (퍼센트/카운트)
for c in ["sem_cov_xml→html(%)","sem_cov_xml→htmlocr(%)"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["xml_blocks","html_blocks","htmlocr_blocks","html_not_covered_cnt","htmlocr_not_covered_cnt"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

print("총 문서 수:", len(df))
print()

# ===== 전체 요약(coverage) =====
cov_cols = [c for c in ["sem_cov_xml→html(%)","sem_cov_xml→htmlocr(%)"] if c in df.columns]
if cov_cols:
    cov_stats = df[cov_cols].agg(["mean","median","min","max"]).round(2)
    print("=== 전체 커버리지 통계(%) ===")
    print(cov_stats)
    print()

# 기준치 이상 커버된 문서 비율
def coverage_report(th=80.0):
    out = {}
    if "sem_cov_xml→html(%)" in df.columns:
        out["XML→HTML ≥{:.0f}% 문서수".format(th)] = int((df["sem_cov_xml→html(%)"] >= th).sum())
    if "sem_cov_xml→htmlocr(%)" in df.columns:
        out["XML→HTML_OCR ≥{:.0f}% 문서수".format(th)] = int((df["sem_cov_xml→htmlocr(%)"] >= th).sum())
    return out

for th in [50, 80, 95, 99]:
    rpt = coverage_report(th)
    if rpt:
        print(f"=== 기준 {th}% 이상 문서 수 ===")
        for k,v in rpt.items():
            print(f" - {k}: {v} / {len(df)}")
        print()

# ===== 커버리지 낮은 문서 Top-N =====
N = 10
if "sem_cov_xml→html(%)" in df.columns:
    low_html = df.sort_values("sem_cov_xml→html(%)", ascending=True)[
        ["doc_id","xml_blocks","html_blocks","sem_cov_xml→html(%)","html_not_covered_cnt"]
    ].head(N)
    print(f"=== XML→HTML 커버리지 낮은 문서 Top {N} ===")
    print(low_html.to_string(index=False))
    print()

if "sem_cov_xml→htmlocr(%)" in df.columns:
    low_ocr = df.sort_values("sem_cov_xml→htmlocr(%)", ascending=True)[
        ["doc_id","xml_blocks","htmlocr_blocks","sem_cov_xml→htmlocr(%)","htmlocr_not_covered_cnt"]
    ].head(N)
    print(f"=== XML→HTML_OCR 커버리지 낮은 문서 Top {N} ===")
    print(low_ocr.to_string(index=False))
    print()

# ===== HTML-OCR 고유 정보가 많은 문서 Top-N =====
if "htmlocr_not_covered_cnt" in df.columns:
    top_only_ocr = df.sort_values("htmlocr_not_covered_cnt", ascending=False)[
        ["doc_id","xml_blocks","htmlocr_blocks","sem_cov_xml→htmlocr(%)","htmlocr_not_covered_cnt"]
    ].head(N)
    print(f"=== HTML-OCR 고유 블록(미커버) 많은 문서 Top {N} ===")
    print(top_only_ocr.to_string(index=False))
    print()

# ===== 분포 시각화 (각 1개 플롯, seaborn 사용 금지) =====
# 히스토그램: XML→HTML 커버리지
if "sem_cov_xml→html(%)" in df.columns:
    plt.figure()
    plt.hist(df["sem_cov_xml→html(%)"].dropna(), bins=20)
    plt.title("Semantic coverage: XML → HTML (%)")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Docs")
    plt.show()

# 히스토그램: XML→HTML_OCR 커버리지
if "sem_cov_xml→htmlocr(%)" in df.columns:
    plt.figure()
    plt.hist(df["sem_cov_xml→htmlocr(%)"].dropna(), bins=20)
    plt.title("Semantic coverage: XML → HTML_OCR (%)")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Docs")
    plt.show()

# 산점도(문서 크기 vs 커버리지): 참고용
if all(c in df.columns for c in ["html_blocks","sem_cov_xml→html(%)"]):
    plt.figure()
    plt.scatter(df["html_blocks"], df["sem_cov_xml→html(%)"])
    plt.title("HTML blocks vs XML→HTML coverage")
    plt.xlabel("HTML blocks")
    plt.ylabel("Coverage (%)")
    plt.show()

if all(c in df.columns for c in ["htmlocr_blocks","sem_cov_xml→htmlocr(%)"]):
    plt.figure()
    plt.scatter(df["htmlocr_blocks"], df["sem_cov_xml→htmlocr(%)"])
    plt.title("HTML_OCR blocks vs XML→HTML_OCR coverage")
    plt.xlabel("HTML_OCR blocks")
    plt.ylabel("Coverage (%)")
    plt.show()

# ===== (선택) 요약 CSV 저장 =====
summary_path = PATH.with_name("semantic_coverage_summary.csv")
summary_rows = []

row = {"metric":"docs","value":len(df)}
summary_rows.append(row)
if cov_cols:
    for stat in ["mean","median","min","max"]:
        for c in cov_cols:
            summary_rows.append({"metric":f"{c}_{stat}", "value":df[c].agg(stat)})

if "htmlocr_not_covered_cnt" in df.columns:
    summary_rows.append({"metric":"docs_with_htmlocr_not_covered>0", 
                         "value":int((df["htmlocr_not_covered_cnt"]>0).sum())})

pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print("요약 저장:", summary_path)


# ## 3) Test :병합/디듑 : xml 텍스트 + html 이미지 ocr + xml 이미지 ocr & 스모크 테스트

# ### 3-1) 최종 말뭉치 병합 & 중복제거

# In[ ]:


# -*- coding: utf-8 -*-
"""
최종 말뭉치 병합 & 중복 제거
- 입력:
  - /home/spai0308/data/raw/converted/xml/**/*.xml  (XML 텍스트)
  - /home/spai0308/data/interim/assets_html_ocr.jsonl  (HTML 이미지 OCR)
  - /home/spai0308/data/interim/assets_xml_ocr.jsonl   (XML 측 이미지 OCR)
  - (옵션) /home/spai0308/data/raw/converted/html/**/*.html  (HTML 순수 텍스트)
- 출력:
  - /home/spai0308/data/interim/rfp_text_merged.final.jsonl
- 중복 키:
  - (doc_id, md5(normalized_text))
"""

import re, json, hashlib, unicodedata
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

# ===== 경로 =====
DATA = Path("/home/spai0308/data")
XML_ROOT    = DATA / "raw" / "converted" / "xml"
HTML_ROOT   = DATA / "raw" / "converted" / "html"
INTERIM     = DATA / "interim"
HTML_OCR_JL = INTERIM / "assets_html_ocr.jsonl"
XML_OCR_JL  = INTERIM / "assets_xml_ocr.jsonl"
OUT_JSONL   = INTERIM / "rfp_text_merged.final.jsonl"

# ===== 옵션 =====
INCLUDE_HTML_TEXT = False  # HTML 순수 텍스트는 기본 꺼둠 (XML이 본문 커버)
DEDUP_SCOPE = "doc"        # "doc" 또는 "global"
MIN_CHARS = 20
MIN_ALPHA_RATIO = 0.20
HANGUL_BONUS = 0.10

# OCR 노이즈 문구 드롭
OCR_NOISE_PATTERNS = [
    "There is no image attached", 
    "no visible text in the image",
    "이미지가 없습니다", "텍스트가 보이지 않습니다"
]

# 우선순위(낮을수록 우선)
SOURCE_RANK = {
    "xml_text": 0,
    "html_text": 1,
    "asset_ocr_xml": 2,
    "asset_ocr_html": 3,
    "asset_ocr_html_tiled_agg": 4,
    "asset_ocr_xml_tiled_agg": 5,
    "asset_ocr_html_from_cache": 6,
    "asset_ocr_xml_from_cache": 7,
    "default": 9,
}

KST = timezone(timedelta(hours=9))
now_iso = lambda: datetime.now(KST).isoformat(timespec="seconds")

# ===== 유틸 =====
def should_ignore(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts: return True
    if any(part.startswith(".") for part in p.parts): return True
    return False

def extract_doc_id(path: Path) -> str:
    parts = path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+2 < len(parts):
            return parts[i+2]
    return "unknown"

_ZWS = "".join(chr(c) for c in [0x200B,0x200C,0x200D,0x2060,0xFEFF])
_WS_RE = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0"," ").replace("\u00A0"," ")
    for ch in _ZWS: s = s.replace(ch, "")
    s = _WS_RE.sub(" ", s).strip()
    return s

def text_quality_ok(s: str) -> bool:
    t = normalize_text(s)
    if len(t) < MIN_CHARS: return False
    total = len(t)
    alpha = sum(ch.isalpha() for ch in t)
    hang = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in t)
    ratio = (alpha/total if total else 0.0) + (HANGUL_BONUS if hang>0 else 0.0)
    return ratio >= MIN_ALPHA_RATIO

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===== 로더 =====
def iter_xml_files(root: Path):
    for p in root.rglob("*.xml"):
        if p.is_file() and not should_ignore(p): yield p

def iter_html_files(root: Path):
    for p in root.rglob("*.html"):
        if p.is_file() and not should_ignore(p): yield p

def xml_to_blocks(p: Path):
    try:
        tree = ET.parse(p); root = tree.getroot()
    except Exception: return []
    raw = "".join(root.itertext()).replace("\r\n","\n").replace("\r","\n")
    paras = re.split(r"\n{2,}", raw)
    out=[]
    for para in paras:
        para = normalize_text(para.replace("\n"," "))
        if text_quality_ok(para): out.append(para)
    return out

def html_to_blocks(p: Path):
    text = ""
    try:
        from bs4 import BeautifulSoup
        html = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        from html.parser import HTMLParser
        class _Stripper(HTMLParser):
            def __init__(self): super().__init__(); self.buf=[]
            def handle_data(self, d): 
                if d: self.buf.append(d)
        s = _Stripper()
        s.feed(p.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(s.buf)
    text = text.replace("\r\n","\n").replace("\r","\n")
    paras = re.split(r"\n{2,}", text)
    out=[]
    for para in paras:
        para = normalize_text(para.replace("\n"," "))
        if text_quality_ok(para): out.append(para)
    return out

def iter_jsonl(path: Path):
    if not path.exists(): return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                yield r
            except json.JSONDecodeError:
                continue

def is_noise_ocr(text: str) -> bool:
    t = normalize_text(text).lower()
    return any(pat.lower() in t for pat in OCR_NOISE_PATTERNS)

# ===== 병합 =====
def source_rank(stype: str) -> int:
    return SOURCE_RANK.get(stype, SOURCE_RANK["default"])

def merge_all():
    stats_in = Counter()
    stats_keep = Counter()
    dup_count = 0

    if DEDUP_SCOPE == "global":
        key_builder = lambda rec, md5: ("__GLOBAL__", md5)
    else:
        key_builder = lambda rec, md5: (rec["doc_id"], md5)

    merged = {}  # key -> record

    # 1) XML 텍스트 (최우선)
    for xp in iter_xml_files(XML_ROOT):
        doc_id = extract_doc_id(xp)
        for txt in xml_to_blocks(xp):
            rec = {
                "doc_id": doc_id,
                "source_path": str(xp),
                "text": txt,
                "source_type": "xml_text",
                "provider": "local",
                "model": "",
                "ts": now_iso(),
                "lang": "ko+en"
            }
            stats_in[rec["source_type"]] += 1
            key = key_builder(rec, md5_text(txt))
            prev = merged.get(key)
            if prev is None:
                merged[key] = rec; stats_keep[rec["source_type"]] += 1
            else:
                if source_rank(rec["source_type"]) < source_rank(prev["source_type"]):
                    merged[key] = rec
                dup_count += 1

    # 2) (옵션) HTML 순수 텍스트
    if INCLUDE_HTML_TEXT:
        for hp in iter_html_files(HTML_ROOT):
            doc_id = extract_doc_id(hp)
            for txt in html_to_blocks(hp):
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(hp),
                    "text": txt,
                    "source_type": "html_text",
                    "provider": "local",
                    "model": "",
                    "ts": now_iso(),
                    "lang": "ko+en"
                }
                stats_in[rec["source_type"]] += 1
                key = key_builder(rec, md5_text(txt))
                prev = merged.get(key)
                if prev is None:
                    merged[key] = rec; stats_keep[rec["source_type"]] += 1
                else:
                    if source_rank(rec["source_type"]) < source_rank(prev["source_type"]):
                        merged[key] = rec
                    dup_count += 1

    # 3) HTML 이미지 OCR
    for r in iter_jsonl(HTML_OCR_JL):
        txt = normalize_text(r.get("text",""))
        if not text_quality_ok(txt): 
            continue
        if is_noise_ocr(txt):
            continue
        st = r.get("source_type","asset_ocr_html")
        # 타일 합본/캐시 표기 표준화
        if "tiled" in st and "html" in st: st = "asset_ocr_html_tiled_agg"
        if "from_cache" in st and "html" in st: st = "asset_ocr_html_from_cache"
        rec = {
            "doc_id": r.get("doc_id","unknown"),
            "source_path": r.get("source_path",""),
            "text": txt,
            "source_type": st if "html" in st else "asset_ocr_html",
            "provider": r.get("provider","openai"),
            "model": r.get("model",""),
            "ts": r.get("ts", now_iso()),
            "lang": r.get("lang","ko+en")
        }
        stats_in[rec["source_type"]] += 1
        key = key_builder(rec, md5_text(txt))
        prev = merged.get(key)
        if prev is None:
            merged[key] = rec; stats_keep[rec["source_type"]] += 1
        else:
            if source_rank(rec["source_type"]) < source_rank(prev["source_type"]):
                merged[key] = rec
            dup_count += 1

    # 4) XML 측 이미지 OCR
    for r in iter_jsonl(XML_OCR_JL):
        txt = normalize_text(r.get("text",""))
        if not text_quality_ok(txt): 
            continue
        if is_noise_ocr(txt):
            continue
        # 파일 경로로 xml/ html 구분 보정
        st = r.get("source_type","asset_ocr_xml")
        sp = r.get("source_path","")
        if "/converted/xml/" in sp and "asset_ocr" in st and "xml" not in st:
            st = st.replace("html", "xml") if "html" in st else "asset_ocr_xml"
        if "tiled" in st and "xml" in st: st = "asset_ocr_xml_tiled_agg"
        if "from_cache" in st and "xml" in st: st = "asset_ocr_xml_from_cache"

        rec = {
            "doc_id": r.get("doc_id","unknown"),
            "source_path": sp,
            "text": txt,
            "source_type": st,
            "provider": r.get("provider","openai"),
            "model": r.get("model",""),
            "ts": r.get("ts", now_iso()),
            "lang": r.get("lang","ko+en")
        }
        stats_in[rec["source_type"]] += 1
        key = key_builder(rec, md5_text(txt))
        prev = merged.get(key)
        if prev is None:
            merged[key] = rec; stats_keep[rec["source_type"]] += 1
        else:
            if source_rank(rec["source_type"]) < source_rank(prev["source_type"]):
                merged[key] = rec
            dup_count += 1

    # 저장
    out_list = list(merged.values())
    write_jsonl(OUT_JSONL, out_list)

    print("=== MERGE REPORT ===")
    print("입력(원천별):", dict(stats_in))
    print("유지(머지 후):", dict(stats_keep))
    print("중복(키 충돌):", dup_count)
    print(f"최종 JSONL: {OUT_JSONL} (라인수: {len(out_list)})")

merge_all()


# ### 3-2) 스모크 테스트: 청킹→임베딩→리트리브→생성

# - **임베딩**: `text-embedding-3-small` (가성비 좋고 충분)
# - **생성**: `gpt-4.1-mini` (네가 지정한 정책 준수)
# - **토픽**: RFP에서 자주 묻는 항목 체크(사업명/발주기관/예산/기간/계약방식/마감/제출방법/문의처)

# In[ ]:


# # tiktoken 설치 : 커널
# import sys; print(sys.executable)

# 설치
# !python -m pip install -U pip setuptools wheel tiktoken

# 이후 커널 재시작

# # 설치 확인
# import tiktoken
# enc = tiktoken.get_encoding("cl100k_base")
# print(len(enc.encode("토큰화 테스트")))


# In[ ]:


# # tiktoken 설치 : 터미널
# which python # /home/spai0308/myenv/bin/python 확인

# # 설치
# python -m pip install -U pip setuptools wheel
# python -m pip install -U tiktoken

# # 설치 확인
# python -c "import tiktoken, sys; print('tiktoken:', tiktoken.__version__, '\npython:', sys.executable)"


# In[ ]:


# -*- coding: utf-8 -*-
"""
최종 말뭉치로 RAG 스모크 테스트
- 청킹(문단 기반, 길이/중첩 가변)
- OpenAI 임베딩으로 간단 벡터스토어(in-memory)
- top-k 리트리브 + gpt-4.1-mini로 답변 생성
"""

import os, json, math, time
from pathlib import Path
from collections import defaultdict
import tiktoken
import numpy as np
from openai import OpenAI

DATA = Path("/home/spai0308/data")
INTERIM = DATA / "interim"
MERGED = INTERIM / "rfp_text_merged.final.jsonl"

# === 파라미터 ===
CHUNK_TOKENS = 480      # 한 청크 최대 토큰수
CHUNK_OVERLAP = 60      # 중첩
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4.1-mini"
TOP_K = 5
MAX_CTX_DOCS = 1500     # 프롬프트에 실을 컨텍스트 토큰 상한(대략)

client = OpenAI()  # OPENAI_API_KEY 필요

# === 로드 ===
records=[]
with open(MERGED, encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: records.append(json.loads(line))
        except: pass

# === 청킹 ===
enc = tiktoken.get_encoding("cl100k_base")
def chunk_text(text, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    toks = enc.encode(text)
    i, n = 0, len(toks)
    out=[]
    while i < n:
        j = min(i+max_tokens, n)
        out.append(enc.decode(toks[i:j]))
        if j == n: break
        i = j - overlap
        if i < 0: i = 0
    return out

chunks=[]
for r in records:
    doc_id = r.get("doc_id","unknown")
    sp = r.get("source_path","")
    st = r.get("source_type","")
    for c in chunk_text(r["text"]):
        chunks.append({
            "doc_id": doc_id,
            "source_type": st,
            "source_path": sp,
            "text": c
        })

print(f"[chunks] total: {len(chunks)}")

# === 임베딩(배치) ===
def embed_texts(texts, batch=64):
    vecs=[]
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    return np.vstack(vecs)

chunk_texts = [c["text"] for c in chunks]
emb = embed_texts(chunk_texts, batch=64)
# 정규화(코사인 빠르게)
norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
emb = emb / norm

print("[emb] shape:", emb.shape)

# === 검색 ===
def search(query, top_k=TOP_K):
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q = np.array(q, dtype=np.float32)
    q = q / (np.linalg.norm(q)+1e-9)
    scores = emb @ q
    idx = np.argpartition(-scores, top_k)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    results = []
    tok_budget = MAX_CTX_DOCS
    ctxs=[]
    for i in idx:
        c = chunks[i]; sc=float(scores[i])
        ctxs.append(f"[doc:{c['doc_id']}] {c['text']}")
        results.append((sc, c))
    # 컨텍스트 길이 초과되면 잘라내기(대략)
    joined = ""
    for seg in ctxs:
        toks = enc.encode(seg)
        if len(toks) <= tok_budget:
            joined += seg + "\n\n"
            tok_budget -= len(toks)
        else:
            break
    return results, joined

def answer(query):
    results, ctx = search(query, top_k=TOP_K)
    prompt = f"""당신은 공공입찰 RFP 분석 도우미입니다.
다음 컨텍스트만을 근거로 질문에 정확히 답하세요. 문서에 없는 정보는 모른다고 답하세요.

[컨텍스트]
{ctx}

[질문]
{query}

요구사항, 예산, 기간, 제출마감, 제출방식, 발주기관, 문의처 등은 정확한 수치·날짜·기관명으로 명시하고,
근거가 있는 부분만 답하세요. 출처 doc_id도 함께 표기하세요."""
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role":"system","content":"You are a concise assistant for Korean public RFP QA."},
            {"role":"user","content": prompt}
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    print("\n[Q]", query)
    print("[A]", text)
    print("\n[top-k]")
    for sc, c in results:
        print(f" - score={sc:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:80]}")

# === 스모크 테스트 예시 질의 ===
queries = [
    "사업명과 발주기관, 계약방식은?",
    "총 사업기간과 제출마감일은 언제야?",
    "추정가격(또는 예산)은 얼마인가?",
    "제출 방식과 제출 서류 목록 핵심만 알려줘.",
    "담당 부서와 문의처 전화번호를 알려줘."
]

for q in queries:
    answer(q)


# In[ ]:


# q3 숫자 신뢰도 검증
get_ipython().system('grep -nR "509,506,000" /home/spai0308/data/interim/assets_*_ocr.jsonl')


# - question의 검색 출처와 신뢰도 검증 코드 결과의 출저가 동일함 --> 검증 완료

# #### 3-3) 스모크 테스트 이후 개선 코드로 재실행

# - html 순수 텍스트 포함
# - 섹션 인지 청킹
# - TOP-K=10으로 수정
# - MMR 리랭크
# - 간단 정규식 추가
# 
# 
#   --> 로 테스트 재실행

# In[ ]:


# -*- coding: utf-8 -*-
"""
RAG 스모크 패치 (5가지 적용판)
1) INCLUDE_HTML_TEXT=True: 기존 최종 JSONL에 'HTML 순수 텍스트'를ㄴ 추가로 로드하여 병합(런타임 내 dedupe)
2) 섹션 인지 청킹: 제목/번호/로만숫자/장표지 패턴으로 먼저 분할 후 토큰(또는 문자) 기반 2차 청킹
3) TOP_K=10
4) MMR 리랭크 (λ=0.35, 후보 50 → 10개)
5) 간단 정규식 추출: 사업명/발주기관/계약방식/기간/마감/예산/전화

필요 파일:
- 최종 병합본: /home/spai0308/data/interim/rfp_text_merged.final.jsonl
- HTML 경로:   /home/spai0308/data/raw/converted/html/**/*.html
"""

import os, re, json, math, time, unicodedata
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np

# ============ 경로/모델 ============
DATA    = Path("/home/spai0308/data")
INTERIM = DATA / "interim"
MERGED  = INTERIM / "rfp_text_merged.final.jsonl"
HTML_ROOT = DATA / "raw" / "converted" / "html"

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4.1-mini"

TOP_K = 10
MMR_LAMBDA = 0.35
TOP_N_CANDIDATES = 50
MAX_CTX_TOKENS = 2600  # 컨텍스트 토큰 상한

# ============ tiktoken 있는 경우/없는 경우 모두 지원 ============
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
    def trim_to_tokens(text, limit):
        toks = enc.encode(text)
        return enc.decode(toks[:limit])
    TOK = True
except Exception:
    TOK = False
    # 대략 1토큰 ≈ 4문자 가정
    def count_tokens(text): return int(len(re.sub(r"\s+"," ",text)) / 4) + 1
    def trim_to_tokens(text, limit):
        s = re.sub(r"\s+"," ",text)
        max_chars = limit * 4
        return s[:max_chars]

# ============ 정규화/유틸 ============
_ZWS = "".join(chr(c) for c in [0x200B,0x200C,0x200D,0x2060,0xFEFF])
def norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0"," ").replace("\u00A0"," ")
    for ch in _ZWS: s = s.replace(ch, "")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()

def md5_text(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def should_ignore(p: Path) -> bool:
    return (".ipynb_checkpoints" in p.parts) or any(part.startswith(".") for part in p.parts)

def extract_doc_id(path: Path) -> str:
    parts = path.parts
    if "converted" in parts:
        i = parts.index("converted")
        if i+2 < len(parts): return parts[i+2]
    return "unknown"

# ============ 섹션 인지 1차 분할 ============
# 제목/섹션 헤더로 보이는 라인 앞뒤로 경계 삽입
HEAD_PAT = re.compile(
    r"(?m)^(?:"
    r"(?:제\s*\d+\s*장)"                    # 제 1 장
    r"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+"                     # 로만 숫자(전각)
    r"|[IVX]+\."                             # Roman I./II.
    r"|\d+(?:\.\d+){0,3}"                    # 1 / 1.2 / 1.2.3.4
    r"|목\s*차|개요|요약|사업\s*개요|과업\s*개요|제안요청서"
    r")\b.*$"
)

def split_by_headings(text: str):
    t = text.replace("\r\n","\n")
    # 헤더 라인 앞에 구분자 삽입
    t = HEAD_PAT.sub(lambda m: "\n§§§ " + m.group(0), t)
    chunks = [seg.strip() for seg in t.split("\n§§§ ") if seg.strip()]
    return chunks

# ============ 2차 청킹 (토큰/문자 기준) ============
def chunk_text(text, max_tokens=480, overlap=60):
    # 문단 → 섹션 → 토큰 단위(또는 문자) 쪼갬
    out=[]
    for section in split_by_headings(text):
        if TOK:
            toks = enc.encode(section)
            i, n = 0, len(toks)
            while i < n:
                j = min(i+max_tokens, n)
                out.append(enc.decode(toks[i:j]))
                if j == n: break
                i = max(0, j - overlap)
        else:
            # 문자 기반 대체(문장 경계 고려)
            s = re.sub(r"\s+"," ", section)
            max_chars = max_tokens*4
            ov_chars  = overlap*4
            i, n = 0, len(s)
            while i < n:
                j = min(i+max_chars, n)
                k = max(s.rfind(".", i, j), s.rfind("?", i, j), s.rfind("!", i, j), s.rfind(" ", i, j))
                cut = (k+1) if k != -1 and k+1 > i else j
                out.append(s[i:cut].strip())
                if cut == n: break
                i = max(0, cut - ov_chars)
    return [c for c in out if c]

# ============ 데이터 로드 (최종 JSONL + HTML 텍스트 추가) ============
records=[]
with open(MERGED, encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try:
            r=json.loads(line)
            txt=norm(r.get("text",""))
            if txt:
                records.append({
                    "doc_id": r.get("doc_id","unknown"),
                    "source_type": r.get("source_type",""),
                    "source_path": r.get("source_path",""),
                    "text": txt
                })
        except: pass

# HTML 순수 텍스트 추가 로드
def iter_html_files(root: Path):
    for p in root.rglob("*.html"):
        if p.is_file() and not should_ignore(p):
            yield p

def html_to_text(p: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        html = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        return soup.get_text(separator="\n")
    except Exception:
        from html.parser import HTMLParser
        class _Stripper(HTMLParser):
            def __init__(self): super().__init__(); self.buf=[]
            def handle_data(self, d): 
                if d: self.buf.append(d)
        s = _Stripper()
        s.feed(p.read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(s.buf)

html_records=[]
for hp in iter_html_files(HTML_ROOT):
    doc = extract_doc_id(hp)
    txt = norm(html_to_text(hp).replace("\r\n","\n"))
    if not txt: continue
    html_records.append({
        "doc_id": doc, "source_type":"html_text", "source_path": str(hp), "text": txt
    })

# 런타임 dedupe: (doc_id, md5(text)) 기준
seen=set()
merged_all=[]
for r in records + html_records:
    key=(r["doc_id"], md5_text(r["text"]))
    if key in seen: continue
    seen.add(key); merged_all.append(r)

print(f"[load] merged.final: {len(records)} | +html_text: {len(html_records)} | after-dedupe total: {len(merged_all)}")

# ============ 청킹 ============
chunks=[]
for r in merged_all:
    for c in chunk_text(r["text"], max_tokens=480, overlap=60):
        chunks.append({
            "doc_id": r["doc_id"],
            "source_type": r["source_type"],
            "source_path": r["source_path"],
            "text": c
        })
print(f"[chunks] total: {len(chunks)}")

# ============ 임베딩 ============
from openai import OpenAI
client = OpenAI()

def embed_texts(texts, batch=64):
    vecs=[]
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch])
        vecs.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    return np.vstack(vecs)

chunk_texts=[c["text"] for c in chunks]
emb = embed_texts(chunk_texts, batch=64)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
print("[emb]", emb.shape)

# ============ 검색 + MMR ============
def search_mmr(query, top_n=TOP_N_CANDIDATES, top_k=TOP_K, lambda_mult=MMR_LAMBDA):
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q = np.array(q, dtype=np.float32); q = q / (np.linalg.norm(q)+1e-9)
    sims = emb @ q
    cand_idx = np.argpartition(-sims, min(top_n, len(sims)-1))[:top_n]
    cand_idx = cand_idx[np.argsort(-sims[cand_idx])]
    cand_vecs = emb[cand_idx]

    selected=[]; selected_idx=[]
    if len(cand_idx)==0: return []
    selected_idx.append(cand_idx[0])
    selected.append(cand_vecs[0])

    for _ in range(1, min(top_k, len(cand_idx))):
        rest = cand_vecs[len(selected):]  # 남은 후보
        rest_idx = cand_idx[len(selected):]
        # 유사도 계산
        sim_to_q = sims[rest_idx]
        sim_to_sel = np.max(rest @ np.stack(selected, axis=1), axis=1) if selected else np.zeros(len(rest))
        mmr_scores = lambda_mult * sim_to_q - (1-lambda_mult) * sim_to_sel
        j = np.argmax(mmr_scores)
        selected_idx.append(rest_idx[j])
        selected.append(rest[j])
        # 선택한 것을 cand_vecs의 앞에 있다고 가정하고 다음 루프에서 제외
        cand_vecs = np.concatenate([cand_vecs[:len(selected)], rest[np.arange(len(rest))!=j]], axis=0)
        cand_idx  = np.concatenate([cand_idx[:len(selected)],  rest_idx[np.arange(len(rest))!=j]], axis=0)

    idx = np.array(selected_idx)
    # 유사도 순으로 재정렬(선택셋 내부)
    idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[i]), chunks[i]) for i in idx]

def build_context(results, token_budget=MAX_CTX_TOKENS):
    segments=[]; used=0
    for sc, c in results:
        seg = f"[doc:{c['doc_id']}] {c['text']}".strip()
        t = count_tokens(seg)
        if used + t > token_budget: break
        segments.append(seg); used += t
    return "\n\n".join(segments)

# ============ 간단 정규식 추출 ============
K_DATE = r"(?:\d{4}[.\-/년]\s*\d{1,2}[.\-/월]?\s*\d{1,2}\s*(?:일)?)"
K_PERIOD = rf"(?:계약\s*후\s*\d+\s*개월|{K_DATE}\s*~\s*{K_DATE})"
K_MONEY = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|\d+)\s*원"
K_PHONE = r"0\d{1,2}-\d{3,4}-\d{4}"

EXTRACTORS = OrderedDict([
    ("사업명",      [re.compile(r"(?:사\s*업\s*명|과업명|사업명)\s*[:：]?\s*([^\n]{3,80})")]),
    ("발주기관",    [re.compile(r"(?:발주기관|주관기관|수요기관|발주처)\s*[:：]?\s*([^\n]{2,80})")]),
    ("계약방식",    [
        re.compile(r"(?:계약방식|계약 방법)\s*[:：]?\s*([^\n]{2,40})"),
        re.compile(r"(협상에\s*의한\s*계약|제한경쟁입찰|수의계약|일반경쟁|지명경쟁)")
    ]),
    ("사업기간",    [re.compile(rf"(?:사업기간|과업기간)\s*[:：]?\s*({K_PERIOD})")]),
    ("제출마감일",  [re.compile(rf"(?:제출마감|접수마감|제안서\s*제출\s*마감)\s*[:：]?\s*({K_DATE})")]),
    ("예산",        [re.compile(rf"(?:추정가격|예산|추정\s*금액)\s*[:：]?\s*({K_MONEY})")]),
    ("전화",        [re.compile(K_PHONE)])
])

def extract_fields_from_blocks(blocks):
    """blocks: [(doc_id, text)]"""
    found={}
    where={}
    for doc, txt in blocks:
        t = norm(txt)
        for key, patterns in EXTRACTORS.items():
            if key in found: continue
            for pat in patterns:
                m = pat.search(t)
                if m:
                    val = m.group(1) if m.groups() else m.group(0)
                    val = norm(val)
                    found[key] = val
                    where[key] = doc
                    break
    return found, where

# ============ Q&A ============
def answer(query):
    results = search_mmr(query, top_n=TOP_N_CANDIDATES, top_k=TOP_K, lambda_mult=MMR_LAMBDA)
    ctx_blocks = [(c["doc_id"], c["text"]) for _, c in results]
    ctx_text = build_context(results, token_budget=MAX_CTX_TOKENS)

    # 1) 정규식 추출 시도
    extracted, where = extract_fields_from_blocks(ctx_blocks)

    # 질문 키워드 기준으로 골라 답변 구성
    keymap = {
        "사업명": ["사업명","과업명"],
        "발주기관": ["발주기관","주관기관","수요기관","발주처"],
        "계약방식": ["계약","계약방식","협상","경쟁","수의"],
        "사업기간": ["기간","사업기간","과업기간"],
        "제출마감일": ["마감","제출","접수"],
        "예산": ["예산","추정가격","금액","비용"],
        "전화": ["전화","문의","연락"]
    }
    want = set()
    ql = query.lower()
    for k, kws in keymap.items():
        if any(kw in ql for kw in [w.lower() for w in kws]):
            want.add(k)

    # 추출 성공 & 질문이 필드성일 때는 규칙 기반으로 우선 답변
    if extracted and (want or any(k in ql for k in ["사업","발주","계약","기간","마감","예산","전화"])):
        lines=[]
        for k in ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]:
            if k in extracted and (not want or k in want):
                src = where.get(k,"-")
                lines.append(f"- {k}: {extracted[k]} (doc:{src})")
        if lines:
            print("\n[Q]", query)
            print("[A]\n" + "\n".join(lines))
            print("\n[top-k]")
            for sc, c in results:
                print(f" - score={sc:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:90]}")
            return

    # 2) 생성 모델로 답변 (컨텍스트 포함)
    prompt = f"""당신은 공공입찰 RFP 분석 도우미입니다.
다음 컨텍스트만을 근거로 질문에 정확히 답하세요. 문서에 없는 정보는 모른다고 답하세요.

[컨텍스트]
{trim_to_tokens(ctx_text, MAX_CTX_TOKENS)}

[질문]
{query}

요구사항, 예산, 기간, 제출마감, 제출방식, 발주기관, 문의처 등은 정확한 수치·날짜·기관명으로 명시하고,
근거가 있는 부분만 답하세요. 각 항목마다 출처 doc_id를 함께 적으세요."""
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role":"system","content":"You are a concise assistant for Korean public RFP QA."},
            {"role":"user","content": prompt}
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    print("\n[Q]", query)
    print("[A]", text)
    print("\n[top-k]")
    for sc, c in results:
        print(f" - score={sc:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:90]}")

# ======= 테스트 질의 =======
tests = [
    "사업명과 발주기관, 계약방식은?",
    "총 사업기간과 제출마감일은 언제야?",
    "추정가격(또는 예산)은 얼마인가?",
    "제출 방식과 제출 서류 목록 핵심만 알려줘.",
    "담당 부서와 문의처 전화번호를 알려줘."
]
for q in tests:
    answer(q)


# #### 3-4) Multi-Query Retrieval (질문 분해 + 통합 검색 + MMR 재랭크) 추가하여 retry

# - 이전 패치 셀에서 만들어둔 ``client, emb, chunks, build_context, extract_fields_from_blocks`` 등을 그대로 재사용함

# In[ ]:


# -*- coding: utf-8 -*-
# Multi-Query: 하나의 질문을 여러 개의 하위 질의로 확장 → 각 질의로 후보 모으고 → 통합 풀에 MMR 재랭크

import re, json, numpy as np

MQ_MODEL = "gpt-4.1-mini"     # 또는 "gpt-4.1-nano"
NUM_SUB_QUERIES = 5
PER_QUERY_CANDIDATES = 30     # 각 서브질의당 가져올 후보 개수
FINAL_TOP_K = 10
MMR_LAMBDA_MQ = 0.35

# 안전: 이전 셀 함수가 없을 때 대체 build_context 준비
def _fallback_count_tokens(text):
    try:
        return count_tokens(text)
    except NameError:
        return int(len(re.sub(r"\s+"," ", text)) / 4) + 1

def _fallback_build_context(results, token_budget=2600):
    try:
        return build_context(results, token_budget=token_budget)
    except NameError:
        segs=[]; used=0
        for sc, c in results:
            seg = f"[doc:{c['doc_id']}] {c['text']}"
            t = _fallback_count_tokens(seg)
            if used + t > token_budget: break
            segs.append(seg); used += t
        return "\n\n".join(segs)

# 간단 후보 검색(top-N) — 임베딩 코사인 점수 상위 N
# 안전: 빈 후보 처리 추가
def search_candidates(query, top_n=30):
    q = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    q = np.array(q, dtype=np.float32); q = q / (np.linalg.norm(q)+1e-9)
    sims = emb @ q
    n = min(top_n, len(sims))
    if n <= 0:
        return []  # ← 빈 문서/임베딩 방어
    idx = np.argpartition(-sims, n-1)[:n]
    idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[i]), chunks[i], i) for i in idx]

# MMR 재랭크
def mmr_rerank(indices, sims, lambda_mult=0.35, top_k=10):
    """
    indices: (M,) np.ndarray[int]  — chunks에서의 인덱스
    sims   : (M,) np.ndarray[float] — 쿼리와의 코사인 유사도
    반환   : [(score, chunk_dict), ...] 길이 <= top_k
    """
    # 넘파이 배열은 truth value 비교 금지
    if isinstance(indices, np.ndarray):
        if indices.size == 0:
            return []
    elif not indices:  # 리스트 등
        return []

    # 유사도 내림차순으로 풀 정렬
    order = np.argsort(-sims)
    pool_idx  = indices[order]         # (M,)
    pool_sims = sims[order]            # (M,)
    cand_vecs = emb[pool_idx]          # (M, D) — 미리 벡터 모음

    # 첫번째는 최고 유사도 선택
    selected_pos = [0]                 # pool상의 위치 인덱스 보관(전역 인덱스 아님)
    M = len(pool_idx)
    if M == 1:
        i = pool_idx[0]
        return [(float(pool_sims[0]), chunks[i])]

    # 반복 선택 (MMR)
    for _ in range(1, min(top_k, M)):
        rest_pos = [p for p in range(M) if p not in selected_pos]
        if not rest_pos:
            break
        # 쿼리 유사도(이미 정렬된 pool_sims에서 위치 선택)
        sim_to_q = pool_sims[rest_pos]                    # (R,)

        # 다양성 페널티: 현재 선택된 것들과의 최대 유사도
        rest_vecs = cand_vecs[rest_pos]                   # (R, D)
        sel_vecs  = cand_vecs[selected_pos]               # (S, D)
        # (R, S) → 각 rest가 선택셋과 얼마나 비슷한지
        sim_matrix = rest_vecs @ sel_vecs.T
        max_sim_to_sel = np.max(sim_matrix, axis=1)       # (R,)

        mmr = lambda_mult * sim_to_q - (1 - lambda_mult) * max_sim_to_sel
        j_rel = int(np.argmax(mmr))
        chosen_pos = rest_pos[j_rel]
        selected_pos.append(chosen_pos)

        if len(selected_pos) >= top_k:
            break

    # 선택된 것들을 '원래 쿼리-유사도' 기준으로 정렬해 반환
    final = sorted(
        [(float(pool_sims[p]), chunks[int(pool_idx[p])]) for p in selected_pos],
        key=lambda x: -x[0]
    )
    return final

# 질의 확장 (JSON 파싱 내성 포함)
def expand_queries(query, k=NUM_SUB_QUERIES, model=MQ_MODEL):
    sys = "You rewrite a Korean RFP search query into diverse sub-queries. Output ONLY JSON: {\"queries\": [..]}."
    user = f"""원 질문: "{query}"
제출 형식: JSON 한 줄. 예) {{"queries":["...","..."]}}
지침:
- 동의어/축약/필드지향(사업명, 발주기관, 계약방식, 기간, 마감, 예산, 문의처 등)으로 다양화
- 중복/의역 반복 금지
- 한국어 위주, 필요시 숫자/기호 포함
개수: {k}개"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system", "content":sys},
                  {"role":"user", "content":user}],
        temperature=0.7,
        max_tokens=300
    )
    txt = resp.choices[0].message.content.strip()
    # JSON 파싱
    import json, ast
    queries=[]
    try:
        data=json.loads(txt)
        queries=list(dict.fromkeys(data.get("queries",[])))
    except Exception:
        # 대괄호 부분만 추출
        m=re.search(r"\[.*\]", txt, flags=re.S)
        if m:
            try:
                arr=ast.literal_eval(m.group(0))
                queries=list(dict.fromkeys([str(x) for x in arr]))
            except Exception:
                pass
    # fallback: 원 질문 1개라도
    if not queries:
        queries=[query]
    return queries[:k]

def search_multiquery(query, num_sub=NUM_SUB_QUERIES, per_q=PER_QUERY_CANDIDATES, final_k=FINAL_TOP_K, lambda_m=MMR_LAMBDA_MQ):
    subs = expand_queries(query, k=num_sub)
    pool_idx=set(); pool_scores=[]
    for sq in subs:
        cands = search_candidates(sq, top_n=per_q)
        for s, c, i in cands:
            if i not in pool_idx:
                pool_idx.add(i)
                pool_scores.append((i, s))
            else:
                # 동일 chunk가 다른 질의에서도 뜨면 점수 최대값 유지
                for k in range(len(pool_scores)):
                    if pool_scores[k][0]==i:
                        pool_scores[k]=(i, max(pool_scores[k][1], s))
                        break
    if not pool_scores: return []
    idxs = np.array([i for i,_ in pool_scores], dtype=np.int32)
    sims = np.array([s for _,s in pool_scores], dtype=np.float32)
    ranked = mmr_rerank(idxs, sims, lambda_mult=lambda_m, top_k=final_k)
    return ranked

def answer_mq(query):
    results = search_multiquery(query)
    # 컨텍스트 구성
    try:
        ctx = build_context(results, token_budget=2600)
    except NameError:
        ctx = _fallback_build_context(results, token_budget=2600)

    # 규칙 기반 필드 추출이 정의돼 있다면 먼저 시도
    try:
        ctx_blocks = [(c["doc_id"], c["text"]) for _, c in results]
        extracted, where = extract_fields_from_blocks(ctx_blocks)
    except NameError:
        extracted, where = {}, {}

    used_rules=False
    if extracted:
        key_order = ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]
        lines=[]
        for k in key_order:
            if k in extracted:
                lines.append(f"- {k}: {extracted[k]} (doc:{where.get(k,'-')})")
        if lines:
            print("\n[Q][MQ]", query)
            print("[A]\n" + "\n".join(lines))
            used_rules=True

    if not used_rules:
        # 생성 모델로 답변
        prompt = f"""아래 컨텍스트만 근거로 질문에 답하세요. 없으면 모른다고 답하세요.

[컨텍스트]
{ctx}

[질문]
{query}

각 항목은 정확한 수치·날짜·기관명으로, 출처 doc_id를 같이 적으세요."""
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"system","content":"You are a concise assistant for Korean public RFP QA."},
                      {"role":"user","content":prompt}],
            temperature=0.0
        )
        print("\n[Q][MQ]", query)
        print("[A]", resp.choices[0].message.content.strip())

    print("\n[top-k | Multi-Query + MMR]")
    for sc, c in results:
        print(f" - score={sc:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:90]}")

# 예시 실행
for q in [
    "사업명과 발주기관, 계약방식은?",
    "총 사업기간과 제출마감일은 언제야?",
    "추정가격(또는 예산)은 얼마인가?",
    "담당 부서와 문의처 전화번호를 알려줘."
]:
    answer_mq(q)


# #### 3-5) Cross-Encoder Re-ranker (생성모델로 후보 문단 점수화 → 재정렬) 추가하여 테스트

# In[ ]:


# -*- coding: utf-8 -*-
# Cross-Encoder style reranker: 상위 후보들을 gpt-4.1-***로 점수화(0~1) → 재정렬

import re, json, numpy as np

CE_MODEL = "gpt-4.1-nano"     # 빠른 평가용(저비용). 더 정확히 보려면 "gpt-4.1-mini".
CAND_TOP_N = 50               # 1차 임베딩 검색에서 모을 후보 수
FINAL_TOP_K_CE = 10

def search_candidates(query, top_n=CAND_TOP_N):
    q = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    q = np.array(q, dtype=np.float32); q = q / (np.linalg.norm(q)+1e-9)
    sims = emb @ q
    n = min(top_n, len(sims))
    idx = np.argpartition(-sims, n-1)[:n]
    idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[i]), chunks[i], i) for i in idx]

def ce_score_one(query, passage, model=CE_MODEL):
    sys = "You are a strict ranking model. Output ONLY JSON like {\"score\": 0.xx}."
    user = f"""Query: {query}

Candidate passage:ㄴ
\"\"\"{passage.strip()[:2000]}\"\"\"

지침:
- 질의의 답(수치/기관/날짜/정의/목록 등)이 직접적으로 포함되면 0.95~1.0
- 맥락은 있으나 직접 답이 없으면 0.3~0.7
- 거의 무관하면 0.0~0.2
- 오직 JSON 한 줄만 출력: {{"score": <0~1 float>}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":user}],
        temperature=0.0,
        max_tokens=20
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"[-+]?\d*\.?\d+", txt)
    try:
        s = float(m.group(0))
        if s<0: s=0.0
        if s>1: s=1.0
        return s
    except Exception:
        return 0.0

def rerank_cross_encoder(query, top_n=CAND_TOP_N, final_k=FINAL_TOP_K_CE, model=CE_MODEL):
    cands = search_candidates(query, top_n=top_n)
    scored=[]
    for s, c, _ in cands:
        ce = ce_score_one(query, c["text"], model=model)
        scored.append((ce, s, c))
    # CE 점수 우선 정렬, 동점시 임베딩 점수 보조
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [(float(ce), float(s), c) for (ce, s, c) in scored[:final_k]]

def build_context_ce(ranked, token_budget=2600):
    segs=[]; used=0
    for ce, s, c in ranked:
        seg = f"[doc:{c['doc_id']}] {c['text']}"
        try:
            t = count_tokens(seg)
        except NameError:
            t = int(len(re.sub(r'\s+',' ',seg))/4)+1
        if used + t > token_budget: break
        segs.append(seg); used+=t
    return "\n\n".join(segs)

def answer_ce(query):
    ranked = rerank_cross_encoder(query)
    ctx = build_context_ce(ranked, token_budget=2600)

    # 규칙 기반 추출이 있으면 먼저 시도
    try:
        ctx_blocks = [(c["doc_id"], c["text"]) for _,_, c in ranked]
        extracted, where = extract_fields_from_blocks(ctx_blocks)
    except NameError:
        extracted, where = {}, {}

    if extracted:
        key_order = ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]
        lines=[]
        for k in key_order:
            if k in extracted:
                lines.append(f"- {k}: {extracted[k]} (doc:{where.get(k,'-')})")
        if lines:
            print("\n[Q][CE]", query)
            print("[A]\n" + "\n".join(lines))
            print("\n[top-k | Cross-Encoder reranked]")
            for ce, s, c in ranked:
                print(f" - CE={ce:.3f} emb={s:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:90]}")
            return

    # 생성 모델 답변
    prompt = f"""아래 컨텍스트만 근거로 질문에 답하세요. 없으면 모른다고 답하세요.

[컨텍스트]
{ctx}

[질문]
{query}

각 항목은 정확한 수치·날짜·기관명으로, 출처 doc_id를 같이 적으세요."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"system","content":"You are a concise assistant for Korean public RFP QA."},
                  {"role":"user","content":prompt}],
        temperature=0.0
    )
    print("\n[Q][CE]", query)
    print("[A]", resp.choices[0].message.content.strip())

    print("\n[top-k | Cross-Encoder reranked]")
    for ce, s, c in ranked:
        print(f" - CE={ce:.3f} emb={s:.3f} doc={c['doc_id']} type={c['source_type']} path={c['source_path'][:90]}")

# 예시 실행
for q in [
    "사업명과 발주기관, 계약방식은?",
    "총 사업기간과 제출마감일은 언제야?",
    "추정가격(또는 예산)은 얼마인가?",
    "담당 부서와 문의처 전화번호를 알려줘."
]:
    answer_ce(q)


# ### 3-6) 3가지 검색 결과 평가 비교(Retrieval Evalution): MMR 리트리브(패치판) VS Multi-Qeury VS Cross-Encoder 리랭커

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



# In[ ]:


# -*- coding: utf-8 -*-
# 세 가지 접근법(MMR / Multi-Query / Cross-Encoder) 결과 비교 셀

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt

# 안전한 토큰 카운터(없으면 문자기반 대체)
def _count_tokens_safe(text):
    try:
        return count_tokens(text)
    except NameError:
        return int(len(re.sub(r"\s+"," ", text))/4) + 1

# 공통: top-k 블록을 [(doc_id, text)] 형태로 만들기
def _to_blocks(results, kind):
    """results:
       kind='mmr' or 'mq' : [(emb_score, chunk_dict), ...]
       kind='ce'          : [(ce_score, emb_score, chunk_dict), ...]
    """
    blocks=[]
    scores=[]
    src_types=[]
    docs=[]
    if kind in ("mmr","mq"):
        for s, c in results:
            blocks.append((c["doc_id"], c["text"]))
            scores.append(float(s))
            src_types.append(c.get("source_type",""))
            docs.append(c["doc_id"])
    elif kind == "ce":
        for ce, s, c in results:
            blocks.append((c["doc_id"], c["text"]))
            # CE는 주점수로 ce, 보조로 emb_score
            scores.append(float(ce))
            src_types.append(c.get("source_type",""))
            docs.append(c["doc_id"])
    return blocks, np.array(scores, dtype=np.float32), src_types, docs

# 간단 요약 문자열
def _summarize_src_types(src_types):
    from collections import Counter
    cnt = Counter(src_types)
    parts = [f"{k}:{v}" for k,v in cnt.most_common()]
    return ", ".join(parts)

def _top_docs_str(docs, k=3):
    seen=[]; out=[]
    for d in docs:
        if d in seen: continue
        seen.append(d); out.append(d)
        if len(out)>=k: break
    return " | ".join(out)

# 평가 함수
def evaluate_one(query, approach):
    """approach: 'mmr' / 'mq' / 'ce'"""
    if approach == "mmr":
        if "search_mmr" not in globals():
            raise RuntimeError("search_mmr 가 메모리에 없습니다(패치 셀 먼저 실행).")
        results = search_mmr(query)
        kind = "mmr"
    elif approach == "mq":
        if "search_multiquery" not in globals():
            raise RuntimeError("search_multiquery 가 메모리에 없습니다(Multi-Query 셀 먼저 실행).")
        results = search_multiquery(query)
        kind = "mq"
    elif approach == "ce":
        if "rerank_cross_encoder" not in globals():
            raise RuntimeError("rerank_cross_encoder 가 메모리에 없습니다(Cross-Encoder 셀 먼저 실행).")
        ranked = rerank_cross_encoder(query)  # [(ce, emb, chunk), ...]
        # CE는 위 함수가 상위 K만 반환하므로 그대로 사용
        results = [(ce, c) for (ce, emb_s, c) in ranked]  # 형식을 맞추기 위해 변환
        # 다시 원형으로 사용할 수 있도록 아래에서 별 처리
        kind = "ce"
        # CE용으로 blocks/scores는 ce 점수 기준으로 구성
        blocks = [(c["doc_id"], c["text"]) for (ce, c) in results]
        scores = np.array([float(ce) for (ce, c) in results], dtype=np.float32)
        src_types = [c.get("source_type","") for (ce, c) in results]
        docs = [c["doc_id"] for (ce, c) in results]
    else:
        raise ValueError("approach must be one of: 'mmr','mq','ce'")

    # kind 별 표준화
    if approach != "ce":
        blocks, scores, src_types, docs = _to_blocks(results, kind)

    # 규칙 기반 필드 추출
    if "extract_fields_from_blocks" in globals():
        extracted, where = extract_fields_from_blocks(blocks)
    else:
        extracted, where = {}, {}

    fields = ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]
    flags = {f: (f in extracted) for f in fields}
    found_cnt = sum(flags.values())

    # top-k 통계
    topk = len(blocks)
    top1 = float(scores[0]) if topk>0 else 0.0
    mean_s = float(scores.mean()) if topk>0 else 0.0
    uniq_docs = len(set(docs))
    src_summary = _summarize_src_types(src_types)
    docs_summary = _top_docs_str(docs, k=3)
    ctx_tokens = sum(_count_tokens_safe(f"[doc:{d}] {t}") for d,t in blocks)

    row = {
        "approach": approach,
        "query": query,
        "topk": topk,
        "top1_score": round(top1,4),
        "mean_score": round(mean_s,4),
        "uniq_docs": uniq_docs,
        "src_types(topk)": src_summary,
        "top_docs": docs_summary,
        "ctx_tokens_sum": ctx_tokens,
        "fields_found": found_cnt
    }
    for f in fields:
        row[f] = int(flags[f])
        row[f+"_doc"] = where.get(f,"")
        row[f+"_value"] = extracted.get(f,"")
    return row

def evaluate_batch(queries):
    rows=[]
    for q in queries:
        for ap in ["mmr","mq","ce"]:
            try:
                rows.append(evaluate_one(q, ap))
            except Exception as e:
                rows.append({
                    "approach": ap, "query": q, "error": str(e),
                    "topk":0,"top1_score":0,"mean_score":0,"uniq_docs":0,
                    "src_types(topk)":"", "top_docs":"", "ctx_tokens_sum":0, "fields_found":0,
                    "사업명":0,"발주기관":0,"계약방식":0,"사업기간":0,"제출마감일":0,"예산":0,"전화":0
                })
    df = pd.DataFrame(rows)
    # 요약
    field_cols = ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]
    agg = {
        "fields_found": "mean",
        "top1_score": "mean",
        "mean_score": "mean",
        "uniq_docs": "mean",
        "ctx_tokens_sum": "mean",
    }
    for f in field_cols:
        agg[f] = "mean"  # 커버리지율(0~1)의 평균
    df_summary = df.groupby("approach").agg(agg).reset_index()
    # 보기 좋게 퍼센트로
    for f in field_cols + ["fields_found"]:
        if f in df_summary.columns:
            df_summary[f] = (df_summary[f]*100).round(1)
    df_summary["top1_score"] = df_summary["top1_score"].round(3)
    df_summary["mean_score"] = df_summary["mean_score"].round(3)
    df_summary["uniq_docs"] = df_summary["uniq_docs"].round(2)
    df_summary["ctx_tokens_sum"] = df_summary["ctx_tokens_sum"].round(0).astype(int)
    return df, df_summary

# ===== 질의 세트 정의(원하면 바꾸기) =====
queries = [
    "사업명과 발주기관, 계약방식은?",
    "총 사업기간과 제출마감일은 언제야?",
    "추정가격(또는 예산)은 얼마인가?",
    "제출 방식과 제출 서류 목록 핵심만 알려줘.",
    "담당 부서와 문의처 전화번호를 알려줘."
]

df_detail, df_summary = evaluate_batch(queries)

print("=== 상세 결과 (일부 표시) ===")
print(df_detail.head(9).to_string(index=False))  # 첫 9행

print("\n=== 접근법별 요약 ===")
print(df_summary.to_string(index=False))

# ===== 그래프: 전체 필드 커버리지(평균) =====
field_cols = ["사업명","발주기관","계약방식","사업기간","제출마감일","예산","전화"]
plt.figure()
for f in field_cols:
    if f not in df_summary.columns:
        raise RuntimeError(f"Column missing: {f}")
plt.bar(df_summary["approach"], df_summary["fields_found"])
plt.title("전체 필드 커버리지(평균, %)")
plt.xlabel("접근법")
plt.ylabel("커버된 필드 개수 비율(%)")
plt.show()

# ===== 그래프: 필드별 커버리지(평균) =====
plt.figure()
width = 0.25
x = np.arange(len(field_cols))
approaches = df_summary["approach"].tolist()
for i, ap in enumerate(approaches):
    vals = [float(df_summary.loc[df_summary["approach"]==ap, f]) for f in field_cols]
    plt.bar(x + i*width, vals, width, label=ap)
plt.xticks(x + width, field_cols, rotation=0)
plt.title("필드별 커버리지(평균, %)")
plt.xlabel("필드")
plt.ylabel("커버리지(%)")
plt.legend()
plt.show()


# ### 지표 설명
# 
# - `df_summary`
#     - **fields_found(%)**: 질문당 7개 필드 중 몇 개를 평균적으로 찾았는지 비율
#     - **각 필드 컬럼(%)**: 해당 필드를 찾아낸 질문 비율
#     - **uniq_docs**: top-k 안의 고유 문서 수 평균(높을수록 다양성↑)
#     - **top1/mean_score**: 1위/평균 유사도(혹은 CE 점수) – 상대 비교용
#     - **ctx_tokens_sum**: 컨텍스트 토큰 총량 평균(비용/컨텍스트 길이 감각)
# - `df_detail`
#     - 각 질의 × 접근법에 대해 **어떤 필드를 찾았는지**, **그 값/출처(doc_id)**, **top-k 소스 타입 분포**, **상위 문서들** 등을 확인.

# In[ ]:


# df_detail, df_summary 엑셀(시트 2개로) 저장

# -*- coding: utf-8 -*-
# df_detail, df_summary가 메모리에 있다고 가정합니다.

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 저장 경로
INTERIM = Path("/home/spai0308/data/interim")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_path = INTERIM / f"rag_eval_comparison_{ts}.xlsx"
csv_detail = INTERIM / f"rag_eval_detail_{ts}.csv"
csv_summary = INTERIM / f"rag_eval_summary_{ts}.csv"

# 안전 체크
for name in ("df_detail", "df_summary"):
    if name not in globals():
        raise RuntimeError(f"{name} 변수가 메모리에 없습니다. 비교 셀을 먼저 실행하세요.")

# 엑셀 저장 (xlsxwriter 우선, 없으면 openpyxl)
engine = None
try:
    import xlsxwriter  # noqa
    engine = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa
        engine = "openpyxl"
    except Exception:
        engine = None

if engine is None:
    print("[WARN] xlsxwriter/openpyxl 미설치 → CSV로만 저장합니다.")
    df_detail.to_csv(csv_detail, index=False)
    df_summary.to_csv(csv_summary, index=False)
    print(f"[OK] CSV 저장:\n - {csv_detail}\n - {csv_summary}")
else:
    with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
        # 시트 쓰기
        df_detail.to_excel(writer, sheet_name="detail", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

        # 서식 도구
        if engine == "xlsxwriter":
            wb = writer.book
            ws_d = writer.sheets["detail"]
            ws_s = writer.sheets["summary"]

            # 서식
            fmt_wrap   = wb.add_format({"text_wrap": True, "valign":"top"})
            fmt_pct1   = wb.add_format({"num_format": "0.0"})
            fmt_float3 = wb.add_format({"num_format": "0.000"})
            fmt_int    = wb.add_format({"num_format": "0"})

            # 열 너비 자동 맞춤 + 줄바꿈 적용 (detail)
            def autofit(ws, df, wrap_cols=None, max_width=80):
                for col_idx, col in enumerate(df.columns):
                    # 헤더/값 기준 최대 길이
                    sample = [str(col)] + [str(x) for x in df[col].astype(str).head(500)]
                    width = min(max(len(s) for s in sample) + 2, max_width)
                    if wrap_cols and col in wrap_cols:
                        ws.set_column(col_idx, col_idx, width, fmt_wrap)
                    else:
                        ws.set_column(col_idx, col_idx, width)

            wrap_cols_detail = [c for c in df_detail.columns if any(
                key in c for key in ["src_types", "top_docs", "_value"]
            )]
            autofit(ws_d, df_detail, wrap_cols=wrap_cols_detail, max_width=100)
            autofit(ws_s, df_summary, wrap_cols=None, max_width=40)

            # 첫 행 고정
            ws_d.freeze_panes(1, 0)
            ws_s.freeze_panes(1, 0)

            # summary 형식: 퍼센트/점수/정수
            pct_cols   = ["fields_found", "사업명", "발주기관", "계약방식", "사업기간", "제출마감일", "예산", "전화"]
            float_cols = ["top1_score", "mean_score"]
            int_cols   = ["uniq_docs", "ctx_tokens_sum"]

            # 컬럼 인덱스 찾기 헬퍼
            def col_index(df, name):
                try:
                    return list(df.columns).index(name)
                except ValueError:
                    return None

            # 퍼센트/숫자 서식 적용
            for name in pct_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_pct1)

            for name in float_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_float3)

            for name in int_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_int)

            # README 시트(메타데이터)
            ws_r = wb.add_worksheet("README")
            meta = {
                "timestamp": ts,
                "engine": engine,
                "rows_detail": len(df_detail),
                "rows_summary": len(df_summary),
                # 아래 파라미터들은 있으면 기록
                "TOP_K": globals().get("TOP_K", "-"),
                "MMR_LAMBDA": globals().get("MMR_LAMBDA", "-"),
                "NUM_SUB_QUERIES": globals().get("NUM_SUB_QUERIES", "-"),
            }
            ws_r.write(0, 0, "key"); ws_r.write(0, 1, "value")
            for i, (k,v) in enumerate(meta.items(), start=1):
                ws_r.write(i, 0, str(k))
                ws_r.write(i, 1, str(v))

        else:
            # openpyxl 엔진: 기본 저장 + 첫 행 고정만 처리
            ws_d = writer.sheets["detail"]
            ws_s = writer.sheets["summary"]
            try:
                ws_d.freeze_panes = "A2"
                ws_s.freeze_panes = "A2"
            except Exception:
                pass

    print(f"[OK] 엑셀 저장: {xlsx_path}")

    # CSV도 같이 남기고 싶다면 주석 해제
    df_detail.to_csv(csv_detail, index=False)
    df_summary.to_csv(csv_summary, index=False)
    print(f"[OK] CSV 저장:\n - {csv_detail}\n - {csv_summary}")



# - `xlsxwriter`가 있으면 **열 너비 자동 맞춤 + 줄바꿈 + 서식**까지 적용됨
# - `openpyxl`만 있으면 기본 저장/첫 행 고정만 됩니다(서식은 제한).

# In[78]:


# 파일 삭제 후, xlswrtier/openpyxl 설치부터 하기

# !rm /home/spai0308/data/interim/rag_eval_detail_20250922_143758.csv
# !rm /home/spai0308/data/interim/rag_eval_summary_20250922_143758.csv


# In[ ]:


# !ls -l /home/spai0308/data/interim/rag_eval_detail_20250922_143758.csv
# !ls -l /home/spai0308/data/interim/rag_eval_summary_20250922_143758.csv


# In[ ]:


# # xmlwrtier/oepnpyxl 설치 : 터미널에서

# # 사전 설치 : xlswrtier, openpyxl
# # 지금 파이썬/피ップ이 뭔지 확인
# which python
# python -V
# python -m pip -V

# # 설치 (둘 다 설치해두면 편함)
# # 1) pip 최신화(가끔 오래된 pip이면 패키지 인식 못함)
# python -m pip install -U pip setuptools wheel
# # 2) 정확한 패키지명으로 설치
# python -m pip install -U xlsxwriter openpyxl

# # 설치 후 확인
# python -c "import xlsxwriter, openpyxl, sys; print('ok:', sys.executable)"


# In[80]:


# df_detail, df_summary 엑셀(시트 2개로) 저장 (xlsxwriter/openpyxl 설치 이후)

# -*- coding: utf-8 -*-
# df_detail, df_summary가 메모리에 있다고 가정합니다.

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 저장 경로
INTERIM = Path("/home/spai0308/data/interim")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_path = INTERIM / f"rag_eval_comparison_{ts}.xlsx"
csv_detail = INTERIM / f"rag_eval_detail_{ts}.csv"
csv_summary = INTERIM / f"rag_eval_summary_{ts}.csv"

# 안전 체크
for name in ("df_detail", "df_summary"):
    if name not in globals():
        raise RuntimeError(f"{name} 변수가 메모리에 없습니다. 비교 셀을 먼저 실행하세요.")

# 엑셀 저장 (xlsxwriter 우선, 없으면 openpyxl)
engine = None
try:
    import xlsxwriter  # noqa
    engine = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa
        engine = "openpyxl"
    except Exception:
        engine = None

if engine is None:
    print("[WARN] xlsxwriter/openpyxl 미설치 → CSV로만 저장합니다.")
    df_detail.to_csv(csv_detail, index=False)
    df_summary.to_csv(csv_summary, index=False)
    print(f"[OK] CSV 저장:\n - {csv_detail}\n - {csv_summary}")
else:
    with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
        # 시트 쓰기
        df_detail.to_excel(writer, sheet_name="detail", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

        # 서식 도구
        if engine == "xlsxwriter":
            wb = writer.book
            ws_d = writer.sheets["detail"]
            ws_s = writer.sheets["summary"]

            # 서식
            fmt_wrap   = wb.add_format({"text_wrap": True, "valign":"top"})
            fmt_pct1   = wb.add_format({"num_format": "0.0"})
            fmt_float3 = wb.add_format({"num_format": "0.000"})
            fmt_int    = wb.add_format({"num_format": "0"})

            # 열 너비 자동 맞춤 + 줄바꿈 적용 (detail)
            def autofit(ws, df, wrap_cols=None, max_width=80):
                for col_idx, col in enumerate(df.columns):
                    # 헤더/값 기준 최대 길이
                    sample = [str(col)] + [str(x) for x in df[col].astype(str).head(500)]
                    width = min(max(len(s) for s in sample) + 2, max_width)
                    if wrap_cols and col in wrap_cols:
                        ws.set_column(col_idx, col_idx, width, fmt_wrap)
                    else:
                        ws.set_column(col_idx, col_idx, width)

            wrap_cols_detail = [c for c in df_detail.columns if any(
                key in c for key in ["src_types", "top_docs", "_value"]
            )]
            autofit(ws_d, df_detail, wrap_cols=wrap_cols_detail, max_width=100)
            autofit(ws_s, df_summary, wrap_cols=None, max_width=40)

            # 첫 행 고정
            ws_d.freeze_panes(1, 0)
            ws_s.freeze_panes(1, 0)

            # summary 형식: 퍼센트/점수/정수
            pct_cols   = ["fields_found", "사업명", "발주기관", "계약방식", "사업기간", "제출마감일", "예산", "전화"]
            float_cols = ["top1_score", "mean_score"]
            int_cols   = ["uniq_docs", "ctx_tokens_sum"]

            # 컬럼 인덱스 찾기 헬퍼
            def col_index(df, name):
                try:
                    return list(df.columns).index(name)
                except ValueError:
                    return None

            # 퍼센트/숫자 서식 적용
            for name in pct_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_pct1)

            for name in float_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_float3)

            for name in int_cols:
                ci = col_index(df_summary, name)
                if ci is not None:
                    ws_s.set_column(ci, ci, None, fmt_int)

            # README 시트(메타데이터)
            ws_r = wb.add_worksheet("README")
            meta = {
                "timestamp": ts,
                "engine": engine,
                "rows_detail": len(df_detail),
                "rows_summary": len(df_summary),
                # 아래 파라미터들은 있으면 기록
                "TOP_K": globals().get("TOP_K", "-"),
                "MMR_LAMBDA": globals().get("MMR_LAMBDA", "-"),
                "NUM_SUB_QUERIES": globals().get("NUM_SUB_QUERIES", "-"),
            }
            ws_r.write(0, 0, "key"); ws_r.write(0, 1, "value")
            for i, (k,v) in enumerate(meta.items(), start=1):
                ws_r.write(i, 0, str(k))
                ws_r.write(i, 1, str(v))

        else:
            # openpyxl 엔진: 기본 저장 + 첫 행 고정만 처리
            ws_d = writer.sheets["detail"]
            ws_s = writer.sheets["summary"]
            try:
                ws_d.freeze_panes = "A2"
                ws_s.freeze_panes = "A2"
            except Exception:
                pass

    print(f"[OK] 엑셀 저장: {xlsx_path}")

    # CSV도 같이 남기고 싶다면 주석 해제
    df_detail.to_csv(csv_detail, index=False)
    df_summary.to_csv(csv_summary, index=False)
    print(f"[OK] CSV 저장:\n - {csv_detail}\n - {csv_summary}")



# # RAG Retrieval 비교 보고서
# 
# - 평가 대상 접근법
#     1. **MMR 리트리버(개선판)** — 섹션인지 청킹, HTML 순수 텍스트 포함, Top-K=10, 간단 정규식, MMR 재랭크
#     2. **Multi-Query Retrieval** — 쿼리 확장 후 통합 및 MMR
#     3. **Cross-Encoder 리랭커(CE)** — 임베딩 Top-K 후보를 생성모델로 의미 점수화 후 재정렬
# - 평가 질의(예시)
#     
#     “사업명·발주기관·계약방식?”, “총 사업기간·제출마감일?”, “추정가격(예산)?”, “제출 방식·서류 목록?”, “담당 부서·전화번호?”
#     
# - 주요 데이터프레임
#     - **df_summary**: 접근법별 평균 지표
#     - **df_detail**: 질의×접근법별 세부 결과(찾은 필드, 출처 등)
# 
# ---
# ## 공통 전처리/세팅 (세 접근법 모두 동일)
# 
# - **문서 소스**
#     - 텍스트: **HTML 순수 텍스트** + **XML 텍스트** (둘 다 포함)
#     - 이미지: **HTML·XML에서 추출한 자산 이미지 OCR** 결과 블록 포함(필요 시)
# - **청킹(Chunking) 전략**
#     - **섹션 인지 청킹**: 문서의 목차/헤더(예: “Ⅰ. 사업 개요”, “2. 계약방식”)를 **경계**로 우선 분할
#     - 본문은 **문단 단위**로 묶고, 너무 긴 덩어리는 슬라이딩 윈도우로 나눔(경계 손상 최소화)
#     - 표/양식에서 나온 OCR 텍스트는 **레이블:값** 형태 보존(가능한 경우)
# - **임베딩(Embeddings)**
#     - 모델: **OpenAI `text-embedding-3-small` (차원=1536)**
#     - 전처리: 소문자/공백 정규화(기본), **L2 정규화 후 코사인 유사도** 사용
#     - 인덱스: 단순 **메모리 행렬(emb @ q)** 기반 유사도 검색
# 
# ------
# 
# 
# ## 1) Executive Summary (핵심 결론)
# 
# - **정확도(필드 회수율)**: **CE 리랭커**가 최고
#     - fields_found 평균: **CE 200%** > MMR 160% > MQ 120%
#     - 세부 필드 중 **계약방식** 커버리지에서 CE가 우세(CE 60% > MQ 20% > MMR 0%).
#     - 유사도 품질도 CE가 가장 높음(Top-1≈0.89, Mean≈0.72).
# - **비용/속도(컨텍스트 길이)**: **Multi-Query**가 가장 가볍고 빠름
#     - ctx_tokens_sum(평균): **MQ ~1,867** < MMR ~3,172 < CE ~4,060 (낮을수록 비용↓).
# - **문서 다양성(고유 문서 수)**: **MMR**이 근소 우위
#     - uniq_docs(평균): **MMR 4.0** ≥ CE 3.6 = MQ 3.6
#     - 다양한 문서 컨텍스트를 유지하려면 MMR이 유리.
# - **필드별 스냅샷**
#     - **발주기관**: **MMR 80%** ≥ MQ 60% = CE 60%
#     - **사업명**: MMR=CE 40% > MQ 20%
#     - **예산**: MMR=CE 20% > MQ 0%
#     - **전화/기간/마감일**: 전반적으로 낮음(정규식/포맷 보강 필요)
# 
# > 요약 추천:
# > 
# > - **정확도·품질 최우선** 시나리오 → **CE 리랭커**
# > - **비용·속도 최우선** 시나리오 → **MMR 기본 + 선택적 CE** (조건부 ON)
# > - **짧은 키워드성 질의/라벨 질의**에서는 **Multi-Query**를 보완적으로 사용
# 
# ---
# 
# ## 2) 지표 설명 (해석 가이드)
# 
# - **fields_found(%)**: 각 질문에서 **7개 필드**(사업명/발주기관/계약방식/사업기간/제출마감일/예산/전화) 중 몇 개를 찾았는지를 **비율(%)로 평균**.
#     
#     예) 200%는 “질문당 7개 중 평균 2개를 찾았다”로 이해하면 됨(질문 수에 따라 정규화된 평균).
#     
# - **각 필드(%)**: 해당 필드를 **찾아낸 질문의 비율**(0~100%).
#     
#     예) 예산=60% → 10개 질문 중 6개에서 예산을 성공 추출.
#     
# - **top1_score / mean_score**: Top-K 후보의 **유사도(또는 CE 점수)**.
#     
#     Top-1은 가장 높은 후보, mean은 Top-K 평균. 서로 다른 스케일 간 **상대 비교**에 사용.
#     
# - **uniq_docs**: Top-K 컨텍스트에 포함된 **고유 문서 수(평균)**. 높을수록 **다양성/리콜** 우수.
# - **ctx_tokens_sum**: Top-K 컨텍스트의 **총 토큰 수(평균)**. 낮을수록 **추론 비용↓/속도↑**.
# 
# ---
# 
# ## 3) 접근법별 핵심 지표 (요약)
# 
# | approach | fields_found | top1_score | mean_score | uniq_docs | ctx_tokens_sum | 사업명 | 발주기관 | 계약방식 | 사업기간 | 제출마감일 | 예산 | 전화 |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | **CE** | **200.0%** | **0.890** | **0.719** | 3.6 | 4,060 | 40.0% | 60.0% | **60.0%** | 0.0% | 0.0% | 20.0% | 20.0% |
# | **MMR** | 160.0% | 0.547 | 0.447 | **4.0** | 3,172 | **40.0%** | **80.0%** | 0.0% | 0.0% | 0.0% | **20.0%** | **20.0%** |
# | **MQ** | 120.0% | 0.613 | 0.477 | 3.6 | **1,867** | 20.0% | 60.0% | 20.0% | 0.0% | 0.0% | 0.0% | 20.0% |
# 
# > 해석
# > 
# > - **CE**: 전반적인 **회수율/랭킹 품질**이 가장 좋지만, **컨텍스트 길이(비용)**가 큼.
# > - **MMR**: **발주기관·사업명** 등 **텍스트 기반 필드**에서 강하고, **문서 다양성**이 가장 좋음.
# > - **MQ**: **가장 저렴/가벼움**, 다만 필드 회수율은 낮음(특정 질의군 보완용이 적합).
# 
# ---
# 
# ## 4) 필드별 최고 커버리지
# 
# - **발주기관**: **MMR 80%**
# - **계약방식**: **CE 60%**
# - **사업명**: **MMR=CE 40%**
# - **예산**: **MMR=CE 20%**
# - **사업기간/제출마감일/전화**: 전 접근법 공통으로 낮음(0~20%)
# 
# > 시사점
# > 
# > - **규정 용어/단락 표제**(예: 발주기관)는 **MMR**이 강함(섹션인지 청킹+HTML 텍스트 포함 효과).
# > - **결정적 키워드가 문장 내 결합**(예: “계약방법: 제한경쟁입찰”)된 경우 **CE**가 의미 매칭으로 우위를 보임.
# > - **날짜/기간/전화**는 **정규식·패턴 보강**과 **표/레이블 처리**(OCR/표 인식)가 필요.
# 
# ---
# 
# ## 5) 컨텍스트 소스 타입 구성(정성 요약)
# 
# - **MMR/MQ**: `html_text` 비중이 높음 → **텍스트 본문 기반** 질문에 강점.
# - **CE**: `asset_ocr_html/xml`이 상대적으로 많이 섞임 → **표/양식·캡처 영역**에서 추가 근거를 끌어와 **계약방식·예산** 같은 항목에 유리.
# 
# > 운영상 의미:
# > 
# > - 본문 중심 필드는 **MMR**로 충분히 커버.
# > - 표/양식 중심(특히 요구조건 목록, 계약방식/예산 표)은 **CE**의 보강이 체감 효과가 큼.
# 
# ---
# 
# ## 6) 질의별 비교(예)
# 
# - “**사업명·발주기관·계약방식**?” → **CE**가 계약방식까지 회수해 **필드 수 3개** 달성 사례 다수.
# - “**총 사업기간·제출마감일**?” → 전반적으로 낮음. **날짜 정규식·레인지 추출** 보강 필요.
# - “**추정가격(예산)**?” → **MMR/CE**가 일부 정답 회수(20%), MQ는 미회수. **가격/단위 정규식** 정교화 권장.
# - “**담당 부서·전화**?” → 전화 포맷 다양성, 문서 헤더/푸터 위치 문제로 낮음. **국번/하이픈/괄호 포맷 전처리** 필요.
# 
# ---
# 
# ## 7) 운영 제안(파이프라인)
# 
# ### 7.1 기본 전략(비용/속도 균형)
# 
# 1. **Retriever**: MMR Top-K=10 (섹션인지 청킹 + HTML 텍스트 포함)
# 2. **Gate(선택적)**: 질문이 **계약/예산/목록·표 중심**이면 CE 리랭커 **ON** (Top-10 → Top-5 재정렬)
# 3. **Generator**: nano/mini 계열로 시작, 필요 시 gpt-4.1/4o **부분 승급**
# 4. **정규식 후처리**: 전화·날짜·금액 포맷 보강(여러 패턴 동시 지원)
# 
# ### 7.2 저비용/고속 전략
# 
# - **MQ + MMR**만으로 1차 응답.
# - **자신도 낮음/필드 미회수**일 때만 **CE 재시도**.
# 
# ### 7.3 정밀 추출 전략(고난도 질의)
# 
# - 항상 **CE 리랭커** 적용, Top-K=10→5.
# - OCR 블록 포함(특히 표/양식 페이지 우선).
# 
# ---
# 
# ## 8) 단기 개선 체크리스트
# 
# 1. **정규식 보강**:
#     - 전화: `\(?0\d{1,2}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}` + 내선 패턴
#     - 날짜/기한: `YYYY[.\-/ ]?MM[.\-/ ]?DD`, “계약 후 N개월 이내”, “제출 마감: YYYY.MM.DD HH:MM”
#     - 금액: `[\d,]+(?:원|백만원|억원)` + VAT 표기
# 2. **표/양식 인지**:
#     - OCR 블록에서 **레이블:값** 쌍 우선 추출(“계약방법”, “추정가격”, “발주기관”).
#     - 표 라인브레이크 노이즈 정리.
# 3. **쿼리 가이드**(UX):
#     - 사용자가 “사업기간/마감일”을 묻는 질문에는 **날짜/기간 표현** 예시를 함께 유도.
# 
# ---
# 
# ## 9) 결론
# 
# - **CE 리랭커**는 **정확도**(특히 ‘계약방식’ 등 의미 매칭이 필요한 필드)에서 가장 우수.
# - **MMR**은 **발주기관/사업명** 등 본문 단락 기반 필드에서 강하고 **문서 다양성**이 좋음.
# - **MQ**는 **비용·속도** 장점이 뚜렷하나, 현 설정에서는 회수율이 낮아 **보조적**으로 쓰는 것이 적절.
# 
# > 권장 배치: “MMR 기본 + CE 조건부” 전략으로 시작 → 정규식·표처리 보강 후, 필요 영역에서만 CE를 켜 비용을 제어.
# >

# In[ ]:





# In[ ]:




