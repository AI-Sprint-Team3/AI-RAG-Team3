# Data Processing & Personal Experiments (feature/hana)

## 프로젝트 개요

**RFP 문서 기반 Retrieval-Augmented Generation (RAG) 시스템 구축**을 위한 데이터 전처리와 개인 실험 기록을 담고 있습니다.  

- **목표**
  - HTML / XML / HWPX / PDF 문서에서 텍스트와 이미지(OCR)를 추출하여 전처리
  - 청킹(Chunking) → 임베딩(Embedding) → 벡터 검색(Faiss) 파이프라인 구축
  - 검색 품질 개선을 위해 노이즈 제거, 헤딩 보존, OCR 병합 등 다양한 실험 진행
- **역할 (feature/hana)**
  - 원본 데이터 변환 및 구조 정리
  - PDF/HWPX 파싱, 청킹 전략 테스트 (800/200 등)
  - 캐시·중복 제거 정책 실험
  - 개인 작업 테스트 기록 정리 (리베이스, requirements 정리 등)

---

## 1. 개인 테스트 / 실험 기록

### 1-1. HWPX 테스트 (4개 문서 한정)

- **테스트 범위**
  - HWP 원본 문서 중 4개를 HWPX로 변환 후 전처리 실험
- **특징**
  - HWPX는 한글 오피스의 오픈 XML 포맷 → 문단, 표, 머리말/꼬리말 등 구조 보존 우수
  - 압축(zip) 구조라 파싱 용이
- **결과**
  - 문단 경계가 잘 살아있어 청킹 품질은 양호
  - OCR 노이즈가 섞일 경우 품질 저하 발생
  - 최종 파이프라인에는 포함하지 않고, **XML+HTML+PDF 라인업만 사용**

---

### 1-2. XML+HTML Subset 실험

- **목적**
  - 소규모 데이터로 청킹/임베딩/검색 성능 빠르게 점검
- **내용**
  - md5 캐시 정책 적용 → 결과 품질 저하로 폐기
  - Hit@3, MRR@10 지표를 통해 검색 성능 측정
- **기록**
  - notebooks/290930_* → `.py`로 export
  - feature/hana 브랜치에만 반영 (main 미반영)

---

## 2. 실제 데이터 전처리 (XML+HTML+PDF 기반)

- **Raw 데이터 수집 및 구조화**
  - ~/data/raw/converted/{html, xml, pdf}, 문서별 assets 포함
  - hwpx는 전체 파이프라인에는 미사용 (실험용만 사용)

- **텍스트 추출 & 정규화**
  - HWP → HTML/XML 변환
  - PDF 원문은 클린 전처리 (페이지번호/헤더·푸터 제거)

- **머지/조인**
  - join_key → merge_key 통합
  - 충돌 그룹 해시 suffix 처리
  - `~/data/processed/docs_merged.jsonl` (110 docs)

- **청킹**
  - 기본: 길이 800 / overlap 200
  - OCR: 600/150, OCR concat: 700/200
  - 결과: `~/data/processed/chunks_xmlhtml_800_200.jsonl` (8271 chunks)

- **임베딩 & 인덱스**
  - OpenAI text-embedding-3-small (1536d, L2 norm)
  - FAISS IndexFlatIP → `~/data/processed/index/faiss.index`
  - 코사인 유사도 평균 ~0.5 (목표 ≥ 0.8)

- **이슈**
  - 청크 시작부 헤딩 부족
  - 반복되는 헤더/푸터 노이즈
  - OCR 노이즈 혼합 → 검색 정확도 하락

---

## 3. 브랜치 관리 & 커밋 처리

- 개인 작업은 feature/hana 전용, main 미반영
- requirements.txt 정리: streamlit/, notebooks/ 사본 삭제 후 루트에 통일
- non-fast-forward 오류 → `git pull --rebase`로 해결, 필요 시 stash 사용
- 릴리즈 산출물 정리:  
  - `~/data/release/20250925/`  
  - `~/data/release/20250929/`

---

## 4. 개인 협업일지 링크
- https://www.notion.so/Daily-DB_-_M-Project-26bbea5ebd7c80d68e58d93a3c928f7f?source=copy_link
