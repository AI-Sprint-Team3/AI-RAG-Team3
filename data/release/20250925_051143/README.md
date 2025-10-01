# Release Bundle — 20250925_051143

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
- backend: openai
- model:   text-embedding-3-small
- dim:     1536
- vectors: 8271
- faiss:   index/faiss.index

## Notes
- 이 번들은 기본적으로 **symlink** 방식으로 수집됨.
- 대용량 이미지 스냅샷 디렉토리는 필요 시 주석 해제하여 포함 가능.
