# 20250929 Release

이번 릴리스는 본문 누락 이슈를 해소한 정본(CANON)을 포함합니다.

## 핵심 정보
- 정본(CANON): `data/docs_merged.jsonl`
- SHA256: `6ff50832d11f01d7f794680ee13d3e1964848a98b2de23c9c1c1d993dce32138`
- 라인 수: 110

## 빠른 무결성 체크
```bash
# 라인 수 확인 (=110 이어야 정상)
wc -l data/docs_merged.jsonl

# 해시 확인 (아래 값과 동일해야 정상)
sha256sum data/docs_merged.jsonl
# → 6ff50832d11f01d7f794680ee13d3e1964848a98b2de23c9c1c1d993dce32138
```

## 보고서/부속 자료
- 릴리스 리포트: `reports/release_diagnosis.txt`
- 그 외: `interim/` 및 `data/legacy/` 폴더 참고
