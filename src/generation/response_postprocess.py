import re
import json

def clean_response(text: str) -> str:
  """
  모델 응답 후처리
  - 공백 정리
  - 특수문자/불필요한 태그 제거
  """
  if not text:
    return ""
  
  # 불필요한 공백 제거
  # text = re.sub(r"\s+", " ", text).strip()
  
  # 불필요한 마커 제거
  # text = text.replace("Answer:", "").replace("Summary:", "").strip()
  
  return text

def parse_json_answer(text: str) -> dict:
  """
  모델 응답 json으로 변환
  - json이 아니면 fallback으로 {"raw": text}
  """
  cleaned = clean_response(text)
  
  try:
    # JSON 블록만 추출
    json_match = re.search(r"\{.x\}", cleaned, re.DOTALL)
    if json_match:
      return json.loads(json_match.group[0])
    return json.loads(cleaned)
  except Exception:
    return {"raw": cleaned}
  
  
def parse_bullet_points(text: str) -> list[str]:
  """
  모델 응답 bullet-point 리스트 변환
  - '-' 또는 숫자 리스트 패턴 탐지
  """
  cleaned = clean_response(text)
  bullets = re.findall(r"(?:-|\d+\.)\s+(.*)", cleaned)
  
  # bullet이 없으면 그냥 한 줄짜리 리스트
  return bullets if bullets else [cleaned]



def _normalize_dashes(text: str) -> str:
  """
  전화번호, 날짜, 금액 등 숫자 사이 하이픈이나 특수 패턴을 보호하기 위해
  임시 태그로 치환
  """
  # 전화번호: 043-713-8880 등
  text = re.sub(r'(\d{2,4}-\d{3,4}-\d{3,4})', r'<TEL>\1</TEL>', text)
  
  # 날짜: 2024-08-01 등
  text = re.sub(r'(\d{4}-\d{1,2}-\d{1,2})', r'<DATE>\1</DATE>', text)
  
  # 숫자 사이 하이픈 일반
  text = re.sub(r'(\d)-(\d)', r'\1<NUMDASH>\2', text)
  
  # 4) 일반 텍스트 대시 통일
  text = re.sub(r"[–—―]", " - ", text)
  
  return text

def _restore_special_patterns(text: str) -> str:
  """
  보호된 패턴을 원래 문자로 복원
  """
  if isinstance(text, list):
        return [_restore_special_patterns(t) for t in text]  # 재귀 처리
  text = text.replace("<NUMDASH>", "-")
  text = re.sub(r'<TEL>(.*?)</TEL>', r'\1', text)
  text = re.sub(r'<DATE>(.*?)</DATE>', r'\1', text)
  return text

def pretty_format_answer(text: str) -> str:
    """
    모델 응답을 섹션(1.,2.,...) 기준으로 나누고,
    각 섹션의 ' - '로 연결된 항목들을 bullet 라인으로 변환합니다.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 기본 공백/대시 정규화
    text = _normalize_dashes(text)
    text = re.sub(r"\s+", " ", text).strip()

    # "1)" 혹은 "1." 스타일도 커버: 통일해서 "1. " 로 만듦
    text = re.sub(r"(\d+)[\)\.]\s*", r"\1. ", text)

    # 2) 섹션 경계로 분할: 각 파트가 "1. ..." 로 시작
    parts = re.split(r'(?=\d+\.\s)', text)
    out_lines = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # "1. 제목 - 항목A - 항목B ..." 형태로 가정
        m = re.match(r'^(\d+\.)\s*(.*)$', part, flags=re.S)
        if not m:
            # 숫자 표기가 없으면 단락으로 처리
            out_lines.append(_restore_special_patterns(part))
            continue

        num = m.group(1)           # ex: "1."
        rest = m.group(2).strip()  # ex: "입찰 기본정보 - 공고명: ... - 발주기관: ..."

        # 만약 ' - '가 없다면, 단일 줄 섹션으로 처리
        if ' - ' not in rest:
            out_lines.append(f"{num} {_restore_special_patterns(rest)}")
            continue
          
        # " - " 앞뒤 조건: 하이픈 앞이 숫자/영문이면 제외
        parts_after_title = [
            p.strip() for p in re.split(r'(?<![0-9A-Za-z])\s-\s', rest) 
            if p.strip()
        ]
        title = _restore_special_patterns(parts_after_title[0])
        items = _restore_special_patterns(parts_after_title[1:])

        # 출력: 번호+제목 한 줄, 그 아래 각 항목을 불릿으로
        out_lines.append(f"{num} {title}")
        for it in items:
            # 추가 정리: 항목 내부에 ';' 또는 ' / '로 나뉘어 있으면 하위불릿으로 분리 가능
            if ';' in it:
                subitems = [s.strip() for s in it.split(';') if s.strip()]
                for s in subitems:
                    out_lines.append(f"- {s}")
            else:
                out_lines.append(f"- {it}")

        # 섹션 끝에 빈 줄 한 줄 넣어서 가독성 향상
        out_lines.append("")

    # 뒤쪽 불필요한 공백 라인 정리
    formatted = "\n".join(out_lines).rstrip() + "\n"
    return formatted

