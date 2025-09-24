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
  text = re.sub(r"\s+", " ", text).strip()
  
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