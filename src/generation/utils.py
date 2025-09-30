from konlpy.tag import Okt
import re
from dateparser.search import search_dates
import re
from typing import List, Dict


okt = Okt()

def extract_query_keywords(query):
  
  # 1) 명사, 숫자 추출
  nouns_and_numbers = [word for word, pos in okt.pos(query) if pos in ['Noun', 'Number']]
  
  # 2) 날짜 문자열 추출
  dates_found = search_dates(query, languages=['ko'])
  dates = [d[0] for d in dates_found] if dates_found else []
  
  # 3) 합치기 (중복 제거)
  keywords = list(set(nouns_and_numbers + dates))
  return keywords

def extract_keywords_from_text(text: str) -> List[str]:
  """
  후보 문서나 LLM 답변에서 핵심 키워드 추출
  - 명사, 숫자, 날짜, 금액 등을 포함
  """
  # 1) 명사, 숫자
  nouns_and_numbers = [word for word, pos in okt.pos(text) if pos in ['Noun', 'Number']]
    
  # 2) 날짜 (ex: 2025년 5월 20일)
  dates = re.findall(r'\d{4}년 \d{1,2}월 \d{1,2}일', text)
    
  # 3) 금액 (ex: 1,000,000원)
  amounts = re.findall(r'\d{1,3}(?:,\d{3})*원', text)
    
  # 4) 기관명 (간단히 '연구원', '학교', '기관' 등 키워드 매칭)
  orgs = re.findall(r'(연구원|대학교|기관|회사)', text)
    
  # 합치고 중복 제거
  keywords = list(set(nouns_and_numbers + dates + amounts + orgs))
  return keywords

def cross_check_answer(answer: str, candidates: List[Dict]) -> Dict:
  """
  후보 문서 키워드와 LLM 답변 키워드를 비교
  - answer: LLM이 생성한 텍스트
  - candidates: RAG로 검색된 문서 리스트 [{'content': ..., 'meta': {...}}]
  """
  answer_keywords = set(extract_keywords_from_text(answer))
  all_candidate_keywords = set()
    
  for c in candidates:
    content = c.get('content', '')
    kws = extract_keywords_from_text(content)
    all_candidate_keywords.update(kws)
    
  # 교집합 계산
  matched = answer_keywords & all_candidate_keywords
  confidence = len(matched) / len(answer_keywords) if answer_keywords else 0.0
    
  return {
    "answer_keywords": list(answer_keywords),
    "candidate_keywords": list(all_candidate_keywords),
    "matched_keywords": list(matched),
    "confidence": confidence,
    "validity": "근거 충분" if confidence > 0.5 else "근거 불충분"
  }