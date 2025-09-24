from generation.prompt_templates import SUMMARY_PROMPT, QA_PROMPT


def select_prompt_auto(question: str) -> str:
  """
  질문 유형에 따라 프롬프트 선택
  - 단순 키워드 매칭
  """
  
  SUMMARY_KEYWORDS = ["요약", "정리", "핵심", "핵심내용", "주요포인트", "간략히", "간략하게", "한눈에", "개요", '전체']
  QA_KEYWORDS = ['무엇', '어떻게', '언제', '어디서', '누구', '포함', '있나요', '요구사항', '일정', '제출서류', '예산', '설명']
  
  q = question.lower()
  
  if any(kw in q for kw in SUMMARY_KEYWORDS):
    return SUMMARY_PROMPT
  else:
    return QA_PROMPT

def select_prompt_manual(prompt_type: str) -> str:
  """
  수동 선택 - 'summary' 또는 'qa'를 받아 프롬프트 리턴
  """
  if prompt_type == "summary":
    return SUMMARY_PROMPT
  elif prompt_type == "qa":
    return QA_PROMPT
  else:
    raise ValueError(f"지원하지 않는 prompt_type: {prompt_type}")