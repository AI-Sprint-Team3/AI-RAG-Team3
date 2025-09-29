from src.generation.prompt_templates import SUMMARY_PROMPT, QA_PROMPT
import openai

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
  

def prompt_llm(question:str) -> str:
  prompt = f"""
    당신은 질문 분류기입니다. 
    아래 질문이 어떤 의도(intent)를 가지는지 반드시 [검색/필터링, 요약, 안내, 기타] 중 하나로만 답해주세요.
    다른 내용이나 설명을 추가하지 마세요.

    질문: "{question}"
    의도:
    """
  response = openai.chat.completions.create(
        model="gpt-4.1-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0,       
        max_completion_tokens=50
    )
  intent = response.choices[0].message.content

  return intent

def select_prompt_by_intent(question: str) -> str:
    """
    LLM 기반 Intent 분류 후 프롬프트 선택
    """
    # 1. LLM으로 Intent 판단
    intent = prompt_llm(question).strip()  # "검색/필터링", "요약", "안내", "기타"
    print(f"분류기: {intent}")
    
    # 2. Intent에 따라 프롬프트 선택
    if intent == "요약":
        return SUMMARY_PROMPT
    else:  # 검색/필터링, 안내, 기타
        return QA_PROMPT

