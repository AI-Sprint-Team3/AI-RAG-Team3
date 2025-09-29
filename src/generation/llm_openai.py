import openai
from typing import List, Dict, Optional
from config.settings import OPENAI_API_KEY, LLM_MODEL
from langchain.schema import HumanMessage, AIMessage


class OpenAIRAGClient:
  """
  OpenAI 기반 멀티턴 + RAG context 지원 LLM 클라이언트
  """
  def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL,
                temperature: float = 0.7,
                max_tokens: int = 8192,
                top_p: float = 1.0,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0):
    openai.api_key = api_key  # 글로벌 API key 설정
    self.model = model
    self.temperature = temperature
    self.max_tokens=max_tokens
    self.top_p=top_p
    self.frequency_penalty=frequency_penalty
    self.presence_penalty=presence_penalty
    self.conversation_history: List[Dict[str, str]] = []
  
  def set_model(self, model_name: str):
    """
    사용중인 모델 변경
    """
    self.model = model_name
      
  def generate(
    self,
    question: str,
    template: str,
    rag_context: Optional[str] = None,
    history=None
  ) -> str:
    """
    LLM 호출
    - question: 현재 사용자 질문
    - template: QA_PROMPT 또는 SUMMARY_PROMPT
    - rag_context: 검색된 문서 청크 등 참고용 context
    """
    context = rag_context if rag_context else "정보가 없습니다."
    prompt_filled = template.format(context=context, question=question)

    messages = []

    # 기존 히스토리 포함
    if history:
      for msg in history:
        if isinstance(msg, HumanMessage):
          messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
          messages.append({"role": "assistant", "content": msg.content})
            
    
    # 현재 사용자 입력 (템플릿 적용)
    messages.append({
      "role": "user",
      "content": prompt_filled
    })
    
    # 모델별 파라미터 설정
    request_params = dict(
        model=self.model,
        messages=messages,
        top_p=self.top_p,
        frequency_penalty=self.frequency_penalty,
        presence_penalty=self.presence_penalty,
    )
    
    if "gpt-5" in self.model.lower():  # gpt-5 계열이면
      request_params["max_completion_tokens"] = self.max_tokens
    else:  # gpt-4, gpt-3.5 등
      request_params["max_tokens"] = self.max_tokens
      request_params["temperature"] = self.temperature
      
    response = openai.chat.completions.create(**request_params)
    
    answer = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    
    print(f"finish_reason: {finish_reason}")
    
    # history 저장
    # self.conversation_history.append({"role": "user", "content": question})
    # self.conversation_history.append({"role": "assistant", "content": answer})


    
    return answer
  

