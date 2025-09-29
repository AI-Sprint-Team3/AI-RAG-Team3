from typing import List, Dict
# from generation.llm_openai import OpenAIRAGClient
from generation.prompt_selector import select_prompt_auto, select_prompt_manual, select_prompt_by_intent
from generation.response_postprocess import clean_response
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document  

class LLMPipeline:
  """
  RAG + 멀티턴 대화 히스토리 지원 LLM 파이프라인
  """
  
  def __init__(self, llm, retriever=None):
    """
      llm(OpenAILLM): OpenAI 기반 LLM 래퍼
      retriever (Optional): 벡터DB retriever
    """
    self.llm = llm
    self.retriever = retriever
    # self.chat_history: List[Dict] = []  # {"role": "user"/"assistant", "content": str}
    self.memory = ConversationBufferMemory(return_messages=True)  # ✅ 추가
    
  def run(self, query: str, contexts:List[Document] = None, prompt_type: str = None) -> str:
    """
    사용자 쿼리를 받아 LLM 응답 생성
    - prompt_type이 주어지면 수동 선택
    - 없으면 자동 분기(select_prompt)
    """
    
    # RAG context 가져오기
    context = ""
    if self.retriever:
      docs = self.retriever.get_relevant_documents(query)
      # print(docs)
      context = "\n\n".join([context.page_content for context in docs])
    else:
      context = contexts
      
    # 프롬프트 선택 - 키워드
    # if prompt_type:
    #   prompt_template = select_prompt_manual(prompt_type)
    # else:
    #   prompt_template = select_prompt_auto(query)
    
    # 프롬프트 선택 - llm
    prompt_template = select_prompt_by_intent(query)
    
    # memory 불러오기 (이전 대화 기록)
    # history = self.memory.load_memory_variables({})["history"]

    
    # LLM 호출
    raw_response = self.llm.generate(
      question=query,             # 실제 사용자 질문
      template=prompt_template,   # 선택된 프롬프트
      rag_context=context,         # RAG 검색 결과
      # history=history
    )
    
    # 후처리
    response = clean_response(raw_response)
    
    # 히스토리 추가
    # self.memory.save_context(
    #   {"input": query},   # user message
    #   {"output": response}  # assistant message
    #   )
    return response

  def convert_to_documents(self, results: List[Dict], include_score: bool = True) -> List[Document]:
    """
    advanced_retrieve 등 검색 결과를 Document 리스트로 변환
    """
    docs = []
    for r in results:
      meta = r["meta"].copy()
      if include_score:
        meta["score"] = r.get("score", 0)
        
      docs.append(Document(page_content=r["text"], metadata=meta))
    return docs

  @property
  def chat_history(self):
    # llm_openai.OpenAIRAGClient 안에 저장된 대화 히스토리 리턴
    return self.llm.conversation_history