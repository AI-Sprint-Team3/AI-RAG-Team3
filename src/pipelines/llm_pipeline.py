import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict
from src.generation.prompt_selector import select_prompt_auto, select_prompt_manual, select_prompt_by_intent
from src.generation.response_postprocess import pretty_format_answer
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document  
from src.generation.utils import cross_check_answer, extract_keywords_from_text

class LLMPipeline:
  """
  RAG + 멀티턴 대화 히스토리 지원 LLM 파이프라인
  """
  
  def __init__(self, llm, retriever=None, confidence_threshold: float = 0.5, max_retry: int = 2):
    """
      llm(OpenAILLM): OpenAI 기반 LLM 래퍼
      retriever (Optional): 벡터DB retriever
    """
    self.llm = llm
    self.retriever = retriever
    # self.chat_history: List[Dict] = []  # {"role": "user"/"assistant", "content": str}
    self.memory = ConversationBufferMemory(return_messages=True)  # ✅ 추가
    self.confidence_threshold = confidence_threshold  # Confidence 기준
    self.max_retry = max_retry  # 재검색 최대 횟수
    
    
  def run(self, query: str, contexts:List[Document] = None, prompt_type: str = None) -> str:
    """
    사용자 쿼리를 받아 LLM 응답 생성
    - prompt_type이 주어지면 수동 선택
    - 없으면 자동 분기(select_prompt)
    """
    
    # 1) RAG context 가져오기
    context = ""
    if contexts is None and self.retriever:
      results = self.retriever(query)   # advanced_retrieve 반환값
      context_docs = self.convert_to_documents(results)
      context = "\n\n".join([doc.page_content for doc in context_docs])
    else:
      context = contexts or ""
      
    # 2-1) 프롬프트 선택 - 키워드
    # if prompt_type:
    #   prompt_template = select_prompt_manual(prompt_type)
    # else:
    #   prompt_template = select_prompt_auto(query)
    
    # 2-2) 프롬프트 선택 - llm
    prompt_template = select_prompt_by_intent(query)
    
    # 3) memory 불러오기 (이전 대화 기록)
    history = self.memory.load_memory_variables({})["history"]

    
    # 4) LLM 호출
    raw_response = self.llm.generate(
      question=query,             # 실제 사용자 질문
      template=prompt_template,   # 선택된 프롬프트
      rag_context=context,         # RAG 검색 결과
      history=history
    )
    
    # 5) 후처리
    response = pretty_format_answer(raw_response)
    
    # 6) 히스토리 추가
    self.memory.save_context(
      {"input": query},   # user message
      {"output": response}  # assistant message
      )
    
    # 7) cross-check 적용
    candidates = [{"content": doc.page_content, "meta": doc.metadata} for doc in context]
    cross_result = cross_check_answer(response, candidates)
    
    # 8) 최종반화: LLM 응답 + 근거 검증 결과
    return {
      "answer": response,
      "cross_check": cross_result
    }
  
  def run_with_confidence_loop(self, query:str, contexts: List[Document] = None, prompt_type: str = None):
    """
    Confidence 임계값 기반 재검색 + 답변 재생성 루프
    - query: 사용자 질문
    - contexts: 검색 결과 Document 리스트
    """
    attempts = 0
    response_data = None
    
    # self.retriever_original에 원래 retriever 함수 저장
    if not hasattr(self, "retriever_original"):
      self.retriever_original = self.retriever
    while attempts <= self.max_retry:
      # run() 호출
      if not contexts or len(contexts) == 0:
        contexts = self.retriever_original(query)  # advanced_retrieve 래핑
        if not contexts:
          print("⚠️ 문서를 찾지 못했습니다. LLM 호출 없이 종료합니다.")
          return None   # 종료
        
      response = self.run(query, contexts, prompt_type)
      confidence = response['cross_check'].get("confidence", 0)
      
      # confidence 확인
      if confidence >= self.confidence_threshold:
        response_data = response
        break
      else:
        print(f"⚠️ Confidence 낮음 ({confidence:.2f}), 재검색 및 답변 재생성 중... (시도 {attempts})")
        
        # 재검색 전략: top_k 증가
        top_k_retry = 3 + attempts
        if self.retriever_original:
          results = self.retriever_original(query, top_k=top_k_retry)
          contexts = self.convert_to_documents(results)

        # # contexts 재생성
        if not contexts or len(contexts) == 0:
          results = self.retriever_original(query, top_k=top_k_retry)
          contexts = self.convert_to_documents(results)

        attempts += 1
        
    if response_data is None:
      response_data = response  # 마지막 시도 결과 반환
    
    return response_data
    

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


# -----------------------
# retriever 래핑 예제
# -----------------------
def make_retriever(collection, embedding_fn, top_k=3, use_mmr=True, use_bm25=True, bm25=None, filter_title=None):
  """
  advanced_retrieve 래퍼를 만들어 반환
  - use_mmr=True: MMR 기반 검색
  - filter_title: 제목(metadata) 필터링
  """
  from src.retrieval.retriever import advanced_retrieve

  def retriever_fn(query):
    print(f"=== advanced_retrieve 호출 ===")
    print(f"filter_title: {filter_title}")
    results = advanced_retrieve(
      query=query,
      collection=collection,
      embedding_fn=embedding_fn,
      top_k=top_k,
      use_mmr=use_mmr,
      use_bm25=use_bm25,
      bm25=bm25,
      agency_filter=filter_title,
    )
    return results
  
  return retriever_fn

def run_queries(pipeline, queries: List[str]):
  """
  여러 쿼리를 pipeline에 돌려서 결과 확인
  """
  for q in queries:
    results = pipeline.retriever(q)  # retriever에 필터/옵션 적용됨
    contexts = pipeline.convert_to_documents(results, include_score=True)
    # response = pipeline.run(q, contexts)
    # response = pipeline.run(q)
    
    # confidence_loop 적용
    response = pipeline.run_with_confidence_loop(q, contexts)
    cross_check = response['cross_check']
    
    print(f"Q: {q}")
    print(f"A: {response['answer']}\n")
    print(f"Confidence: {cross_check['confidence']:.2f} | Validity: {cross_check['validity']}\n")
    
    mk = cross_check['matched_keywords']
    print("Matched keywords:", ", ".join(mk[:10]) + (", ..." if len(mk) > 10 else ""))
    ak = cross_check['answer_keywords']
    print("Answer keywords:", ", ".join(ak[:10]) + (", ..." if len(ak) > 10 else ""))
    ck = cross_check['candidate_keywords']
    print("Candidate keywords:", ", ".join(ck[:10]) + (", ..." if len(ck) > 10 else ""))
    
    print("\n" + "*" * 50 + "\n")

if __name__ == "__main__":
  from functools import partial
  from src.embeddings.embedder import EmbedderFactory
  from src.embeddings.vectorstore_chroma import get_collection, add_docs_to_chroma
  from src.retrieval.retriever import advanced_retrieve
  from src.retrieval.bm25_helper import BM25Helper
  from src.generation.llm_openai import OpenAIRAGClient
  from config.settings import COLLECTION_NAME
  from src.pipelines.query_pipeline import load_docs
  
  
  # 1) LLM 객체 준비 (OpenAILLMClient)
  llm = OpenAIRAGClient(temperature=0.7)
  
  # 2) 벡터 DB & 임베딩 주비
  collection = get_collection(COLLECTION_NAME)
  embedding_fn = EmbedderFactory.get_embedder(provider="openai")

  # 문서 업로드 (배치)
  documents = load_docs()
  # print("📌 문서 업로드 중...")
  # add_docs_to_chroma(documents, collection=collection, embedding_fn=embedding_fn, batch_size=8)
  # print("✅ 업로드 완료")
      
  # 3) BM25 준비
  # print(f"✅ 불러온 문서 수: {len(documents)}")
  corpus_texts = [d.get("texts", {}).get("merged", "") for d in documents if d.get("texts", {}).get("merged", "")]
  bm25_helper = BM25Helper(corpus_texts)
  
  # 4) advanced_retrieve 래핑 - 조건 o
  retriever = make_retriever(
    collection=collection,
    embedding_fn=embedding_fn,
    bm25=bm25_helper,
    top_k=3,
    use_mmr=True,
    filter_title='bioin_의료기기산업_종합정보시스템_정보관리기관_기능개선_사업_2차'
  )

  # 조건 x
  # retriever = make_retriever(
  #   collection=collection,
  #   embedding_fn=embedding_fn,
  #   bm25=bm25_helper,
  #   top_k=3,
  #   use_mmr=True,
  # )
  
  # 5) LLMPipeLine 생성
  pipeline = LLMPipeline(llm=llm, retriever=retriever)
  
  # 6) 테스트 쿼리 실행
  # 쿼리 실행 - 여러개
  queries = [
      "이 사업의 목적은 무엇인가요?",
      "과업 수행 기간은 언제부터 언제까지인가요?",
      "과업의 범위는 무엇을 포함하나요?",
      "발주 기관은 어디인가요?",
      "입찰 참여 마감일은 언제인가요?",
      "이 사업의 주요 산출물은 무엇인가요?",
      "과업 대상 지역은 어디인가요?",
      "사업 금액은 얼마인가요?",
      "문서 이름 알려줄 수 있어?",
      "사업 기간 알려줘",
      "이 공고는 중소기업만 참여 가능한가요?", # confidence 체크
      "입찰 방식이 제한경쟁인가요, 아니면 일반경쟁인가요?",
      "사업 수행 장소는 서울인가요?",  # hallucination 방지
      "참가자격 제한사항 중 필수 보유 면허가 있나요?"
      "이 공고의 총 직원 수는 몇 명인가요?",
      "계약 보증 관련된 특이사항이 있나요?",
      "제안서 발표는 오프라인만 가능한가요?",
      "문서 요약해줘",    # 요약
      
  ]
  # queries = [
    # "아시아 육상 경기 대회 요구사항이 뭔가요?"
  # ]

  run_queries(pipeline, queries)
