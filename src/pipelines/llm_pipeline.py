import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict
from src.generation.prompt_selector import select_prompt_auto, select_prompt_manual, select_prompt_by_intent
from src.generation.response_postprocess import clean_response
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
    if contexts is None and self.retriever:
      results = self.retriever(query)   # advanced_retrieve 반환값
      context_docs = self.convert_to_documents(results)
      context = "\n\n".join([doc.page_content for doc in context_docs])
    else:
      context = contexts or ""
      
    # 프롬프트 선택 - 키워드
    # if prompt_type:
    #   prompt_template = select_prompt_manual(prompt_type)
    # else:
    #   prompt_template = select_prompt_auto(query)
    
    # 프롬프트 선택 - llm
    prompt_template = select_prompt_by_intent(query)
    
    # memory 불러오기 (이전 대화 기록)
    history = self.memory.load_memory_variables({})["history"]

    
    # LLM 호출
    raw_response = self.llm.generate(
      question=query,             # 실제 사용자 질문
      template=prompt_template,   # 선택된 프롬프트
      rag_context=context,         # RAG 검색 결과
      history=history
    )
    
    # 후처리
    response = clean_response(raw_response)
    
    # 히스토리 추가
    self.memory.save_context(
      {"input": query},   # user message
      {"output": response}  # assistant message
      )
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
    return advanced_retrieve(
      query=query,
      collection=collection,
      embedding_fn=embedding_fn,
      top_k=top_k,
      use_mmr=use_mmr,
      use_bm25=use_bm25,
      bm25=bm25,
      agency_filter=filter_title,
    )

  return retriever_fn

def run_queries(pipeline, queries: List[str]):
  """
  여러 쿼리를 pipeline에 돌려서 결과 확인
  """
  for q in queries:
    results = pipeline.retriever(q)  # retriever에 필터/옵션 적용됨
    contexts = pipeline.convert_to_documents(results, include_score=True)
    answer = pipeline.run(q, contexts)
    # answer = pipeline.run(q)
    print(f"Q: {q}\nA: {answer}\n")
    print("*" * 50 + "\n")

if __name__ == "__main__":
  from functools import partial
  from src.embeddings.embedder import EmbedderFactory
  from src.embeddings.vectorstore_chroma import get_collection
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
  
  # 3) BM25 준비
  documents = load_docs()
  print(f"✅ 불러온 문서 수: {len(documents)}")
  
  corpus_texts = [d.get("texts", {}).get("merged", "") for d in documents if d.get("texts", {}).get("merged", "")]
  bm25_helper = BM25Helper(corpus_texts)
  
  # 4) advanced_retrieve 래핑
  retriever = make_retriever(
    collection=collection,
    embedding_fn=embedding_fn,
    bm25=bm25_helper,
    top_k=3,
    use_mmr=True,
    filter_title="2025_구미_아시아육상경기대회_조직위원회_2025_구미아시아육상"
  )
  
  # 5) LLMPipeLine 생성
  pipeline = LLMPipeline(llm=llm, retriever=retriever)
  
  # 6) 테스트 쿼리 실행
  # 쿼리 실행 - 여러개
  
  queries = [
      "이 사업의 목적은 무엇인가요?",
      "과업 수행 기간은 언제부터 언제까지인가요?"
  ]

  run_queries(pipeline, queries)
  
  query = "아시아 육상 경기 대회 요구사항"
  response = pipeline.run(query)
  
  print("=== LLM 응답 ===")
  print(response)
