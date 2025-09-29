import streamlit as st
from src.pipelines.llm_pipeline import LLMPipeline, make_retriever
from src.embeddings.embedder import EmbedderFactory
from src.embeddings.vectorstore_chroma import get_collection
from src.retrieval.bm25_helper import BM25Helper
from src.generation.llm_openai import OpenAIRAGClient
from config.settings import COLLECTION_NAME
from src.pipelines.query_pipeline import load_docs

# -------------------------
# 초기화 (한 번만)
# -------------------------
@st.cache_resource
def init_pipeline(selected_filter=None):
  # LLM
  llm = OpenAIRAGClient(temperature=0.7)

  # 벡터 DB
  collection = get_collection(COLLECTION_NAME)
  embedding_fn = EmbedderFactory.get_embedder(provider="openai")

  # BM25
  documents = load_docs()
  corpus_texts = [d.get("texts", {}).get("merged", "") for d in documents if d.get("texts", {}).get("merged", "")]
  bm25_helper = BM25Helper(corpus_texts)
  
  # merge_key 추출
  merge_keys = [d.get("merge_key") for d in documents if d.get("merge_key")]


  # Retriever
  retriever = make_retriever(
    collection=collection,
    embedding_fn=embedding_fn,
    bm25=bm25_helper,
    top_k=3,
    use_mmr=True,
    filter_title=selected_filter
  )

  # Pipeline
  pipeline = LLMPipeline(llm=llm, retriever=retriever)
  return pipeline, merge_keys

# -------------------------
# Session state 초기화
# -------------------------
if "agency_filter" not in st.session_state:
    st.session_state["agency_filter"] = None
if "pipeline" not in st.session_state:
    pipeline, merge_keys = init_pipeline()
    st.session_state["pipeline"] = pipeline
    st.session_state["merge_keys"] = merge_keys
else:
    pipeline = st.session_state["pipeline"]
    merge_keys = st.session_state["merge_keys"]



# -------------------------
# Streamlit UI
# -------------------------
st.title("RAG 기반 LLM Q&A")
tab = st.tabs(["문서 개수", "대화 히스토리", "검색"])[0]  # 기본 선택 탭: 문서 개수

# -------------------------
# 탭 1: 문서 개수
# -------------------------
with tab:
  st.header(f"문서 개수: {len(merge_keys)}개")
  
  selected_merge_key = st.selectbox("문서 선택 (클릭하면 해당 agency_filter 적용)", ["전체"] + merge_keys)
  
  if selected_merge_key == "전체":
    st.session_state["agency_filter"] = None
  else:
    st.session_state["agency_filter"] = selected_merge_key
  
  st.markdown(f"**선택된 문서:** {st.session_state['agency_filter']}")
  
  # retriever 필터 업데이트
  pipeline, _ = init_pipeline(selected_filter=st.session_state["agency_filter"])
  st.session_state["pipeline"] = pipeline
  
# -------------------------
# 검색 / 질문 입력
# -------------------------
st.header("질문 입력")
user_input = st.text_input("질문을 입력하세요:")

if st.button("질문하기") and user_input:
  with st.spinner("LLM이 답변을 생성 중..."):
    answer = pipeline.run(user_input)
  st.markdown(f"**질문:** {user_input}")
  st.markdown(f"**답변:** {answer}")

# -------------------------
# 대화 히스토리
# -------------------------
if st.checkbox("대화 히스토리 보기"):
  st.header("대화 히스토리")
  for i, chat in enumerate(pipeline.chat_history):
    role = chat["role"]
    content = chat["content"]
    st.markdown(f"**{role}**: {content}")


    
# user_input = st.text_input("질문을 입력하세요:")

# if st.button("질문하기") and user_input:
#   with st.spinner("LLM이 답변을 생성 중..."):
#     answer = pipeline.run(user_input)
#   st.markdown(f"**질문:** {user_input}")
#   st.markdown(f"**답변:** {answer}")

# # 대화 히스토리 표시
# if st.checkbox("대화 히스토리 보기"):
#   for i, chat in enumerate(pipeline.chat_history):
#     role = chat["role"]
#     content = chat["content"]
#     st.markdown(f"**{role}**: {content}")