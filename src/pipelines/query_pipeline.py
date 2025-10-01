import json
from config.settings import COLLECTION_NAME
from src.embeddings.embedder import EmbedderFactory
from src.embeddings.vectorstore_chroma import add_docs_to_chroma, get_collection, delete_collection, show_collection
from src.retrieval.bm25_helper import BM25Helper
from src.retrieval.retriever import advanced_retrieve

# TODO: 실제 경로 반영하기
# === 실제 데이터 경로 ===
DATA_PATH = "/Users/carki/Desktop/Dev/codeit_project/ai3-team3-RAG/docs_merged.jsonl"

# ==== 0) 문서 로드 ====
def load_docs():
    docs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    return docs


if __name__ == "__main__":
    # 1) 문서 불러오기
    documents = load_docs()
    print(f"✅ 불러온 문서 수: {len(documents)}")
    

    # 2) 임베딩 준비
    embedding_fn = EmbedderFactory.get_embedder(provider="openai")

    # 3) 벡터DB 연결
    collection = get_collection(COLLECTION_NAME)
    
    # 4) 문서 업로드 (배치)
    print("📌 문서 업로드 중...")
    add_docs_to_chroma(documents, collection=collection, embedding_fn=embedding_fn, batch_size=8)
    print("✅ 업로드 완료")

    # 5) BM25 인덱스 준비 (chunk 텍스트를 기반으로 하는 게 이상적)
    corpus_texts = [d.get("texts", {}).get("merged", "") for d in documents if d.get("texts", {}).get("merged", "")]
    bm25_helper = BM25Helper(corpus_texts)

    # 6) 검색 실행
    query = "아시아 육상 경기 대회 요구사항"
    results = advanced_retrieve(
        query,
        collection,                
        embedding_fn=embedding_fn, 
        bm25=bm25_helper,          
        top_k=3
    )

    print("\n== 검색 결과 ==")
    for r in results:
        print(f"merge_key:{r['meta'].get('merge_key','-')} | "
              f"hybrid:{r.get('hybrid_score',0):.4f} | "
              f"내용:{r['text'][:60]}")
        
    show_collection(collection=collection)