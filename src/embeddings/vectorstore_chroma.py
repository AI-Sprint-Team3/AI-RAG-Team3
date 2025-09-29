from chromadb import HttpClient
from config.settings import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME
import os
import time
import pickle
import gc
from typing import List, Dict
import numpy as np

# === 클라이언트 연결 ===
chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def get_collection(name: str = COLLECTION_NAME):
    """기존 collection 가져오거나 없으면 생성"""
    return chroma_client.get_or_create_collection(name=name)


def add_docs_to_chroma(docs: List[Dict],
                            collection,
                            embedding_fn,
                            chunk_size=800,
                            chunk_overlap=100,
                            batch_size=4,
                            cache_dir="cache"):
    """
    메모리 안전하게 Chroma에 문서 업로드
    
    docs: 문서 리스트
    collection: 크로마 컬렉션
    embedding_fn: 임베딩 인스턴스
    chunk_size, chunk_overlap: 문서 chunk 분할 파라미터
    batch_size: 임베딩 batch 크기
    cache_dir: 임베딩 캐시 저장 폴더
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    total_chunks = 0
    
    for doc in docs:
        doc_id = doc.get("doc_id") or doc.get("merge_key") or f"doc_{int(time.time()*1000)}"
        text = doc.get("texts", {}).get("merged", "")
        if not text.strip():
            print(f"⚠️ 텍스트 없음: {doc_id}")
            continue

        # === 텍스트 chunk 분리 ===
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - chunk_overlap if end - chunk_overlap > start else end

        # === 메타데이터 기본값 ===
        base_meta = {
            "merge_key": doc.get("merge_key", ""),
            "doc_id": doc.get("doc_id", ""),
            "chars_merged": len(text),
        }

        # === 배치 단위 임베딩 및 업로드 ===
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = []
            batch_metas = []

            # 캐시 파일 경로
            batch_cache_file = os.path.join(cache_dir, f"{doc_id}__{i}.pkl")

            # === 임베딩 로드 또는 생성 ===
            if os.path.exists(batch_cache_file):
                import pickle
                with open(batch_cache_file, "rb") as f:
                    vecs = pickle.load(f)
            else:
                # 재시도 로직
                for attempt in range(3):
                    try:
                        vecs = embedding_fn.embed_documents(batch_chunks)
                        break
                    except Exception as e:
                        print(f"임베딩 실패, 재시도 {attempt+1}/3: {e}")
                        time.sleep(2)
                else:
                    print(f"⚠️ 임베딩 3회 실패, 스킵: doc_id={doc_id}, batch={i}")
                    continue

                # 캐시 저장
                import pickle
                with open(batch_cache_file, "wb") as f:
                    pickle.dump(vecs, f)

            # === 컬렉션 추가 ===
            for j, chunk_text in enumerate(batch_chunks):
                chunk_index = i + j
                chunk_id = f"{doc_id}__chunk_{chunk_index}"
                meta = base_meta.copy()
                meta.update({
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "chunk_len": len(chunk_text),
                })
                batch_ids.append(chunk_id)
                batch_metas.append(meta)

            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=vecs,
                metadatas=batch_metas
            )

            total_chunks += len(batch_chunks)
            print(f"✅ 업로드 중... {total_chunks} chunks 누적 완료 (doc_id={doc_id})")

    print(f"🎉 전체 업로드 완료: {total_chunks} chunks added")

def delete_collection(collection):
    """ DB에 존재하는 콜렉션을 삭제합니다 """
    chroma_client.delete_collection(collection.name)
    print(f"🗑 Deleted: {collection.name}")
        
def show_collection(collection):
    """ DB에 존재하는 콜렉션을 가져옵니다 """
    print("📂 존재하는 Collection:")
    print("-", collection.name)
