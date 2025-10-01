import numpy as np

def distance_to_similarity(distance, metric="cosine"):
    # distance -> similarity 로 변환 (간단한 처리)
    if metric == "cosine":
        # Chroma cosine distance commonly: distance = 1 - cos_sim (so similarity = 1 - distance)
        return 1.0 - distance
    elif metric == "euclidean":
        # euclidean: 거리가 작을수록 유사. 간단 변환:
        return 1.0 / (1.0 + distance)
    else:
        return 1.0 - distance

def minmax_norm(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return a
    
    mn, mx = a.min(), a.max()
    if mx == mn:
        return np.ones_like(a) * 0.5
    
    return (a - mn) / (mx - mn)

def advanced_retrieve(query, collection, embedding_fn, bm25=None,
                      agency_filter=None, top_k=3, k_large=40,
                      use_bm25=True, use_mmr=True, metric="cosine"):
    # 1) query embedding
    query_emb = embedding_fn.embed_query(query)

    # 2) dense search
    results = collection.query(
        query_embeddings=[query_emb], 
        n_results=k_large
    )
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    
    # for c in results:
    #     similarity = 1 - c["dist"]   # 또는 distance_to_similarity(c["dist"])
    #     print(f"유사도: {similarity:.4f} | merge_key: {c['meta']['merge_key']}")

    # 3) candidates 구성
    candidates = []
    for doc_text, meta, dist, did in zip(docs, metas, dists, ids):
        sim = distance_to_similarity(dist, metric=metric)
        candidates.append({
            "id": did, 
            "text": doc_text, 
            "meta": meta, 
            "dist": dist, 
            "sim": sim
        })

    # 4) metadata 필터링 (merge_key)
    if agency_filter:
        filtered = [c for c in candidates if c["meta"].get("merge_key") == agency_filter]
        if not filtered:
            # fallback: 전체 후보에서 top_k 리턴
            print("⚠️ 메타 필터링 결과 없음 → 전체 후보 사용")
        else:
            candidates = filtered

    # 5) BM25 스코어 결합
    if use_bm25 and bm25 is not None:
        # bm25 corpus index mapping 필요: meta에 'corpus_idx' 또는 별도 map
        bm25_scores = []
        for c in candidates:
            idx = c["meta"].get("corpus_idx", None)
            if idx is not None:
                bm25_scores.append(bm25.get_scores(query.split())[idx])
            else:
                bm25_scores.append(0.0)
        # 정규화
        bm25_norm = minmax_norm(bm25_scores)
        dense_norm = minmax_norm([c["sim"] for c in candidates])
        for i, c in enumerate(candidates):
            c["hybrid_score"] = 0.6 * dense_norm[i] + 0.4 * bm25_norm[i]
        candidates = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
    else:
        candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)

    # 6) MMR (중복제거 + 다양성)
    if use_mmr and len(candidates) > top_k:
        selected = []
        # precompute embeddings of candidate texts (or use meta-stored embeddings)
        cand_embs = [embedding_fn.embed_query(c["text"]) for c in candidates]
        cand_embs = [np.array(e) for e in cand_embs]

        # iterative MMR
        lam = 0.7  # relevance vs diversity
        # start: highest relevance
        selected.append(candidates[0])
        sel_embs = [cand_embs[0]]
        for i in range(1, len(candidates)):
            c = candidates[i]
            # relevance = candidates[i]["sim"] (already)
            # diversity_penalty = max cosine similarity to selected
            sim_to_selected = max(
                (np.dot(cand_embs[i], s) / (np.linalg.norm(cand_embs[i]) * np.linalg.norm(s) + 1e-8))
                for s in sel_embs
            )
            mmr_score = lam * c.get("sim", 0.0) - (1 - lam) * sim_to_selected
            c["_mmr_score"] = mmr_score
            # 선택 로직: keep top mmr up to top_k
            sel_sorted = sorted([*selected, c], key=lambda x: x.get("_mmr_score", x.get("sim", 0)), reverse=True)
            if len(sel_sorted) > len(selected) and len(selected) < top_k:
                selected.append(c)
                sel_embs.append(cand_embs[i])
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    return candidates[:top_k]

# def advanced_retrieve(query, bm25_helper, top_k=3, k_dense=20, alpha=0.6, agency_filter=None):
#     """
#     Dense + BM25 Hybrid Retrieval
#     """
#     # 1) === Dense Search ===
#     q_emb = embedding_fn.embed_query(query)
#     dense_res = query_collection(q_emb, top_k=k_dense)
#     dense_docs = dense_res["documents"][0]
#     dense_metas = dense_res["metadatas"][0]
#     dense_dists = dense_res["distances"][0]

#     candidates = {}
#     for doc, meta, d in zip(dense_docs, dense_metas, dense_dists):
#         key = meta.get("merge_key") or meta.get("doc_id")
#         candidates[key] = {
#             "text": doc,
#             "meta": meta,
#             "dense_sim": distance_to_similarity(d),
#             "bm25_score": 0.0
#         }

#     # 2) === BM25 Search ===
#     bm25_scores = bm25_helper.get_scores(query)
#     for idx, score in enumerate(bm25_scores):
#         key = bm25_helper.corpus_texts[idx][:30]  # 텍스트 앞부분을 key로 임시 매칭
#         if key in candidates:
#             candidates[key]["bm25_score"] = score
#         else:
#             candidates[key] = {
#                 "text": bm25_helper.corpus_texts[idx],
#                 "meta": {},
#                 "dense_sim": 0.0,
#                 "bm25_score": score
#             }

#     # 3) === Hybrid Score (Normalized) ===
#     dense_vals = [v["dense_sim"] for v in candidates.values()]
#     bm25_vals = [v["bm25_score"] for v in candidates.values()]
#     dense_norm = minmax_norm(dense_vals)
#     bm25_norm = minmax_norm(bm25_vals)

#     final_candidates = []
#     for (key, v), dn, bn in zip(candidates.items(), dense_norm, bm25_norm):
#         hybrid_score = alpha * dn + (1 - alpha) * bn
#         final_candidates.append({
#             "key": key,
#             "text": v["text"],
#             "meta": v["meta"],
#             "dense_sim": dn,
#             "bm25_norm": bn,
#             "hybrid_score": hybrid_score
#         })

#     final_candidates = sorted(final_candidates, key=lambda x: -x["hybrid_score"])
#     return final_candidates[:top_k]
