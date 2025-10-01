import os
import chromadb
from chromadb.config import Settings

def main():
    # 환경 변수에서 Chroma 서버 주소 가져오기
    chroma_server = os.getenv("CHROMA_SERVER", "http://chroma:8000")
    host = chroma_server.replace("http://", "").split(":")[0]
    port = int(chroma_server.split(":")[-1])

    print(f"🔗 ChromaDB 서버 연결 시도: {host}:{port}")

    # 클라이언트 생성
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=True)
    )

    # 연결 확인 (heartbeat)
    heartbeat = client.heartbeat()
    print(f"✅ Heartbeat: {heartbeat}")

    # collections 확인 (아직 아무것도 없을 가능성 큼)
    collections = client.list_collections()
    print(f"📂 현재 collections: {collections}")

if __name__ == "__main__":
    main()
