import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def main():
    # 환경 변수에서 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_server = os.getenv("CHROMA_SERVER", "http://chroma:8000")
    persist_dir = "./chroma_data"  # docker-compose에서 마운트된 경로

    if not openai_api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    print("🔑 OpenAI API Key Loaded")
    print(f"🔗 Connecting to Chroma at {chroma_server}")

    # OpenAI 임베딩 모델 설정
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Chroma 벡터 DB 연결 (persist_directory 사용)
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # ✅ 연결 확인: DB에 저장된 문서 수 출력
    try:
        count = db._collection.count()  # 내부적으로 컬렉션 카운트
        print(f"✅ Chroma 연결 성공! 현재 문서 수: {count}")
    except Exception as e:
        print(f"❌ Chroma 연결 실패: {e}")

if __name__ == "__main__":
    main()
