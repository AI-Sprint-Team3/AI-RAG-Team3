from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import OPENAI_API_KEY

class EmbedderFactory:
    """
    여러 임베딩 모델을 관리하는 팩토리
    
    # OpenAI 사용
    embedding_fn = EmbedderFactory.get_embedder(provider="openai")

    # HuggingFace 사용
    embedding_fn = EmbedderFactory.get_embedder(
        provider="huggingface",
        model="jhgan/ko-sroberta-multitask"
    )

    """

    @staticmethod
    def get_embedder(provider="openai", model=None):
        if provider == "openai":
            print(f"[Log - Dev] Provider OpenAI 진입")
            return OpenAIEmbeddings(
                model=model or "text-embedding-3-large",
                api_key=OPENAI_API_KEY
            )
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"❌ 지원하지 않는 provider: {provider}")
