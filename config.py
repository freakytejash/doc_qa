from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"

    # Embeddings
    embedding_provider: Literal["openai", "huggingface"] = "huggingface"
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "all-MiniLM-L6-v2"

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    # Vision (image description during ingestion)
    vision_enabled: bool = True
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    @property
    def active_embedding_model(self) -> str:
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.hf_embedding_model


settings = Settings()
