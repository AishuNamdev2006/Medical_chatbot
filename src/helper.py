from langchain_huggingface import HuggingFaceEmbeddings


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings