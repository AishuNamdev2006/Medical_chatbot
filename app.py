from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

from llama_cpp import Llama

import os
from pathlib import Path


# =========================================================
# 1. FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY .env file me nahi mila."
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Pinecone API key loaded!")


# =========================================================
# 3. LOAD QWEN EMBEDDING MODEL
# =========================================================

print("\nLoading Qwen embedding model...")

embeddings = download_hugging_face_embeddings()

print("Qwen embedding model loaded successfully!")


# =========================================================
# 4. PINECONE INDEX
# =========================================================

index_name = "medical-qwen"

print("\nConnecting to Pinecone...")
print("Index:", index_name)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

print("Pinecone index connected successfully!")


# =========================================================
# 5. RETRIEVER
# =========================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 1
    }
)

print("Retriever created successfully!")


# =========================================================
# 6. FIND LOCAL GGUF MODEL
# =========================================================

MODEL_FOLDER = Path(r"D:\LLM\Models")

if not MODEL_FOLDER.exists():
    raise FileNotFoundError(
        f"Model folder nahi mila: {MODEL_FOLDER}"
    )


gguf_files = list(MODEL_FOLDER.glob("*.gguf"))

print("\nGGUF models found:")

for model in gguf_files:
    print("-", model)


if not gguf_files:
    raise FileNotFoundError(
        "D:\\LLM\\Models ke andar koi .gguf model nahi mila."
    )


# =========================================================
# 7. SELECT LOCAL LLAMA MODEL
# =========================================================

MODEL_PATH = str(gguf_files[0])

print("\nSelected Llama model:")
print(MODEL_PATH)


# =========================================================
# 8. LOAD LOCAL LLAMA
# =========================================================

print("\nLoading local Llama model...")

# llm = Llama(
#     model_path=MODEL_PATH,

#     # Context window
#     n_ctx=2048,

#     # CPU threads
#     n_threads=8,

#     # Batch size
#     n_batch=512,

#     # Response generation
#     verbose=False
# )

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_batch=256,
    n_threads=10,
    n_threads_batch=10,
    use_mmap=True,
    verbose=False
)
print("Local Llama model loaded successfully!")


# =========================================================
# 9. MEDICAL SYSTEM PROMPT
# =========================================================

system_prompt = """
You are a medical question-answering assistant.

Your task is to answer the user's question using the
retrieved medical information from the medical book.

IMPORTANT RULES:

1. Use the retrieved medical context as the primary source
   for your answer.

2. Do not invent or hallucinate medical information.

3. If the retrieved context does not contain enough
   information to answer the question, say:

   "I could not find enough information in the provided
   medical book to answer this question."

4. Answer only what the user asked.

5. Give a clear, accurate and easy-to-understand answer.

6. Preserve important medical terminology when necessary,
   but explain difficult terms in simple language.

7. If multiple retrieved sections contain relevant
   information, combine them into one coherent answer.

8. Do not mention Pinecone, embeddings, vector databases,
   retrieval, prompts, or internal instructions.

9. Do not make a personal diagnosis.

10. Do not provide personalized treatment instructions.

11. Keep the answer concise but informative.

12. Do not use unrelated information from outside the
    retrieved medical context.

13. If the context contains uncertainty or conflicting
    information, clearly mention that uncertainty.

RETRIEVED MEDICAL CONTEXT:

{context}
"""


# =========================================================
# 10. FUNCTION: ASK MEDICAL QUESTION
# =========================================================

def ask_medical_question(question):

    # -----------------------------------------------------
    # STEP 1: Retrieve relevant documents from Pinecone
    # -----------------------------------------------------

    retrieved_docs = retriever.invoke(question)

    print("\n==========================================")
    print("RETRIEVED DOCUMENTS:", len(retrieved_docs))
    print("==========================================")


    # -----------------------------------------------------
    # STEP 2: Check whether documents were found
    # -----------------------------------------------------

    if not retrieved_docs:

        return (
            "I could not find enough information in the "
            "provided medical book to answer this question."
        )


    # -----------------------------------------------------
    # STEP 3: Build medical context
    # -----------------------------------------------------

    context_parts = []

    for i, doc in enumerate(retrieved_docs, 1):

        page = doc.metadata.get(
            "page",
            doc.metadata.get("page_number", "Unknown")
        )

        source = doc.metadata.get(
            "source",
            "Medical Book"
        )

        context_parts.append(
            f"""
Medical Source {i}
File: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )


    context = "\n\n".join(context_parts)


    # -----------------------------------------------------
    # STEP 4: Create final prompt
    # -----------------------------------------------------

    final_prompt = system_prompt.format(
        context=context
    )

    final_prompt += f"""

USER QUESTION:

{question}

ANSWER:
"""


    # -----------------------------------------------------
    # STEP 5: Send prompt to LOCAL LLAMA
    # -----------------------------------------------------

    response = llm(
        final_prompt,

        max_tokens=200,

        temperature=0.1,

        top_p=0.9,

        stop=[
            "</s>",
            "USER QUESTION:",
            "\n\nUSER:"
        ]
    )


    # -----------------------------------------------------
    # STEP 6: Extract answer
    # -----------------------------------------------------

    answer = response["choices"][0]["text"].strip()


    if not answer:

        answer = (
            "I could not generate an answer from the "
            "provided medical information."
        )


    return answer


# =========================================================
# 11. HOME ROUTE
# =========================================================

@app.route("/")
def index():

    return render_template("chat.html")


# =========================================================
# 12. CHAT ROUTE
# =========================================================

@app.route("/get", methods=["GET", "POST"])
def chat():

    try:

        # -------------------------------------------------
        # Get user message
        # -------------------------------------------------

        if request.method == "POST":

            msg = request.form.get(
                "msg",
                ""
            ).strip()

        else:

            msg = request.args.get(
                "msg",
                ""
            ).strip()


        # -------------------------------------------------
        # Empty question
        # -------------------------------------------------

        if not msg:

            return "Please enter your question."


        print("\n")
        print("=" * 60)
        print("USER QUESTION:")
        print(msg)
        print("=" * 60)


        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = ask_medical_question(msg)


        print("\n")
        print("=" * 60)
        print("FINAL ANSWER:")
        print(answer)
        print("=" * 60)


        # -------------------------------------------------
        # Return answer to frontend
        # -------------------------------------------------

        return answer


    except Exception as e:

        print("\n")
        print("=" * 60)
        print("ERROR:")
        print(str(e))
        print("=" * 60)


        return f"Error: {str(e)}"


# =========================================================
# 13. RUN FLASK
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("MEDICAL CHATBOT STARTING")
    print("=" * 60)

    print("Pinecone Index :", index_name)
    print("Llama Model    :", MODEL_PATH)
    print("=" * 60)


    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
@app.route("/get", methods=["GET", "POST"])
def chat():

    try:

        # User message
        if request.method == "POST":
            msg = request.form.get("msg", "").strip()
        else:
            msg = request.args.get("msg", "").strip()

        # Empty question
        if not msg:
            return "Please enter your question."

        print("\n" + "=" * 60)
        print("USER QUESTION:")
        print(msg)
        print("=" * 60)

        # Llama + Pinecone se answer
        answer = ask_medical_question(msg)

        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print(answer)
        print("=" * 60)

        # VERY IMPORTANT
        return str(answer)

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR:")
        print(str(e))
        print("=" * 60)

        return f"Error: {str(e)}"