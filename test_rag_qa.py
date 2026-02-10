from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store
vector_store = FAISS.load_local(
    "vector_store",
    embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Load LLM (small, CPU-safe)
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Guardrail prompt
def answer_question(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an enterprise AI assistant.
Answer the question using ONLY the context below.
If the answer is not present, respond with:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return llm(prompt)

# Test questions
print("\nIN-SCOPE QUESTION:")
print(answer_question("What is considered workplace harassment?"))

print("\nOUT-OF-SCOPE QUESTION:")
print(answer_question("Who is the president of the United States?"))
