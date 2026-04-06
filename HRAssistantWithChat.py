import os
import shutil
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PDF_PATH = "the_nestle_hr_policy_pdf_2012.pdf"
CHROMA_DB_DIR = "chroma_db"


def load_documents():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def reset_vector_db():
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return vector_store


def load_llm():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=text_pipeline)


def create_prompt():
    template = """
You are an AI HR assistant for Nestlé.

Use only the retrieved context below to answer the question.

Rules:
- Answer in 1 to 3 short sentences.
- Do not use outside knowledge.
- Do not guess.
- Do not mention websites, links, email addresses, or external sources.
- If the answer is not clearly stated in the context, respond exactly with:
I could not find that information in the Nestlé HR policy document.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def build_qa_chain(llm, vector_store, prompt):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


def format_sources(source_documents):
    if not source_documents:
        return "No source chunks retrieved."

    formatted = []
    for i, doc in enumerate(source_documents, start=1):
        page = doc.metadata.get("page", "Unknown")
        text = doc.page_content.strip().replace("\n", " ")
        snippet = text[:500] + ("..." if len(text) > 500 else "")
        formatted.append(f"[Source {i} | Page {page + 1}]\n{snippet}")
    return "\n\n".join(formatted)


def chatbot_response(query, qa_chain):
    if not query.strip():
        return "Please enter a question about the Nestlé HR policy.", "No query entered."

    fallback = "I could not find that information in the Nestlé HR policy document."

    try:
        response = qa_chain.invoke({"query": query})
        answer = response["result"].strip()

        if fallback in answer:
            return fallback, "No supporting passage was found for this question."

        sources = format_sources(response.get("source_documents", []))
        return answer, sources

    except Exception as e:
        return f"Error: {str(e)}", "No sources available due to error."


def launch_interface(qa_chain):
    def respond(message):
        return chatbot_response(message, qa_chain)

    interface = gr.Interface(
        fn=respond,
        inputs=gr.Textbox(
            lines=2,
            placeholder="Ask a question about the Nestlé HR policy..."
        ),
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Retrieved Context", lines=18)
        ],
        title="Nestlé HR Policy Assistant",
        description="Ask questions based only on the Nestlé HR policy document."
    )

    interface.launch()


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("Resetting vector database...")
    reset_vector_db()

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Loading model...")
    llm = load_llm()

    print("Building QA chain...")
    prompt = create_prompt()
    qa_chain = build_qa_chain(llm, vector_store, prompt)

    print("Launching chatbot...")
    launch_interface(qa_chain)


if __name__ == "__main__":
    main()



'''
Conclusion

This project focused on building an AI-powered HR assistant using Nestlé’s HR policy document by using retrieval-augmented generation with a language model. 
Loaded the PDF, split it into chunks, converted them into embeddings, and stored them in a vector database so the system could retrieve relevant information. 
Then integrated the Qwen2.5-3B-Instruct model through Hugging Face and connected everything using LangChain to generate answers based only on the document. 
I built a Gradio interface so users could interact with it. 
Finally the system works well for answering policy-related questions and stays grounded in the document, which was the main goal. 
Getting everything into a single script that runs from the command line also made it feel like a complete application rather than just a small experiment.
'''

'''
Summary

### Summary

This project had a bigger learning curve than I initially expected because it required understanding how multiple components work together instead of just using a model directly. 
I had to experiment with chunk sizes, retrieval settings, and model parameters like temperature and token limits to get better performance and more accurate responses. 
One of the main challenges was dealing with hallucinations, where the model would add information not present in the document, 
which I improved by refining the prompt to be more strict and limiting responses to only the provided context. 
I also made changes to the retrieval process by adjusting chunk size, increasing the number of retrieved documents, 
and resetting the vector database to avoid stale data issues. Performance was initially slow until I optimized the model settings. 
Another issue I ran into was that the system would show retrieved context even when no valid answer existed, which I fixed by updating the logic to only display context when a real answer is found. 
I also realized that the system doesn’t handle certain types of questions well, like counting or very specific queries, because it only retrieves a limited number of chunks. 
Overall, this project helped me understand the difference between just running a model and actually building a structured AI system, and it gave me a much better sense of how to control model behavior, 
improve retrieval quality, and make the system more reliable and user-friendly.
'''    