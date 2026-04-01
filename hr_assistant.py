import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

pdfPath = "the_nestle_hr_policy_pdf_2012.pdf"

def loadDocuments():
    loader = PyPDFLoader(pdfPath)
    documents = loader.load()
    return documents

def splitDocuments(documents):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = textSplitter.split_documents(documents)
    return chunks

def createVectorStore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorStore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorStore

def loadLlm():
    modelName = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    textPipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=textPipeline)
    return llm

def buildQaChain(llm, vectorStore, prompt):
    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

    qaChain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qaChain

def createPrompt():
    promptTemplate = """
You are an AI HR assistant for Nestlé.

Answer the question using only the context provided below.

Rules:
- Do not use outside knowledge.
- Do not mention websites, links, email addresses, or external sources.
- Do not add extra advice or explanations.
- If the answer is not in the context, respond with exactly:
I could not find that information in the Nestlé HR policy document.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=promptTemplate,
        input_variables=["context", "question"]
    )

    return prompt

def chatbotResponse(userQuery, qaChain):
    if not userQuery.strip():
        return "Please enter a question about the Nestlé HR policy."

    response = qaChain.invoke({"query": userQuery})
    answer = response["result"].strip()

    if "I could not find that information in the Nestlé HR policy document." in answer:
        return "I could not find that information in the Nestlé HR policy document."

    return answer

def launchInterface(qaChain):
    def respond(message):
        return chatbotResponse(message, qaChain)

    interface = gr.Interface(
        fn=respond,
        inputs=gr.Textbox(lines=2, placeholder="Ask a question about the Nestlé HR policy..."),
        outputs="text",
        title="Nestlé HR Policy Assistant",
        description="Ask questions about the Nestlé HR policy document."
    )

    interface.launch()

def main():
    documents = loadDocuments()
    chunks = splitDocuments(documents)
    vectorStore = createVectorStore(chunks)
    llm = loadLlm()
    prompt = createPrompt()
    qaChain = buildQaChain(llm, vectorStore, prompt)
    launchInterface(qaChain)

if __name__ == "__main__":
    main()

'''
Conclusion

This project focused on building an AI-powered HR assistant using Nestlé’s HR policy document by combining retrieval-augmented generation with a language model and a simple interface. 
Loaded the PDF, split it into chunks, converted them into embeddings, and stored them in a vector database so the system could retrieve relevant information. 
Then integrated the Qwen2.5-3B-Instruct model through Hugging Face and connected everything using LangChain to generate answers based only on the document. 
I built a Gradio interface so users could interact with it. 
Finally the system works well for answering policy-related questions and stays grounded in the document, which was the main goal. 
Getting everything into a single script that runs from the command line also made it feel like a complete application rather than just a small experiment.
'''

'''
Summary

This project had a bigger learning curve than I initially expected because it required understanding how multiple components work together instead of just using a model directly. 
I had to experiment with chunk sizes, retrieval settings, and model parameters like temperature and token limits to get better performance and more accurate responses. 
One of the main challenges was dealing with hallucinations, where the model would add information not present in the document, which I fixed by refining the prompt to be more strict. 
Performance was also an issue at first, since responses were very slow until I adjusted the model settings. 
Another obstacle was realizing that the system doesn’t handle certain types of questions well, like counting how many times a word appears, because it only looks at a few chunks at a time. 
This project helped me understand the difference between just running a model and actually building a structured AI system, and it gave me a better sense of how to control model behavior and improve reliability.
'''