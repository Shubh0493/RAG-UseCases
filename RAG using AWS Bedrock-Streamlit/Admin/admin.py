import boto3
import streamlit as st
import os
import uuid
import langchain
from PIL import Image
## s3_client
s3_client = boto3.client("s3", verify=False)
BUCKET_NAME = ""

## Bedrock
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", verify=False)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name="my_faiss"
    #change hardcoded values
    folder_path=r"\tmp"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    llm=Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client,
                model_kwargs={'max_tokens_to_sample': 512})
    return llm
#change hardcoded values
folder_path=r"\tmp"

# get_response()
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']


## main method
def main():
    st.title("Novatio RAG Demo")
    #change hardcoded values
    image = Image.open(r"\tmp\images")
    st.image(image=image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
            load_index()

            dir_list = os.listdir(folder_path)
            st.write(f"Files and Directories used in RAG Solution")
            st.write(dir_list)

            ## create index
            faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path = folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
            )

            st.write("INDEX IS READY")
            question = st.text_input("Please ask your question")
            if st.button("Ask Question"):
                with st.spinner("Querying..."):
                    llm = get_llm()

                    # get_response
                    st.write(get_response(llm, faiss_index, question))
                    st.success("Done")
        else:
            st.write("Error!! Please check logs.")

    

if __name__ == "__main__":
    main()