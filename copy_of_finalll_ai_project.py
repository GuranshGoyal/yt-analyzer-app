
########
from pyngrok import ngrok

# Replace with your actual token
ngrok.set_auth_token("2vrqwdGeE49a4VViZ2zXpC9dd0G_ohmncX1pzjR6uVxv1NrN")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn



#########



def extract_video_id(url: str) -> str:
    import re

    # Patterns for different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # For regular and shortened URLs
        r'youtu\.be\/([0-9A-Za-z_-]{11})',   # For youtu.be URLs
        r'shorts\/([0-9A-Za-z_-]{11})',      # For YouTube Shorts
    ]

    # Clean the URL
    url = url.strip()

    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

# def fetch_comments_and_set_url(url) :
import googleapiclient.discovery
import pandas as pd
import time

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDyhMnSmfcBgyP8Rf3ROC4G_V-T8otmAiE"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY
)

comments = []
video_id = extract_video_id("https://youtu.be/gz8chxY7elU?si=xVmyJMqqb05RR0fN")
next_page_token = None
total_fetched = 0

while True:
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        pageToken=next_page_token
    )
    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['updatedAt'],
            comment['likeCount'],
            comment['textDisplay']
        ])
    total_fetched += len(response['items'])
    print(f"Fetched {total_fetched} comments...")

    next_page_token = response.get('nextPageToken')
    if not next_page_token:
        break

    # Optional: Respect YouTube rate limits
    time.sleep(0.1)

df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
print(f"Total comments fetched: {len(df)}")

df.head()


print('done')

import getpass
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAnkoOk8BA4_rO4lJbYx2TAJmgEe6DOcrc"
# "AIzaSyCqA6CjPqlIAEANndOv8NQbIXvuaptuFpI"


from dotenv import load_dotenv

load_dotenv(override=True)  # Force override old values

os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
print(llm)

from langchain_core.documents import Document
import pandas as pd

# Load CSV

# Convert to Document objects
documents = [
    Document(
        page_content=row["text"],
        metadata={
            "author": row["author"],
            "likes": row["like_count"]
        }
    )
    for _, row in df.iterrows()
]
documents

print(len(documents))
documents[0]

#Preparing a list of youtube comments to pass into the prompt for feature 1: summarization

# Prepare youtube_comments list from Document objects
youtube_comments = [doc.page_content for doc in documents]

# Check the first few comments to ensure it's working
print(youtube_comments[:5])  # Print first 5 comments


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed this line
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


print(documents[0])
#print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Load the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# HuggingFace token
hf_token = os.getenv("HF_TOKEN") or "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" # Guransh's new key


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 8}) #retrieve top 8 chunks/elements/embeddings similar to query embedding
)

def summarize_comments(youtube_comments, chunk_size=10000):
    """
    Summarizes YouTube comments by splitting them into chunks if needed.
    """
    summarization_prompt = (
        "Summarize the following YouTube comments into a concise paragraph highlighting key points, "
        "sentiments, and common themes:\n\n"
    )

    # Join all comments
    all_text = "\n".join(youtube_comments)
    chunks = []

    # Split long comment text into smaller chunks of ~chunk_size characters
    while len(all_text) > chunk_size:
        split_at = all_text[:chunk_size].rfind('\n')  # break at last newline
        if split_at == -1:
            split_at = chunk_size
        chunks.append(all_text[:split_at])
        all_text = all_text[split_at:]

    if all_text.strip():
        chunks.append(all_text.strip())

    summaries = []
    print(f"Total chunks created: {len(chunks)}")

    # Summarize each chunk
    for chunk in chunks:
        prompt = summarization_prompt + chunk
        try:
            chunk_summary = qa.invoke(prompt)
            summaries.append(chunk_summary)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk]: {str(e)}")


    # Final summary of all the chunk summaries (optional)
    try:
        final_summary = qa.run(summarization_prompt + "\n".join(summaries))
        return final_summary
    except Exception as e:
        return f"Error during final summarization: {str(e)}"


def get_answer(user_prompt):
    """
    Accepts a user's prompt, modifies it with additional context, and returns the answer.
    """
    # You can modify this prefix to guide the LLM (acts like few-shot prompt tuning)
    additional_instruction = (
        "Answer the question using only the information available in the YouTube comments. "
        "If information is insufficient, say so clearly.\n\n"
    )

    # Combine it with the user's prompt
    modified_prompt = additional_instruction + user_prompt

    # Get the answer from the QA chain
    try:
      answer = qa.run(modified_prompt)
      return answer
    except Exception as e:
      return f"An error occured: {str(e)}"

def get_similar_comments(user_input, top_k):
    """
    Retrieves top k similar comments from the vector database based on the input text.

    Args:
        user_input (str): The input text/phrase to compare against
        top_k (int): Number of similar comments to retrieve (default: 10)

    Returns:
        List of tuples containing (comment_text, similarity_score)
    """
    try:
        # Get similar documents using similarity search
        similar_docs = db.similarity_search_with_score(user_input, k=top_k)

        # Extract comments and scores
        results = []
        for doc, score in similar_docs:
            result_dict = {
                'comment_text': doc.page_content,
                'similarity_score': float(1 - score),  # ✅ Convert to Python float
                'author': doc.metadata.get('author', 'Unknown'),  # Default to 'Unknown' if author not found
                'likes': doc.metadata.get('likes', 0)  # Default to 0 if likes not found
            }
            results.append(result_dict)
        print(results)

        # Convert to JSON string
        import json
        return json.dumps(results)

    except Exception as e:
        print(f"An error occurred while retrieving similar comments: {str(e)}")
        return []

get_similar_comments("hello", 5 )
print("kunal")



# !pip install fastapi uvicorn pyngrok nest-asyncio

# !pip install flask-cors

# from pyngrok import ngrok

# # Replace with your actual token
# ngrok.set_auth_token("2vrqwdGeE49a4VViZ2zXpC9dd0G_ohmncX1pzjR6uVxv1NrN")

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import nest_asyncio
# from pyngrok import ngrok
# import uvicorn

# # Create FastAPI app
# app = FastAPI()

# # Allow frontend (localhost:3000) to access the API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # or ["*"] for all
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Sample prediction function
# def predict_model(yturl , query , count ):
#     comments = [
#       {
#           "id": "1",
#           "author": "Kunal Sonkar",
#           "text": "This is a comment",
#           "likes": 10,
#           "timestamp": "2025-04-20T14:30:00",
#       },
#       {
#           "id": "2",
#           "author": "Mohammad Mojij Ansari",
#           "text": "This is another comment",
#           "likes": 5,
#           "timestamp": "2025-04-20T15:00:00",
#       }
#   ]

#     return comments # Replace with your real model


# class InputData(BaseModel):
#     yturl: str

# @app.post("/seturl")
# async def predict(data: InputData):
#     fetch_comments_and_set_url(data.yturl)
#     return {"success": True}

# Input schema

# Create FastAPI app
app = FastAPI()

# Allow frontend (localhost:3000) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all
    # allow_origins=["http://localhost:3000"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    yturl: str

@app.post("/seturl")
async def predict(data: InputData):
    # fetch_comments_and_set_url(data.yturl)
    url = data.yturl
    print(url)
    return {"success": True}

class InputData(BaseModel):
    yturl: str
    query: str
    count: int

@app.post("/comment")
async def predict(data: InputData):
    return {"comment": get_similar_comments(data.query, data.count)}
    # list of top similar comments:  get_similar_comments(data.query, data.count)

class InputData(BaseModel):
    url: str

#yaha pai url duga tu summary bhej
@app.post("/yturl")
async def predict(data: InputData):
    print(data)
    print("Fetching comments for URL: kunal ")
    return {"summary": summarize_comments(youtube_comments)}


class InputData(BaseModel):
    url: str
    query :str

# yaha mai tuhje url or chat message duga tu fir reply bhej dena
@app.post("/chat")
async def predict(data: InputData):
    print(data)
    responce = get_answer(data.query)
    return {"message": responce }



# Setup ngrok and run the server
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)

nest_asyncio.apply()
uvicorn.run(app, port=8000)