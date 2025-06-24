from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import re
import time
import json
import pandas as pd
from typing import Optional, List, Dict, Any

# Google API imports
import googleapiclient.discovery

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyAnkoOk8BA4_rO4lJbYx2TAJmgEe6DOcrc"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# YouTube API setup
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY") or "AIzaSyDyhMnSmfcBgyP8Rf3ROC4G_V-T8otmAiE"

# Initialize YouTube API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY
)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
current_video_id = None
df = None
documents = None
db = None
qa = None

# Helper functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats"""
    if not url:
        return None
        
    patterns = [
        r'(?:v=|\/)([-\w]{11}).*',  # For regular and shortened URLs
        r'youtu\.be\/([-\w]{11})',   # For youtu.be URLs
        r'shorts\/([-\w]{11})',      # For YouTube Shorts
    ]

    url = url.strip()

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

def fetch_comments(video_id: str) -> pd.DataFrame:
    """Fetch comments for a YouTube video"""
    if not video_id:
        raise ValueError("Invalid video ID")
        
    comments = []
    next_page_token = None
    total_fetched = 0

    try:
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

            # Respect YouTube rate limits
            time.sleep(0.1)

        df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        print(f"Total comments fetched: {len(df)}")
        return df
    except Exception as e:
        print(f"Error fetching comments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching comments: {str(e)}")

def setup_vector_db(df: pd.DataFrame):
    """Set up vector database from comments"""
    global documents, db, qa
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=row["text"],
            metadata={
                "author": row["author"],
                "likes": row["like_count"],
                "id": str(i)  # Add an ID for each comment
            }
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]
    
    # Create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    db = FAISS.from_documents(text_chunks, embedding_model)
    
    # Save to disk
    DB_FAISS_PATH = "vectorstore/db_faiss"
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 8})
    )
    
    return documents

def summarize_comments(youtube_comments, chunk_size=10000):
    """Summarize YouTube comments"""
    if not qa:
        raise ValueError("QA system not initialized. Please set a YouTube URL first.")
        
    summarization_prompt = (
        "Summarize the following YouTube comments into a concise paragraph highlighting key points, "
        "sentiments, and common themes:\n\n"
    )

    # Join all comments
    all_text = "\n".join(youtube_comments)
    chunks = []

    # Split long comment text into smaller chunks
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

    # Final summary of all the chunk summaries
    try:
        final_summary = qa.invoke(summarization_prompt + "\n".join(summaries))
        return final_summary['result']
    except Exception as e:
        return f"Error during final summarization: {str(e)}"

def get_answer(user_prompt):
    """Get answer to user query about comments"""
    if not qa:
        raise ValueError("QA system not initialized. Please set a YouTube URL first.")
        
    additional_instruction = (
        "Answer the question using only the information available in the YouTube comments. "
        "If information is insufficient, say so clearly.\n\n"
    )

    modified_prompt = additional_instruction + user_prompt

    try:
        answer = qa.invoke(modified_prompt)
        return answer['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_similar_comments(user_input, top_k):
    """Get top-k similar comments"""
    if not db:
        raise ValueError("Vector database not initialized. Please set a YouTube URL first.")
        
    try:
        # Get similar documents
        similar_docs = db.similarity_search_with_score(user_input, k=top_k)

        # Extract comments and scores
        results = []
        for doc, score in similar_docs:
            result_dict = {
                'id': doc.metadata.get('id', '0'),
                'author': doc.metadata.get('author', 'Unknown'),
                'comment_text': doc.page_content,
                'likes': int(doc.metadata.get('likes', 0)),
                'similarity_score': float(1 - score)
            }
            results.append(result_dict)

        return results
    except Exception as e:
        print(f"Error retrieving similar comments: {str(e)}")
        return []

# API Models
class SetUrlRequest(BaseModel):
    yturl: str

class CommentRequest(BaseModel):
    yturl: str
    query: str
    count: int

class SummaryRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    url: str
    query: str

# API Endpoints
@app.post("/seturl")
async def set_url(data: SetUrlRequest):
    """Set YouTube URL and fetch comments"""
    global current_video_id, df, documents
    
    try:
        video_id = extract_video_id(data.yturl)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
        # Only fetch comments if video ID has changed
        if video_id != current_video_id:
            current_video_id = video_id
            df = fetch_comments(video_id)
            documents = setup_vector_db(df)
            
        return {"success": True, "message": "URL set successfully", "video_id": video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/comment")
async def get_comments(data: CommentRequest):
    """Get top-k similar comments"""
    try:
        # Check if we need to update the video
        video_id = extract_video_id(data.yturl)
        if video_id != current_video_id:
            # Set the new URL first
            await set_url(SetUrlRequest(yturl=data.yturl))
            
        # Get similar comments
        comments = get_similar_comments(data.query, data.count)
        return {"comment": json.dumps(comments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yturl")
async def get_summary(data: SummaryRequest):
    """Get summary of YouTube comments"""
    try:
        # Check if we need to update the video
        video_id = extract_video_id(data.url)
        if video_id != current_video_id:
            # Set the new URL first
            await set_url(SetUrlRequest(yturl=data.url))
            
        # Get comment texts
        youtube_comments = [doc.page_content for doc in documents]
        
        # Generate summary
        summary = summarize_comments(youtube_comments)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(data: ChatRequest):
    """Chat about YouTube comments"""
    try:
        # Check if we need to update the video
        video_id = extract_video_id(data.url)
        if video_id != current_video_id:
            # Set the new URL first
            await set_url(SetUrlRequest(yturl=data.url))
            
        # Get answer
        response = get_answer(data.query)
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)