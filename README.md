# YouTube Comment Analyzer

This application analyzes YouTube comments using AI to provide summaries and retrieve the most relevant comments based on user queries.

## Features

1. **YouTube Comment Fetching**: Retrieves all comments from a given YouTube video link
2. **Comment Summarization**: Generates an AI-powered summary of the comments
3. **Top-K Query Search**: Finds the most relevant comments based on a user query
4. **AI Chatbot**: Allows users to ask questions about the comments

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- YouTube API key (for fetching comments)
- Google API key (for Gemini AI model)

## Setup

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key
YOUTUBE_API_KEY=your_youtube_api_key
```

### Installation

1. Clone the repository
2. Run the start script:

```
./start.bat
```

This will:
- Create a Python virtual environment if it doesn't exist
- Install all required Python dependencies
- Start the backend server on port 8000
- Start the frontend server on port 3000

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Enter a YouTube URL in the input field on the home page
3. Use the different features:
   - **YouTube Summarizer**: Get an AI-generated summary of the comments
   - **Top Comments**: Search for the most relevant comments based on a query
   - **AI Chatbot**: Ask questions about the comments

## Architecture

- **Frontend**: Next.js application with React components
- **Backend**: FastAPI server with LangChain for AI processing
- **Vector Database**: FAISS for efficient similarity search

## Troubleshooting

- If you encounter issues with the YouTube API, check your API key and quota limits
- If the AI features aren't working, verify your Google API key
- For any other issues, check the console logs in both the frontend and backend terminals