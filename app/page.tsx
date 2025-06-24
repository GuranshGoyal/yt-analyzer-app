"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { BASE_URL } from "@/data"
import axios from "axios"
import { MessageSquareIcon, SearchIcon, YoutubeIcon } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import { useYouTube } from "./context/youtube-context"

export default function Home() {



  


  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12">
      <div className="container mx-auto px-4">
        <h1 className="text-4xl font-bold text-center mb-2">AI-Powered Web Application</h1>
        <p className="text-gray-600 text-center mb-6">Analyze YouTube videos, search content, and chat with AI</p>

        {/* YouTube URL Input */}
        <YouTubeInputSection />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <YoutubeIcon className="w-12 h-12 mx-auto text-red-600 mb-2" />
              <CardTitle>YouTube Summarizer</CardTitle>
              <CardDescription>Get AI-generated summaries of any YouTube video</CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p>Get an instant summary of the YouTube video content.</p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Link href="/youtube-summary">
                <Button>Try Summarizer</Button>
              </Link>
            </CardFooter>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <SearchIcon className="w-12 h-12 mx-auto text-purple-600 mb-2" />
              <CardTitle>Top Comments</CardTitle>
              <CardDescription>View top comments from the YouTube video</CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p>Enter a search term and number to see matching top comments.</p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Link href="/query-search">
                <Button>View Comments</Button>
              </Link>
            </CardFooter>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <MessageSquareIcon className="w-12 h-12 mx-auto text-green-600 mb-2" />
              <CardTitle>AI Chatbot</CardTitle>
              <CardDescription>Have a conversation with our AI assistant</CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p>Chat with our AI bot about the YouTube video content.</p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Link href="/chatbot">
                <Button>Start Chatting</Button>
              </Link>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  )
}

// YouTube URL Input Component - Only on home page
function YouTubeInputSection() {
  const { youtubeUrl, setYoutubeUrl, isLoading } = useYouTube()
  const [inputUrl, setInputUrl] = useState(youtubeUrl)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!inputUrl) {
      alert("Please enter a YouTube URL")
      return
    }
    
    try {
      setIsLoading(true)
      setYoutubeUrl(inputUrl)
      
      const response = await axios.post(`${BASE_URL}/seturl`, { yturl: inputUrl })
      console.log("URL set successfully:", response.data)
      
      // Keep the URL in the input field for reference
      // setInputUrl("") 
    } catch (error) {
      console.error("Error setting URL:", error)
      alert("Error setting URL. Please check if the backend server is running.")
      // Revert the URL if there was an error
      setYoutubeUrl("")
    } finally {
      setIsLoading(false)
    }
  }

  

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <YoutubeIcon className="h-6 w-6 text-red-600 mr-2" />
          Enter YouTube URL
        </h2>
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <Input
            type="text"
            placeholder="https://www.youtube.com/watch?v=..."
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            className="flex-grow"
          />
          <Button type="submit" disabled={isLoading}>
            {isLoading ? "Loading..." : "Set URL"}
          </Button>
        </form>
        {youtubeUrl && (
          <p className="mt-2 text-sm text-gray-600">
            Current URL: <span className="font-medium">{youtubeUrl}</span>
          </p>
        )}
      </div>
    </div>
  )
}
