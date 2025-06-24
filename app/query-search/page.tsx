"use client"

import { Button } from "@/components/ui/button"
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { BASE_URL } from "@/data"
import axios from "axios"
import { ArrowLeft, MessageCircle } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import LoadingSpinner from "../components/loading-spinner"
import { useYouTube } from "../context/youtube-context"

// Updated Comment type
type Comment = {
  id: string
  author: string
  comment_text: string
  likes: number
  similarity_score: number
}

export default function QuerySearch() {
  const { youtubeUrl, isLoading, setIsLoading } = useYouTube()
  const [query, setQuery] = useState("")
  const [number, setNumber] = useState("5")
  const [comments, setComments] = useState<Comment[]>([])
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!youtubeUrl) {
      setError("Please enter a YouTube URL on the home page first")
      return
    }

    if (!number || isNaN(Number.parseInt(number))) {
      setError("Please enter a valid number")
      return
    }

    try {
      setIsLoading(true)
      setError("")

      const response = await axios.post(`${BASE_URL}/comment`, {
        yturl: youtubeUrl,
        query: query,
        count: parseInt(number),
      })

      if (!response.data || !response.data.comment) {
        throw new Error("Failed to get comments")
      }

      console.log("Comment response:", response.data.comment)
      
      try {
        const parsedComments = JSON.parse(response.data.comment)
        setComments(parsedComments || [])
      } catch (parseError) {
        console.error("Error parsing comments:", parseError)
        setError("Error parsing comments response")
        setComments([])
      }
    } catch (err) {
      setError("Error getting comments. Please try again.")
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <Link
        href="/"
        className="flex items-center text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Home
      </Link>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            <MessageCircle className="h-10 w-10 text-purple-600 mr-2" />
            <CardTitle className="text-2xl">Top Comments</CardTitle>
          </div>
          <CardDescription className="text-center">
            View top comments from the YouTube video:{" "}
            {youtubeUrl ? (
              <span className="font-medium">{youtubeUrl}</span>
            ) : (
              "No URL set"
            )}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {!youtubeUrl ? (
            <div className="text-center py-4">
              <p className="text-amber-600">
                Please enter a YouTube URL on the home page first
              </p>
              <Link href="/" className="mt-4 inline-block">
                <Button>Go to Home Page</Button>
              </Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex flex-col space-y-2">
                <label htmlFor="query" className="text-sm font-medium">
                  Search Term (optional)
                </label>
                <Input
                  id="query"
                  type="text"
                  placeholder="Filter comments by keyword"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>

              <div className="flex flex-col space-y-2">
                <label htmlFor="number" className="text-sm font-medium">
                  Number of Comments
                </label>
                <Input
                  id="number"
                  type="number"
                  placeholder="Enter a number"
                  value={number}
                  onChange={(e) => setNumber(e.target.value)}
                  min="1"
                  max="20"
                />
              </div>

              {error && <p className="text-red-500 text-sm">{error}</p>}

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? "Loading Comments..." : "Get Top Comments"}
              </Button>
            </form>
          )}

          {isLoading ? (
            <LoadingSpinner />
          ) : (
            <div className="mt-6">
              <h3 className="font-medium text-lg mb-2">Top Comments:</h3>
              {comments.length > 0 ? (
                <div className="space-y-4">
                  {comments.map((comment, index) => (
                    <div
                      key={index}
                      className="bg-gray-50 border p-4 rounded-md shadow-sm"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <p className="font-semibold text-purple-800">
                          {comment.author}
                        </p>
                        <div className="text-sm text-gray-500 text-right">
                          <span className="block">👍 {comment.likes}</span>
                          <span className="block">
                            🔍 Score:{" "}
                            {comment.similarity_score.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      <p className="text-gray-800">{comment.comment_text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-center text-gray-500">
                  No comments found for this video.
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
