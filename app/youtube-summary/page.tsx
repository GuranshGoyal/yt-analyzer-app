"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import axios from "axios"
import { ArrowLeft, YoutubeIcon } from "lucide-react"
import Link from "next/link"
import { useEffect, useState } from "react"
import { BASE_URL } from "../../data.js"
import LoadingSpinner from "../components/loading-spinner"
import { useYouTube } from "../context/youtube-context"

export default function YoutubeSummary() {
  const { youtubeUrl, setIsLoading, isLoading } = useYouTube()
  const [summary, setSummary] = useState("")
  const [error, setError] = useState("")
  console.log("BASE_URL", BASE_URL )


  const generateSummary = async () => {
    if (!youtubeUrl) {
      setError("Please enter a YouTube URL on the home page")
      return
    }

    try {
      setIsLoading(true)
      setError("")
      setSummary("")

      const response = await axios.post(`${BASE_URL}/yturl`, {
        url: youtubeUrl,
      });
      
      console.log("Summary response:", response.data);
      
      if (!response.data || !response.data.summary) {
        throw new Error("Failed to generate summary")
      }
      
      setSummary(response.data.summary);
    } catch (err) {
      setError("Error generating summary. Please check the URL and try again.")
      console.error("Summary error:", err)
    } finally {
      setIsLoading(false)
    }
  }
  console.log("youtubeUrl", youtubeUrl)
  // Generate summary when the page loads if URL is available
  useEffect(() => {
    // if (youtubeUrl) {
      generateSummary()
    // }
  }, [])

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900 mb-6">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Home
      </Link>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            <YoutubeIcon className="h-10 w-10 text-red-600 mr-2" />
            <CardTitle className="text-2xl">YouTube Video Summarizer</CardTitle>
          </div>
          <CardDescription className="text-center">
            Get an AI-generated summary of the YouTube video:{" "}
            {youtubeUrl ? <span className="font-medium">{youtubeUrl}</span> : "No URL set"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!youtubeUrl ? (
            <div className="text-center py-4">
              <p className="text-amber-600">Please enter a YouTube URL on the home page to generate a summary</p>
              <Link href="/" className="mt-4 inline-block">
                <Button>Go to Home Page</Button>
              </Link>
            </div>
          ) : (
            <>
              <Button onClick={generateSummary} className="w-full mb-6" disabled={isLoading || !youtubeUrl}>
                {isLoading ? "Generating..." : "Regenerate Summary"}
              </Button>

              {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

              {isLoading ? (
                <LoadingSpinner />
              ) : summary ? (
                <div>
                  <h3 className="font-medium text-lg mb-2">Summary:</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    <p className="whitespace-pre-line">{summary}</p>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
