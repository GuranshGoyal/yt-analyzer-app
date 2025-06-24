"use client"

import { useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ArrowLeft, MessageSquareIcon, SendIcon } from "lucide-react"
import Link from "next/link"
import { useChat } from "ai/react"
import { useYouTube } from "../context/youtube-context"
import axios from "axios"
import { BASE_URL } from "@/data"

export default function Chatbot() {
  const { youtubeUrl } = useYouTube()

  const {
    messages,
    input,
    handleInputChange,
    isLoading,
    append,
    setInput,
  } = useChat({
    initialMessages: youtubeUrl
      ? [
          {
            id: "system-1",
            role: "system",
            content: `The user is analyzing this YouTube video: ${youtubeUrl}. Provide helpful responses related to this video if they ask about it.`,
          },
        ]
      : [],
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const handlechat = async () => {
    if (!input.trim()) return

    console.log("input", input)

    const userMessage = input
    setInput("") // Clear the input field immediately

    // Add user message to UI
    await append({
      role: "user",
      content: userMessage,
    })

    try {
      const response = await axios.post(`${BASE_URL}/chat`, {
        url: youtubeUrl,
        query: userMessage,
      })
      
      if (!response.data) {
        throw new Error("Failed to get response")
      }

      console.log("response", response)
      const botReply = response.data?.message || "No response received."

      await append({
        role: "assistant",
        content: botReply,
      })
    } catch (error) {
      console.error("Error sending message:", error)
      await append({
        role: "assistant",
        content: "Sorry, there was an error processing your request.",
      })
    }
  }

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <Link
        href="/"
        className="flex items-center text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Home
      </Link>

      <Card className="h-[70vh] flex flex-col">
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            <MessageSquareIcon className="h-10 w-10 text-green-600 mr-2" />
            <CardTitle className="text-2xl">AI Chatbot</CardTitle>
          </div>
          <CardDescription className="text-center">
            Chat with our AI assistant about the YouTube video:{" "}
            {youtubeUrl ? (
              <span className="font-medium">{youtubeUrl}</span>
            ) : (
              "No URL set"
            )}
          </CardDescription>
        </CardHeader>

        <CardContent className="flex-grow overflow-y-auto mb-4 space-y-4">
          {!youtubeUrl ? (
            <div className="h-full flex items-center justify-center flex-col gap-4 text-center">
              <p className="text-amber-600">
                Please enter a YouTube URL on the home page first
              </p>
              <Link href="/">
                <Button>Go to Home Page</Button>
              </Link>
            </div>
          ) : messages.length <= (youtubeUrl ? 1 : 0) ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              <p>Start a conversation by sending a message below.</p>
            </div>
          ) : (
            messages.map(
              (message) =>
                message.role !== "system" && (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === "user"
                        ? "justify-end"
                        : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        message.role === "user"
                          ? "bg-green-600 text-white"
                          : "bg-gray-200 text-gray-800"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                )
            )
          )}
          <div ref={messagesEndRef} />
        </CardContent>

        <CardFooter className="border-t pt-4">
          {youtubeUrl && (
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handlechat()
              }}
              className="w-full flex space-x-2"
            >
              <Input
                value={input}
                onChange={handleInputChange}
                placeholder="Type your message..."
                className="flex-grow"
                disabled={isLoading}
              />
              <Button type="submit" disabled={isLoading}>
                {isLoading ? (
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                ) : (
                  <SendIcon className="h-4 w-4" />
                )}
                <span className="sr-only">Send</span>
              </Button>
            </form>
          )}
        </CardFooter>
      </Card>
    </div>
  )
}
