"use client"

import { createContext, useState, useContext, useEffect, type ReactNode } from "react"

type YouTubeContextType = {
  youtubeUrl: string
  setYoutubeUrl: (url: string) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

const YouTubeContext = createContext<YouTubeContextType | undefined>(undefined)

export function YouTubeProvider({ children }: { children: ReactNode }) {
  // Initialize state from localStorage if available (client-side only)
  const [youtubeUrl, setYoutubeUrlState] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  // Load saved URL from localStorage on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem("youtubeUrl")
    if (savedUrl) {
      setYoutubeUrlState(savedUrl)
    }
  }, [])

  // Save URL to localStorage when it changes
  const setYoutubeUrl = (url: string) => {
    setYoutubeUrlState(url)
    localStorage.setItem("youtubeUrl", url)
  }

  return (
    <YouTubeContext.Provider value={{ youtubeUrl, setYoutubeUrl, isLoading, setIsLoading }}>
      {children}
    </YouTubeContext.Provider>
  )
}

export function useYouTube() {
  const context = useContext(YouTubeContext)
  if (context === undefined) {
    throw new Error("useYouTube must be used within a YouTubeProvider")
  }
  return context
}
