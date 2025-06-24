import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"

export async function POST(req: Request) {
  try {
    const { query, number, youtubeUrl } = await req.json()

    // Generate results based on the query, number, and YouTube URL context
    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: `Based on the YouTube video at ${youtubeUrl}, generate ${number} distinct results related to the query: "${query}". 
      Format each result as a separate item in a list. If the query is not directly related to the video, still try to make the results relevant to both the query and the video context.`,
    })

    // Parse the results into an array
    const results = text
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => line.replace(/^\d+\.\s*/, "").trim())
      .slice(0, number)

    return Response.json({ results })
  } catch (error) {
    console.error("Error generating search results:", error)
    return Response.json({ error: "Failed to generate search results" }, { status: 500 })
  }
}
