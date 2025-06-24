import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"

export async function POST(req: Request) {
  try {
    const { query, count, youtubeUrl } = await req.json()

    // Generate simulated comments based on the YouTube URL and query
    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: `Generate ${count} realistic YouTube comments for a video at ${youtubeUrl}${
        query ? ` that mention or relate to "${query}"` : ""
      }. 
      
      Format the response as a JSON array of comment objects with the following structure:
      [
        {
          "id": "unique-id",
          "author": "username",
          "text": "comment text",
          "likes": number of likes,
          "timestamp": "time ago (e.g., '2 days ago')"
        }
      ]
      
      Make the comments realistic, with varying lengths, styles, and engagement levels. Include some emoji in some comments.`,
    })

    // Parse the JSON response
    let comments
    try {
      // Extract JSON from the response if it's wrapped in markdown code blocks
      const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/) || text.match(/```\n([\s\S]*?)\n```/)
      const jsonString = jsonMatch ? jsonMatch[1] : text
      comments = JSON.parse(jsonString)
    } catch (parseError) {
      console.error("Error parsing comments JSON:", parseError)
      return Response.json({ error: "Failed to parse comments" }, { status: 500 })
    }

    return Response.json({ comments })
  } catch (error) {
    console.error("Error generating comments:", error)
    return Response.json({ error: "Failed to generate comments" }, { status: 500 })
  }
}
