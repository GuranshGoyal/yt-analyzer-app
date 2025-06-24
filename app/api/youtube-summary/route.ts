import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"

export async function POST(req: Request) {
  try {
    const { url } = await req.json()

    // In a real application, you would extract video content here
    // For this example, we'll simulate it by generating a summary based on the URL

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: `Generate a comprehensive summary of the YouTube video at this URL: ${url}. 
      Since I can't actually watch the video, please create a plausible summary that might 
      represent what this video could be about based on the URL. Include key points that 
      might be covered in such a video.`,
    })

    return Response.json({ summary: text })
  } catch (error) {
    console.error("Error generating YouTube summary:", error)
    return Response.json({ error: "Failed to generate summary" }, { status: 500 })
  }
}
