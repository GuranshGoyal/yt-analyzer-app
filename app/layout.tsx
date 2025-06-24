import type { ReactNode } from "react";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { YouTubeProvider } from "./context/youtube-context";

// ✅ Properly export metadata
export const metadata = {
  title: "Your App Title",
  description: "Your app description",
  generator: "v0.dev",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="light">
          <YouTubeProvider>{children}</YouTubeProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
