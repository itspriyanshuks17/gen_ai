// Load environment variables from .env into process.env at startup.
import "dotenv/config";
// Built-in Node module to create an HTTP server for local development.
import http from "node:http";
// Promise-based filesystem API for reading static files.
import fs from "node:fs/promises";
// Utilities for safe cross-platform file/folder path operations.
import path from "node:path";
// Converts import.meta.url into a normal file path.
import { fileURLToPath } from "node:url";
// Official Gemini SDK client.
import { GoogleGenAI } from "@google/genai";

// -----------------------------------------------------------------------------
// FILE OVERVIEW
// -----------------------------------------------------------------------------
// This single file supports two execution modes:
// 1) Local mode (`npm start`): runs an HTTP server and serves the frontend.
// 2) Lambda mode (`index.handler`): handles API Gateway events in AWS.

// Absolute path of this file at runtime (ESM does not provide __filename by default).
const __filename = fileURLToPath(import.meta.url);
// Directory containing this file.
const __dirname = path.dirname(__filename);
// Folder that stores frontend static assets (index.html).
const publicDir = path.join(__dirname, "public");

// Read Gemini API key from environment.
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
// Stop startup early with a clear message if the key is missing.
if (!GEMINI_API_KEY) {
  throw new Error("Missing GEMINI_API_KEY. Add it to ai-agents/weather/.env or Lambda environment variables.");
}

// Create a reusable Gemini client instance for all requests.
const client = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
// Preferred local server port (uses env PORT if set, otherwise 3000).
const PORT = Number(process.env.PORT || 3000);
// Bind host explicitly to avoid IPv4/IPv6 ambiguity on some systems.
const HOST = process.env.HOST || "0.0.0.0";

// Common CORS headers used by both local server and Lambda responses.
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// Local mock weather tool used as deterministic app-side data.
// This is separate from Gemini so you can compare AI text vs local tool output.
function getWeatherDetails(city = "") {
  // Normalize input to reduce case/space issues.
  const normalized = city.trim().toLowerCase();

  // Mock city responses.
  if (normalized === "pune") {
    return "The weather in Pune is currently sunny with a temperature of 25 C.";
  }
  if (normalized === "mumbai") {
    return "The weather in Mumbai is currently rainy with a temperature of 15 C.";
  }
  if (normalized === "ahmedabad") {
    return "The weather in Ahmedabad is currently cloudy with a temperature of 20 C.";
  }

  return `Sorry, I don't have weather information for ${city}.`;
}

// Calls Gemini model to produce a short natural-language weather reply.
async function getGeminiWeather(city) {
  // Prompt keeps output brief and focused.
  const prompt = [
    "You are a helpful weather assistant.",
    `User asks: What is the weather in ${city}?`,
    "Give a short 2-3 line response in plain English.",
  ].join(" ");

  const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
  });

  // Fallback string if SDK returns empty text.
  return response.text ?? "No response from Gemini.";
}

// Build a JSON HTTP response object usable by both Lambda and local adapter.
function jsonResponse(statusCode, data, extraHeaders = {}) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
      ...extraHeaders,
    },
    body: JSON.stringify(data),
  };
}

// Build an HTML HTTP response object usable by both Lambda and local adapter.
function htmlResponse(statusCode, html) {
  return {
    statusCode,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      ...CORS_HEADERS,
    },
    body: html,
  };
}

// Reads request body stream and parses JSON safely (local Node server only).
async function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";

    // Collect request chunks as they arrive.
    req.on("data", (chunk) => {
      body += chunk;
      // Basic payload size guard (about 1 MB).
      if (body.length > 1e6) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });

    // Parse JSON after full body arrives.
    req.on("end", () => {
      try {
        resolve(JSON.parse(body || "{}"));
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });

    // Propagate stream errors.
    req.on("error", reject);
  });
}

// Core router/handler shared by:
// 1) local Node server
// 2) AWS Lambda handler
async function handleRequest(method, pathname, parsedBody = {}) {
  // CORS preflight support.
  if (method === "OPTIONS") {
    return {
      statusCode: 204,
      headers: CORS_HEADERS,
      body: "",
    };
  }

  // Route: GET / or /index.html -> return frontend page.
  if (method === "GET" && (pathname === "/" || pathname === "/index.html")) {
    const html = await fs.readFile(path.join(publicDir, "index.html"), "utf8");
    return htmlResponse(200, html);
  }

  // Route: POST /api/weather -> accept city and return Gemini + local weather.
  if (method === "POST" && pathname === "/api/weather") {
    const { city } = parsedBody;

    // Validate required input.
    if (!city || typeof city !== "string") {
      return jsonResponse(400, { error: "Please provide a city name." });
    }

    // Run both calls together for lower response time.
    const [geminiReply, localWeather] = await Promise.all([
      getGeminiWeather(city),
      Promise.resolve(getWeatherDetails(city)),
    ]);

    // Success response payload consumed by frontend.
    return jsonResponse(200, {
      city,
      geminiReply,
      localWeather,
    });
  }

  // Unknown route fallback.
  return jsonResponse(404, { error: "Route not found" });
}

// AWS Lambda entrypoint.
// Deploy with handler value: index.handler
export async function handler(event) {
  try {
    // Support API Gateway HTTP API payload format.
    const method = event?.requestContext?.http?.method || event?.httpMethod || "GET";
    const pathname = event?.rawPath || event?.path || "/";

    let parsedBody = {};

    // Parse JSON body only for methods that typically carry payloads.
    // API Gateway sends body as string, possibly base64 encoded.
    if (method === "POST" || method === "PUT" || method === "PATCH") {
      const rawBody = event?.isBase64Encoded
        ? Buffer.from(event.body || "", "base64").toString("utf8")
        : (event.body || "");

      if (rawBody) {
        try {
          parsedBody = JSON.parse(rawBody);
        } catch {
          return jsonResponse(400, { error: "Invalid JSON body" });
        }
      }
    }

    return await handleRequest(method, pathname, parsedBody);
  } catch (error) {
    if (error?.message?.includes("RESOURCE_EXHAUSTED") || error?.status === 429) {
      return jsonResponse(429, {
        error: "Gemini rate limit exceeded. Please wait and retry.",
      });
    }

    return jsonResponse(500, {
      error: "Internal server error",
      details: error?.message || String(error),
    });
  }
}

// Start local server only when not running inside AWS Lambda runtime.
if (!process.env.AWS_LAMBDA_FUNCTION_NAME) {
  // This adapter converts Node HTTP requests into the same method/path/body
  // shape used by `handleRequest(...)`, so business logic stays centralized.
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
      const pathname = url.pathname;
      const method = req.method || "GET";

      let parsedBody = {};
      if (method === "POST" || method === "PUT" || method === "PATCH") {
        parsedBody = await readJsonBody(req);
      }

      const response = await handleRequest(method, pathname, parsedBody);
      res.writeHead(response.statusCode, response.headers);
      res.end(response.body);
    } catch (error) {
      const fallback = jsonResponse(500, {
        error: "Internal server error",
        details: error?.message || String(error),
      });
      res.writeHead(fallback.statusCode, fallback.headers);
      res.end(fallback.body);
    }
  });

  // If preferred port is occupied, retry once with a random free port.
  let retriedWithRandomPort = false;

  server.on("error", (error) => {
    if (error?.code === "EADDRINUSE" && !retriedWithRandomPort) {
      retriedWithRandomPort = true;
      console.warn(`Port ${PORT} is busy. Retrying with an available port...`);
      server.listen(0, HOST);
      return;
    }

    console.error("Server failed to start:", error?.message || error);
    process.exit(1);
  });

  // Start listening for browser/API requests.
  server.on("listening", () => {
    const address = server.address();
    const activePort = typeof address === "object" && address ? address.port : PORT;
    console.log(`Weather web app is running at http://localhost:${activePort}`);
    console.log(`Also available at http://127.0.0.1:${activePort}`);
  });

  server.listen(PORT, HOST);
}
