import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod";
import type { EventCategory, ThreatLevel, GeoLocation } from "@/types";
import { geocodeLocation, extractLocationsFromText } from "./geocoding";
import {
  classifyCategory as keywordClassifyCategory,
  classifyThreatLevel as keywordClassifyThreatLevel,
} from "./event-classifier";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-nano";

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

// Zod schema for structured event classification
const EventClassificationSchema = z.object({
  category: z.enum([
    "conflict",
    "protest",
    "disaster",
    "diplomatic",
    "economic",
    "terrorism",
    "cyber",
    "health",
    "environmental",
    "military",
  ]).describe("The primary category of the event"),
  threatLevel: z.enum(["critical", "high", "medium", "low", "info"]).describe(
    "Severity level: critical (imminent danger, mass casualties), high (significant threat), medium (developing situation), low (minor/contained), info (routine update)"
  ),
  primaryLocation: z.string().describe(
    "The main geographic location (city, region, or country) where the event is occurring. Use proper names."
  ),
  country: z.string().optional().describe(
    "The country where the event is occurring, if identifiable"
  ),
});

type EventClassification = z.infer<typeof EventClassificationSchema>;

export interface ClassificationResult {
  category: EventCategory;
  threatLevel: ThreatLevel;
  location: GeoLocation | null;
}

/**
 * Classify an event using OpenAI structured outputs
 * Extracts category, threat level, and location in a single API call
 */
async function classifyWithAI(
  title: string,
  content: string
): Promise<EventClassification | null> {
  if (!openai) return null;

  try {
    const completion = await openai.chat.completions.parse({
      model: OPENAI_MODEL,
      messages: [
        {
          role: "system",
          content: `You are an intelligence analyst classifying global events. Analyze the headline and content to determine:
1. Category - the type of event (conflict, protest, disaster, etc.)
2. Threat Level - severity based on potential impact and urgency
3. Location - the primary geographic location where this is happening

Be precise with locations - use actual place names (cities, countries, regions).
For threat level:
- critical: imminent danger, mass casualties, nuclear/WMD threats
- high: significant active threats, major incidents, escalating situations
- medium: developing situations, moderate concern, ongoing tensions
- low: minor incidents, contained events, localized issues
- info: routine updates, announcements, analysis pieces`,
        },
        {
          role: "user",
          content: `Headline: ${title}\n\nContent: ${content.slice(0, 1000)}`,
        },
      ],
      response_format: zodResponseFormat(EventClassificationSchema, "event_classification"),
      max_tokens: 200,
      temperature: 0,
    });

    const message = completion.choices[0]?.message;
    if (message?.parsed) {
      return message.parsed;
    }

    return null;
  } catch (error) {
    console.error("AI classification error:", error);
    return null;
  }
}

/**
 * Classify an event - uses AI if available, falls back to keyword matching
 * Returns category, threat level, and geocoded location
 */
export async function classifyEvent(
  title: string,
  content: string
): Promise<ClassificationResult> {
  const fullText = `${title} ${content}`;

  // Try AI classification first
  const aiResult = await classifyWithAI(title, content);

  if (aiResult) {
    // AI classification succeeded - geocode the location
    let location: GeoLocation | null = null;

    if (aiResult.primaryLocation) {
      location = await geocodeLocation(aiResult.primaryLocation);

      // If AI's location couldn't be geocoded, try with country
      if (!location && aiResult.country) {
        location = await geocodeLocation(aiResult.country);
      }
    }

    return {
      category: aiResult.category as EventCategory,
      threatLevel: aiResult.threatLevel as ThreatLevel,
      location,
    };
  }

  // Fall back to keyword-based classification
  const category = keywordClassifyCategory(fullText);
  const threatLevel = keywordClassifyThreatLevel(fullText);

  // Fall back to regex-based location extraction
  const locationCandidates = extractLocationsFromText(fullText);
  let location: GeoLocation | null = null;

  for (const candidate of locationCandidates) {
    location = await geocodeLocation(candidate);
    if (location) break;
  }

  return {
    category,
    threatLevel,
    location,
  };
}

/**
 * Check if AI classification is available
 */
export function isAIClassificationEnabled(): boolean {
  return !!openai;
}
