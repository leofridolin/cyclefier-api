import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { openai } from '../lib/openai.js';
import { supabase } from '../lib/supabase.js';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ScanBody {
  imageBase64: string;
  userId: string;
}

interface BikeSpecs {
  weight_kg?: number | null;
  size?: string | null;
  color?: string | null;
}

interface BikeComponents {
  frame?: string | null;
  fork?: string | null;
  wheels?: string | null;
  tires?: string | null;
  crankset?: string | null;
  cassette?: string | null;
  front_derailleur?: string | null;
  rear_derailleur?: string | null;
  chain?: string | null;
  brakes?: string | null;
  bar_tape?: string | null;
  saddle?: string | null;
  seatpost?: string | null;
  pedals?: string | null;
}

interface GPTBikeResult {
  brand: string;
  model: string;
  model_year: number | null;
  specs: BikeSpecs;
  components: BikeComponents;
  market_value_chf: number;
  confidence: number;
  error?: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const RATE_LIMIT_WINDOW_MINUTES = 60;
const RATE_LIMIT_MAX_SCANS = 10;
const MIN_CONFIDENCE = 0.4;
const MODEL = 'gpt-4o' as const;

const SYSTEM_PROMPT = `Du bist ein Fahrrad-Experte mit tiefem Wissen über Fahrradmarken, Modelle, Komponenten und Schweizer Marktpreise. Analysiere das Bild so präzise wie möglich.

Vorgehensweise:
1. Erkenne Markenlogos, Aufkleber und spezifische Designmerkmale
2. Bestimme den genauen Modellnamen inkl. Variantbezeichnung und Farbnamen (z.B. "X-Lite 04 aggro papaya" statt nur "X-Lite")
3. Erkenne Rahmengrösse anhand sichtbarer Beschriftungen oder Proportionen
4. Bestimme die genaue Farbe/Lackierung des Rahmens
5. Identifiziere alle sichtbaren Komponenten anhand von Logos, Beschriftungen und Form
6. Schätze einen realistischen CHF-Wiederverkaufswert (nicht UVP)

Antworte NUR mit diesem JSON-Objekt ohne Markdown oder erklärender Text.
Füge nur Felder ein, bei denen du Informationen gefunden hast. Verwende null für unbekannte Werte:
{
  "brand": "z.B. Rose, Trek, Specialized, Canyon, Giant, Scott, Cannondale, BMC",
  "model": "vollständiger Modellname inkl. Variante und Farbe z.B. 'X-Lite 04 aggro papaya', 'Tarmac SL7 Pro Sagan Collection', 'Emonda SLR 9 eTap'",
  "model_year": 2023,
  "specs": {
    "weight_kg": 8.2,
    "size": "M / 54cm",
    "color": "z.B. Papaya Orange / Schwarz Matt"
  },
  "components": {
    "frame": "z.B. Rose X-Lite Carbon Monocoque, Trek OCLV 700 Series Carbon",
    "fork": "z.B. Rose Carbon Starrgabel integriert, Fox 34 Float Factory 140mm",
    "wheels": "z.B. DT Swiss R470 700c, Shimano RS500, Reynolds 46 Carbon",
    "tires": "z.B. Continental GP5000 700x28c, Schwalbe One 700x25c",
    "crankset": "z.B. Shimano Ultegra R8000 172.5mm 52/36T, SRAM Force AXS 48/35T",
    "cassette": "z.B. Shimano Ultegra R8000 11-28T 11-fach, SRAM XG-1270 10-33T",
    "front_derailleur": "z.B. Shimano Ultegra R8000, SRAM Force AXS, nicht vorhanden (1x)",
    "rear_derailleur": "z.B. Shimano Ultegra R8000, SRAM Force AXS eTap",
    "chain": "z.B. Shimano Ultegra HG701 11-fach, KMC X11",
    "brakes": "z.B. Shimano Ultegra R8070 Hydraulic Disc, SRAM Force Hydraulic Disc, Shimano Dura-Ace Felgenbremse",
    "bar_tape": "z.B. Fizik Tempo Microtex 2mm Schwarz, Bontrager Comp",
    "saddle": "z.B. Fizik Antares R3 Adaptive, Selle Italia SLR Boost Kit Carbonio",
    "seatpost": "z.B. Rose Carbon Sattelstütze 27.2mm, Thomson Elite 27.2mm",
    "pedals": "z.B. Shimano PD-R8000 SPD-SL, Nicht montiert, Crankbrothers Candy 3"
  },
  "market_value_chf": 3800,
  "confidence": 0.88
}
Falls kein Fahrrad erkennbar: { "error": "no_bike_detected", "confidence": 0 }`;

// ── Rate-limit helper ─────────────────────────────────────────────────────────

const checkRateLimit = async (userId: string): Promise<boolean> => {
  const windowStart = new Date(
    Date.now() - RATE_LIMIT_WINDOW_MINUTES * 60 * 1000,
  ).toISOString();

  const { count, error } = await supabase
    .from('scan_events')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', userId)
    .gte('created_at', windowStart);

  if (error) {
    // Fail open — don't block user if Supabase is unavailable
    console.error('Rate-limit check failed:', error.message);
    return false;
  }

  return (count ?? 0) >= RATE_LIMIT_MAX_SCANS;
};

// ── GPT vision call ───────────────────────────────────────────────────────────

const analyseImage = async (imageBase64: string): Promise<GPTBikeResult> => {
  const dataUrl = `data:image/jpeg;base64,${imageBase64}`;

  const response = await openai.chat.completions.create({
    model: MODEL,
    max_tokens: 1000,
    messages: [
      {
        role: 'system',
        content: SYSTEM_PROMPT,
      },
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'Analysiere dieses Fahrrad-Bild genau und gib das JSON zurück.',
          },
          { type: 'image_url', image_url: { url: dataUrl, detail: 'high' } },
        ],
      },
    ],
  });

  const raw = response.choices[0]?.message?.content ?? '';

  // Strip accidental markdown code fences if GPT wraps the JSON anyway
  const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();

  return JSON.parse(cleaned) as GPTBikeResult;
};

// ── Insert scan event ─────────────────────────────────────────────────────────

const insertScanEvent = async (
  userId: string,
  confidence: number,
  durationMs: number,
): Promise<void> => {
  const { error } = await supabase.from('scan_events').insert({
    user_id: userId,
    confidence,
    model_used: MODEL,
    duration_ms: durationMs,
  });

  if (error) {
    console.error('Failed to insert scan_event:', error.message);
  }
};

// ── Route handler ─────────────────────────────────────────────────────────────

const scanHandler = async (
  request: FastifyRequest<{ Body: ScanBody }>,
  reply: FastifyReply,
): Promise<void> => {
  const { imageBase64, userId } = request.body;

  // 1. Rate-limit check
  const isRateLimited = await checkRateLimit(userId);
  if (isRateLimited) {
    return reply.code(429).send({
      error: 'rate_limit_exceeded',
      message: `Maximal ${RATE_LIMIT_MAX_SCANS} Scans pro ${RATE_LIMIT_WINDOW_MINUTES} Minuten.`,
    });
  }

  const startMs = Date.now();

  // 2. GPT-4o Vision
  let result: GPTBikeResult;
  try {
    result = await analyseImage(imageBase64);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'JSON parse failed';
    request.log.error({ err }, 'OpenAI call or JSON parse failed');
    return reply.code(500).send({ error: 'analysis_failed', message });
  }

  const durationMs = Date.now() - startMs;

  // 3. No bike detected
  if (result.error === 'no_bike_detected') {
    return reply.code(422).send({ error: 'no_bike' });
  }

  // 4. Low confidence
  if (result.confidence < MIN_CONFIDENCE) {
    return reply.code(422).send({
      error: 'low_confidence',
      confidence: result.confidence,
    });
  }

  // 5. Persist scan event (fire-and-forget, non-blocking)
  void insertScanEvent(userId, result.confidence, durationMs);

  // 6. Return ScanResult
  return reply.code(200).send({
    brand: result.brand,
    model: result.model,
    model_year: result.model_year,
    specs: result.specs,
    components: result.components,
    market_value_chf: result.market_value_chf,
    confidence: result.confidence,
  });
};

// ── Fastify plugin ────────────────────────────────────────────────────────────

export const scanRoutes = async (fastify: FastifyInstance): Promise<void> => {
  fastify.post<{ Body: ScanBody }>(
    '/api/scan',
    {
      schema: {
        body: {
          type: 'object',
          required: ['imageBase64', 'userId'],
          properties: {
            imageBase64: { type: 'string', minLength: 1 },
            userId: { type: 'string', minLength: 1 },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              brand: { type: 'string' },
              model: { type: 'string' },
              model_year: { type: ['number', 'null'] },
              specs: { type: 'object' },
              components: { type: 'object' },
              market_value_chf: { type: 'number' },
              confidence: { type: 'number' },
            },
          },
          422: {
            type: 'object',
            properties: {
              error: { type: 'string' },
              confidence: { type: 'number' },
            },
          },
          429: {
            type: 'object',
            properties: {
              error: { type: 'string' },
              message: { type: 'string' },
            },
          },
        },
      },
    },
    scanHandler,
  );
};
