import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { openai } from '../lib/openai.js';
import { supabase } from '../lib/supabase.js';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ScanBody {
  imageBase64: string;
  userId: string;
}

interface BikeSpecs {
  frame?: string;
  fork?: string;
  weight_kg?: number | null;
}

interface BikeComponents {
  drivetrain?: string;
  brakes?: string;
  wheels?: string;
  saddle?: string;
  handlebar?: string;
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

const SYSTEM_PROMPT = `Analysiere dieses Bild. Erkennst du ein Fahrrad?
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Markdown, kein Text davor oder danach):
{
  "brand": "Markenname oder 'Unbekannt'",
  "model": "Modellname oder 'Unbekannt'",
  "model_year": null,
  "specs": { "frame": "", "fork": "", "weight_kg": null },
  "components": { "drivetrain": "", "brakes": "", "wheels": "", "saddle": "", "handlebar": "" },
  "market_value_chf": 0,
  "confidence": 0.0
}
Falls kein Fahrrad sichtbar: { "error": "no_bike_detected", "confidence": 0 }`;

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
    max_tokens: 600,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: SYSTEM_PROMPT },
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
