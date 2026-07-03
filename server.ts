import 'dotenv/config';

import cors from '@fastify/cors';
import Fastify from 'fastify';
import { scanRoutes } from './routes/scan.js';

const server = Fastify({ logger: true });

await server.register(cors, {
  origin: true,
});

await server.register(scanRoutes);

server.get('/health', async () => ({ status: 'ok' }));

try {
  const port = Number(process.env.PORT ?? 3000);
  await server.listen({ port, host: '0.0.0.0' });
} catch (err) {
  server.log.error(err);
  process.exit(1);
}
