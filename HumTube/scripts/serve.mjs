#!/usr/bin/env node
// Tiny dependency-free static server for HumTube (ES modules need http://, not file://).
//   node scripts/serve.mjs            → http://localhost:5173
//   PORT=8080 node scripts/serve.mjs

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC = new Set(['index.html', 'styles.css', 'favicon.svg', 'src', 'data']);
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

export function createHumTubeServer() {
  return createServer(async (req, res) => {
    try {
      const { pathname } = new URL(req.url, 'http://localhost');
      let rel = decodeURIComponent(pathname).replace(/^\/+/, '') || 'index.html';
      const file = normalize(join(ROOT, rel));
      const top = rel.split('/')[0];
      if (!file.startsWith(ROOT + sep) || !PUBLIC.has(top) || rel.split('/').some((s) => s.startsWith('.'))) {
        res.writeHead(404).end('Not found');
        return;
      }
      const info = await stat(file).catch(() => null);
      if (!info?.isFile()) {
        res.writeHead(404).end('Not found');
        return;
      }
      res.writeHead(200, {
        'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
        'Cache-Control': 'no-cache',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(await readFile(file));
    } catch {
      res.writeHead(500).end('Server error');
    }
  });
}

const invokedDirectly = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  const port = Number(process.env.PORT) || 5173;
  createHumTubeServer().listen(port, () => console.log(`HumTube running at http://localhost:${port}`));
}
