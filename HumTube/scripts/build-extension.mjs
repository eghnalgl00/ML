#!/usr/bin/env node
// Writes extension/catalog.json from data/catalog.json (run after build:catalog).
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PEOPLE } from '../src/core/people.js';
import { ALL_TOPICS } from '../src/core/topics.js';
import { youtubeOrder } from '../src/app/views.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const catalog = JSON.parse(await readFile(resolve(ROOT, 'data/catalog.json'), 'utf8'));
const keep = ['id', 'title', 'channelId', 'channelTitle', 'views', 'publishedAt', 'durationSec', 'personIds', 'ranks', 'topics'];
const videos = catalog.videos.map((v) => Object.fromEntries(keep.filter((k) => v[k] !== undefined).map((k) => [k, v[k]])));
const out = {
  generatedAt: catalog.generatedAt,
  people: Object.fromEntries(PEOPLE.map((p) => [p.id, { name: p.name, category: p.category }])),
  topics: Object.fromEntries(ALL_TOPICS.map((t) => [t.id, t.label])),
  videos,
  order: youtubeOrder(videos).map((v) => v.id),
};
await writeFile(resolve(ROOT, 'extension/catalog.json'), JSON.stringify(out));
console.log(`extension/catalog.json: ${videos.length} videos`);
