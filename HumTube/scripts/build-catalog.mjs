#!/usr/bin/env node
// Build data/catalog.json from the YouTube Data API v3.
//
//   YOUTUBE_API_KEY=... node scripts/build-catalog.mjs [options]
//
// Every API response is cached under .cache/, so the run is resumable: when the
// daily quota runs out, run the same command tomorrow and it picks up where it
// stopped. `--offline` rebuilds the catalog from the cache without any API calls
// (handy after changing thresholds or matching rules).

import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { PEOPLE, PEOPLE_BY_ID, CATEGORIES, queryFor } from '../src/core/people.js';
import { DEFAULT_THRESHOLDS, REJECTION_LABELS } from '../src/core/filters.js';
import { createClient, UNIT_COST, YouTubeError } from '../src/core/youtube.js';
import { buildCatalog, channelIdsToCheck, mergeCatalogs, SEARCH_ORDER } from '../src/core/catalog.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DAY = 86_400_000;

const HELP = `Build HumTube's video catalog from YouTube.

Usage: YOUTUBE_API_KEY=... node scripts/build-catalog.mjs [options]

Each person gets one YouTube search for their name, ranked by YouTube's own
relevance ordering (the same as searching on youtube.com). Results that pass the
quality bar are kept in YouTube's order.

Selection
  --people <ids>          comma-separated person ids (default: all 100)
  --category <id>         ${CATEGORIES.map((c) => c.id).join(' | ')}

Search
  --lang <code>           relevanceLanguage passed to YouTube (default en)
  --refresh-days <n>      redo searches older than n days (default 30)
  --stats-days <n>        refresh view/like/subscriber counts older than n days (default 7)

Quality bar (defaults are HumTube's rules)
  --min-views <n>         default ${DEFAULT_THRESHOLDS.minViews}
  --min-likes <n>         default ${DEFAULT_THRESHOLDS.minLikes}
  --min-subscribers <n>   default ${DEFAULT_THRESHOLDS.minSubscribers}
  --min-duration <sec>    default ${DEFAULT_THRESHOLDS.minDurationSec} (skips Shorts; 0 disables)

Quota & output
  --budget <units>        max quota units this run may spend (default 9500;
                          a search costs 100, a stats lookup 1)
  --offline               no API calls; rebuild from the cache only
  --fresh                 don't carry over people from the existing output file
                          that aren't in the cache (default: keep them)
  --dry-run               print the plan and quota estimate, then exit
  --out <file>            default data/catalog.json
  --cache <dir>           default .cache
  -h, --help

Environment
  YOUTUBE_API_KEY         your YouTube Data API v3 key
  YT_API_BASE             override the API base URL (tests)
`;

function parseCli(argv) {
  const { values } = parseArgs({
    args: argv,
    options: {
      people: { type: 'string' },
      category: { type: 'string' },
      lang: { type: 'string', default: 'en' },
      'refresh-days': { type: 'string', default: '30' },
      'stats-days': { type: 'string', default: '7' },
      'min-views': { type: 'string' },
      'min-likes': { type: 'string' },
      'min-subscribers': { type: 'string' },
      'min-duration': { type: 'string' },
      budget: { type: 'string', default: '9500' },
      offline: { type: 'boolean', default: false },
      fresh: { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
      out: { type: 'string', default: 'data/catalog.json' },
      cache: { type: 'string', default: '.cache' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  const num = (name, fallback) => {
    const raw = values[name];
    if (raw == null) return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) throw new Error(`--${name} must be a non-negative number`);
    return n;
  };

  let people = PEOPLE;
  if (values.category) {
    if (!CATEGORIES.some((c) => c.id === values.category)) throw new Error(`Unknown --category ${values.category}`);
    people = people.filter((p) => p.category === values.category);
  }
  if (values.people) {
    const wanted = values.people.split(',').map((s) => s.trim()).filter(Boolean);
    const unknown = wanted.filter((id) => !PEOPLE.some((p) => p.id === id));
    if (unknown.length) throw new Error(`Unknown person id(s): ${unknown.join(', ')}`);
    people = people.filter((p) => wanted.includes(p.id));
  }

  return {
    help: values.help,
    people,
    lang: values.lang,
    refreshMs: num('refresh-days', 30) * DAY,
    statsMs: num('stats-days', 7) * DAY,
    thresholds: {
      minViews: num('min-views', DEFAULT_THRESHOLDS.minViews),
      minLikes: num('min-likes', DEFAULT_THRESHOLDS.minLikes),
      minSubscribers: num('min-subscribers', DEFAULT_THRESHOLDS.minSubscribers),
      minDurationSec: num('min-duration', DEFAULT_THRESHOLDS.minDurationSec),
    },
    budget: num('budget', 9500),
    offline: values.offline,
    fresh: values.fresh,
    dryRun: values['dry-run'],
    out: resolve(ROOT, values.out),
    cacheDir: resolve(ROOT, values.cache),
  };
}

async function readJson(file, fallback) {
  try {
    return JSON.parse(await readFile(file, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return fallback;
    throw new Error(`Could not read ${file}: ${err.message}`);
  }
}

async function writeJsonAtomic(file, text) {
  await mkdir(dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  await writeFile(tmp, text);
  await rename(tmp, file);
}

/** One video per line: small diffs in git, still valid JSON. */
export function serializeCatalog(catalog) {
  const { videos, ...rest } = catalog;
  const head = JSON.stringify(rest, null, 2).replace(/\n}$/, '');
  const body = videos.map((v) => `    ${JSON.stringify(v)}`).join(',\n');
  return `${head},\n  "videos": [${videos.length ? `\n${body}\n  ` : ''}]\n}\n`;
}

const searchKey = (personId, query, order) => `${personId}::${order}::${query}`;
const fmt = (n) => n.toLocaleString('en-US');

export async function run(argv = process.argv.slice(2), { env = process.env, log = console.log, fetchImpl } = {}) {
  const opts = parseCli(argv);
  if (opts.help) {
    log(HELP);
    return { exitCode: 0 };
  }

  const cacheFiles = {
    searches: resolve(opts.cacheDir, 'searches.json'),
    videos: resolve(opts.cacheDir, 'videos.json'),
    channels: resolve(opts.cacheDir, 'channels.json'),
  };
  const cache = {
    searches: await readJson(cacheFiles.searches, {}),
    videos: await readJson(cacheFiles.videos, {}),
    channels: await readJson(cacheFiles.channels, {}),
  };
  const saveCache = async (name) => writeJsonAtomic(cacheFiles[name], JSON.stringify(cache[name]));
  const now = Date.now();
  const stale = (iso, maxAge) => !iso || now - Date.parse(iso) > maxAge;

  // The existing output file counts as a cache too: cloud sessions and fresh
  // clones start without .cache/, and re-searching everyone would waste a day's quota.
  const previous = opts.fresh ? null : await readJson(opts.out, null);
  const inPrevious = (personId, query) => {
    const r = previous?.people?.[personId];
    return Boolean(r && r.query === query && r.order === SEARCH_ORDER && !stale(r.searchedAt, opts.refreshMs));
  };
  // The current search for each person, if it's cached.
  const cachedSearch = (p) => cache.searches[searchKey(p.id, queryFor(p), SEARCH_ORDER)];

  const plan = [];
  for (const p of opts.people) {
    const query = queryFor(p);
    if (stale(cachedSearch(p)?.fetchedAt, opts.refreshMs) && !inPrevious(p.id, query)) plan.push({ person: p, query });
  }

  log(`HumTube catalog builder — ${opts.people.length} ${opts.people.length === 1 ? 'person' : 'people'}, one YouTube search each`);
  log(`Bar: ≥${fmt(opts.thresholds.minViews)} views · ≥${fmt(opts.thresholds.minLikes)} likes · channel ≥${fmt(opts.thresholds.minSubscribers)} subscribers · ≥${opts.thresholds.minDurationSec}s long`);
  log(`Searches needed: ${plan.length} (≈${fmt(plan.length * UNIT_COST.search)} units). Budget this run: ${fmt(opts.budget)} units.`);

  if (opts.dryRun) {
    for (const { person, query } of plan) log(`  would search: ${person.id.padEnd(24)} "${query}"`);
    return { exitCode: 0, plan };
  }

  let used = 0;
  let quotaHit = null;
  const apiKey = env.YOUTUBE_API_KEY?.trim() || undefined;
  const client = createClient({
    apiKey,
    baseUrl: env.YT_API_BASE || undefined,
    fetchImpl: fetchImpl ?? globalThis.fetch,
    onRequest: (_endpoint, units) => {
      used += units;
    },
  });
  const canSpend = (units) => used + units <= opts.budget;

  if (!opts.offline) {
    if (!apiKey) log('Note: YOUTUBE_API_KEY is not set; continuing in case a proxy adds the key header.');

    // ── 1. Searches (100 units each). Keep units in reserve for the stats
    // lookups that follow: ~2 units (videos + channels) per 50 video IDs.
    const cachedStale = new Set(
      opts.people.map(cachedSearch).filter(Boolean)
        .flatMap((s) => s.videoIds)
        .filter((id) => stale(cache.videos[id]?.fetchedAt, opts.statsMs)),
    ).size;
    const baseReserve = 2 * Math.ceil(cachedStale / 50) + 4;
    let done = 0;
    for (const { person, query } of plan) {
      const reserve = baseReserve + 2 * (done + 1);
      if (!canSpend(UNIT_COST.search + reserve)) {
        log(`Budget reached after ${done} searches; ${plan.length - done} left for the next run.`);
        break;
      }
      try {
        const r = await client.search(query, { order: SEARCH_ORDER, relevanceLanguage: opts.lang });
        cache.searches[searchKey(person.id, query, SEARCH_ORDER)] = {
          personId: person.id,
          query,
          order: SEARCH_ORDER,
          fetchedAt: new Date().toISOString(),
          videoIds: r.items.map((i) => i.videoId),
        };
        await saveCache('searches');
        done += 1;
        log(`  searched ${String(done).padStart(3)}/${plan.length}  ${person.name} — "${query}" → ${r.items.length} results`);
      } catch (err) {
        if (err instanceof YouTubeError && (err.isQuota || err.isAuth)) {
          quotaHit = err;
          break;
        }
        throw err;
      }
    }
  }

  // Searches for the people in scope (cached ones included).
  const searchesInScope = opts.people.map(cachedSearch).filter(Boolean);

  if (!opts.offline && !quotaHit) {
    try {
      // ── 2. Video stats (1 unit per 50 videos).
      const allIds = [...new Set(searchesInScope.flatMap((s) => s.videoIds))];
      const needVideos = allIds.filter((id) => stale(cache.videos[id]?.fetchedAt, opts.statsMs));
      if (needVideos.length) {
        const units = Math.ceil(needVideos.length / 50);
        if (!canSpend(units)) throw new YouTubeError('Run budget too small for the video stats lookup', { reason: 'quotaExceeded' });
        const got = await client.videos(needVideos);
        const fetchedAt = new Date().toISOString();
        const seen = new Set();
        for (const v of got) {
          cache.videos[v.id] = { fetchedAt, video: v };
          seen.add(v.id);
        }
        // Deleted/private videos come back missing; remember that too.
        for (const id of needVideos) if (!seen.has(id)) cache.videos[id] = { fetchedAt, video: null };
        await saveCache('videos');
        log(`Fetched stats for ${fmt(got.length)} videos (${fmt(needVideos.length - got.length)} unavailable).`);
      }

      // ── 3. Channel subscriber counts, only for videos that passed so far.
      const videos = Object.fromEntries(Object.entries(cache.videos).filter(([, e]) => e.video).map(([id, e]) => [id, e.video]));
      const chIds = channelIdsToCheck({ searches: searchesInScope, videos, thresholds: opts.thresholds });
      const needChannels = chIds.filter((id) => stale(cache.channels[id]?.fetchedAt, opts.statsMs));
      if (needChannels.length) {
        const units = Math.ceil(needChannels.length / 50);
        if (!canSpend(units)) throw new YouTubeError('Run budget too small for the channel lookup', { reason: 'quotaExceeded' });
        const got = await client.channels(needChannels);
        const fetchedAt = new Date().toISOString();
        const seen = new Set();
        for (const c of got) {
          cache.channels[c.id] = { fetchedAt, channel: c };
          seen.add(c.id);
        }
        for (const id of needChannels) if (!seen.has(id)) cache.channels[id] = { fetchedAt, channel: null };
        await saveCache('channels');
        log(`Fetched subscriber counts for ${fmt(got.length)} channels.`);
      }
    } catch (err) {
      if (err instanceof YouTubeError && (err.isQuota || err.isAuth)) quotaHit = err;
      else throw err;
    }
  }

  // ── 4. Build the catalog from everything cached (all people, not just this run's).
  const allSearches = PEOPLE.map(cachedSearch).filter(Boolean);
  const videos = Object.fromEntries(Object.entries(cache.videos).filter(([, e]) => e.video).map(([id, e]) => [id, e.video]));
  const channels = Object.fromEntries(Object.entries(cache.channels).filter(([, e]) => e.channel).map(([id, e]) => [id, e.channel]));
  let catalog = buildCatalog({ searches: allSearches, videos, channels, thresholds: opts.thresholds });

  // A fresh clone has a committed catalog but no .cache/. Keep the people that
  // aren't cached yet instead of silently dropping them from the output.
  // Only results from the current search method (one YouTube relevance search) carry over.
  const carry = Object.entries(previous?.people ?? {})
    .filter(([id, r]) => PEOPLE_BY_ID.has(id) && !catalog.people[id] && r.order === SEARCH_ORDER && r.query === queryFor(PEOPLE_BY_ID.get(id)))
    .map(([id]) => id);
  if (carry.length) {
    const keep = new Set(carry);
    const t = opts.thresholds;
    const carried = {
      people: Object.fromEntries(carry.map((id) => [id, previous.people[id]])),
      videos: (previous.videos ?? [])
        .filter((v) => (v.personIds ?? []).some((id) => keep.has(id)) &&
          v.views >= t.minViews && v.likes >= t.minLikes && v.channelSubscribers >= t.minSubscribers &&
          (t.minDurationSec === 0 || v.durationSec >= t.minDurationSec))
        .map((v) => ({
          ...v,
          personIds: v.personIds.filter((id) => keep.has(id)),
          ranks: Object.fromEntries(Object.entries(v.ranks ?? {}).filter(([id]) => keep.has(id))),
        })),
    };
    catalog = { ...mergeCatalogs(carried, catalog), generatedAt: catalog.generatedAt, thresholds: catalog.thresholds };
    log(`Kept ${carry.length} ${carry.length === 1 ? 'person' : 'people'} from the existing ${opts.out} that ${carry.length === 1 ? "isn't" : "aren't"} in the cache yet (--fresh to drop).`);
  }
  await writeJsonAtomic(opts.out, serializeCatalog(catalog));

  // ── 5. Report.
  const counts = new Map();
  for (const v of catalog.videos) for (const id of v.personIds) counts.set(id, (counts.get(id) ?? 0) + 1);
  log('');
  log(`Wrote ${opts.out} — ${fmt(catalog.videos.length)} videos.`);
  const searched = PEOPLE.filter((p) => catalog.people[p.id]);
  const empty = searched.filter((p) => !counts.get(p.id));
  const notSearched = PEOPLE.filter((p) => !catalog.people[p.id]);
  log(`People searched: ${searched.length}/100 · with at least one video: ${searched.length - empty.length}`);
  if (empty.length) {
    log('Searched but nothing passed the bar:');
    for (const p of empty) {
      const r = catalog.people[p.id];
      const why = Object.entries(r.rejected).sort((a, b) => b[1] - a[1]).slice(0, 2)
        .map(([k, n]) => `${n} ${REJECTION_LABELS[k] ?? k}`).join(', ');
      log(`  ${p.name.padEnd(32)} ${r.candidates} candidates — ${why || 'no results'}`);
    }
  }
  if (notSearched.length) log(`Not searched yet: ${notSearched.length} (run again to continue).`);
  if (!opts.offline) log(`Quota used this run: ~${fmt(used)} units.`);

  if (quotaHit) {
    if (quotaHit.isAuth) {
      log(`\nYouTube refused the request: ${quotaHit.message}`);
      log('Check YOUTUBE_API_KEY and that "YouTube Data API v3" is enabled for its Google Cloud project.');
      return { exitCode: 2, catalog };
    }
    log(`\nStopped early: ${quotaHit.message}`);
    log('Progress is cached. Run the same command again after the quota resets (midnight Pacific time).');
    return { exitCode: 3, catalog };
  }
  return { exitCode: 0, catalog };
}

const invokedDirectly = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  run().then(
    ({ exitCode }) => process.exit(exitCode),
    (err) => {
      console.error(`Error: ${err.message}`);
      process.exit(1);
    },
  );
}
