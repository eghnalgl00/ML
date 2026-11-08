import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { createClient, YouTubeError } from '../src/core/youtube.js';
import { buildCatalog, mergeCatalogs, fetchPersonLive } from '../src/core/catalog.js';
import { PEOPLE_BY_ID } from '../src/core/people.js';
import { run } from '../scripts/build-catalog.mjs';
import { createMockFetch } from './helpers/mock-youtube.js';

const quiet = () => {};

test('client: key param, batching by 50, unit accounting, entity decoding', async () => {
  const mock = createMockFetch();
  let units = 0;
  const client = createClient({ apiKey: 'TESTKEY', fetchImpl: mock.fetchImpl, onRequest: (_e, u) => { units += u; } });
  const s = await client.search('Winston Churchill');
  assert.equal(s.items.length, 10);
  assert.equal(mock.calls[0].params.key, 'TESTKEY');
  assert.equal(mock.calls[0].params.type, 'video');
  assert.equal(mock.calls[0].params.order, 'relevance', 'YouTube’s own ranking');
  const ids = Array.from({ length: 120 }, (_, i) => `v${String(i).padStart(10, '0')}`);
  await client.videos(ids);
  assert.equal(mock.calls.filter((c) => c.endpoint === 'videos').length, 3);
  assert.equal(units, 100 + 3);
});

test('client: quota and auth errors are typed', async () => {
  const quota = createClient({ fetchImpl: createMockFetch({ quotaAfterSearches: 0 }).fetchImpl });
  await assert.rejects(quota.search('x'), (e) => e instanceof YouTubeError && e.isQuota);
  const bad = createClient({ fetchImpl: createMockFetch({ badKey: true }).fetchImpl });
  await assert.rejects(bad.search('x'), (e) => e instanceof YouTubeError && e.isAuth);
});

test('live fetch: one YouTube search, only the bar applied, YouTube’s order kept', async () => {
  const mock = createMockFetch();
  const client = createClient({ fetchImpl: mock.fetchImpl });
  const cat = await fetchPersonLive(PEOPLE_BY_ID.get('winston-churchill'), { client });
  assert.equal(mock.calls.filter((c) => c.endpoint === 'search').length, 1);
  assert.deepEqual(cat.videos.map((v) => v.id).sort(), ['catsTopTen4', 'churchDoc01', 'churchHitl1']);
  const r = cat.people['winston-churchill'];
  assert.equal(r.query, 'Winston Churchill');
  assert.equal(r.order, 'relevance');
  assert.equal(r.candidates, 10);
  assert.equal(r.accepted, 3);
  assert.deepEqual(r.rejected, { subscribers: 2, duration: 2, views: 1, likes: 1, missing: 1 });
  const byId = Object.fromEntries(cat.videos.map((v) => [v.id, v]));
  assert.deepEqual(byId.churchDoc01.ranks, { 'winston-churchill': 1 });
  assert.deepEqual(byId.catsTopTen4.ranks, { 'winston-churchill': 5 }, 'position in YouTube’s results');
  assert.deepEqual(byId.churchHitl1.personIds, ['winston-churchill'], 'no extra tagging beyond YouTube’s results');
  assert.equal(byId.churchHitl1.channelSubscribers, 2_400_000);
  assert.equal(mock.calls.filter((c) => c.endpoint === 'channels').length, 1);
});

test('mergeCatalogs unions people and ranks, newer stats win', () => {
  const a = { generatedAt: 'x', people: { p1: { accepted: 1 } }, videos: [{ id: 'aaaaaaaaaaa', views: 10, personIds: ['p1'], ranks: { p1: 3 } }] };
  const b = { people: { p2: { accepted: 1 } }, videos: [{ id: 'aaaaaaaaaaa', views: 20, personIds: ['p2'], ranks: { p2: 1 } }, { id: 'bbbbbbbbbbb', views: 5, personIds: ['p2'], ranks: { p2: 2 } }] };
  const m = mergeCatalogs(a, b);
  assert.deepEqual(Object.keys(m.people).sort(), ['p1', 'p2']);
  assert.equal(m.videos[0].views, 20);
  assert.deepEqual(m.videos[0].personIds, ['p1', 'p2']);
  assert.deepEqual(m.videos[0].ranks, { p1: 3, p2: 1 });
  assert.equal(m.videos.length, 2);
});

async function tmp() {
  const dir = await mkdtemp(join(tmpdir(), 'humtube-'));
  return { dir, out: join(dir, 'catalog.json'), cache: join(dir, 'cache') };
}
const THREE = ['--people', 'winston-churchill,mohandas-gandhi,nelson-mandela'];
const ALL_EIGHT = ['catsTopTen4', 'churchDoc01', 'churchHitl1', 'edgeExactl1', 'gandhiFilm1', 'indiraGand1', 'mandelaDoc1', 'mandelaEff1'];

test('pipeline: builds, caches, resumes and rebuilds offline', async (t) => {
  const { dir, out, cache } = await tmp();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const args = [...THREE, '--out', out, '--cache', cache];

  const m1 = createMockFetch();
  const r1 = await run(args, { env: {}, log: quiet, fetchImpl: m1.fetchImpl });
  assert.equal(r1.exitCode, 0);
  assert.equal(m1.searches, 3, 'one search per person');
  const cat = JSON.parse(await readFile(out, 'utf8'));
  assert.deepEqual(cat.videos.map((v) => v.id).sort(), ALL_EIGHT);
  assert.ok(cat.videos.find((v) => v.id === 'gandhiFilm1').topics.includes('film'));
  assert.deepEqual(Object.keys(cat.people).sort(), ['mohandas-gandhi', 'nelson-mandela', 'winston-churchill']);

  // Second run: searches and stats are fresh in the cache → no API calls at all.
  const m2 = createMockFetch();
  const r2 = await run(args, { env: {}, log: quiet, fetchImpl: m2.fetchImpl });
  assert.equal(r2.exitCode, 0);
  assert.equal(m2.calls.length, 0);

  // Offline rebuild with a stricter bar, no fetch at all.
  const r3 = await run([...args, '--offline', '--min-views', '1000000'], { env: {}, log: quiet, fetchImpl: () => { throw new Error('no network in offline mode'); } });
  assert.equal(r3.exitCode, 0);
  assert.deepEqual(r3.catalog.videos.map((v) => v.id).sort(), ['catsTopTen4', 'churchDoc01', 'gandhiFilm1', 'indiraGand1', 'mandelaDoc1', 'mandelaEff1']);
});

test('pipeline: the plan is one search per person, ranked by YouTube relevance', async () => {
  const r = await run(['--people', 'winston-churchill,unknown-rebel', '--dry-run', '--cache', '/nonexistent-humtube/cache', '--out', '/nonexistent-humtube/catalog.json'], { log: quiet });
  assert.deepEqual(r.plan.map((p) => [p.person.id, p.query]), [
    ['winston-churchill', 'Winston Churchill'],
    ['unknown-rebel', 'Tiananmen Square Tank Man'],
  ]);
});

test('pipeline: respects the quota budget and resumes the next day', async (t) => {
  const { dir, out, cache } = await tmp();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const args = [...THREE, '--out', out, '--cache', cache];

  const m1 = createMockFetch();
  const r1 = await run([...args, '--budget', '210'], { env: {}, log: quiet, fetchImpl: m1.fetchImpl });
  assert.equal(r1.exitCode, 0);
  assert.equal(m1.searches, 2, 'only two 100-unit searches fit in 210 units with the stats reserve');
  assert.equal(Object.keys(r1.catalog.people).length, 2);

  const m2 = createMockFetch();
  const r2 = await run(args, { env: {}, log: quiet, fetchImpl: m2.fetchImpl });
  assert.equal(m2.searches, 1, 'only the missing person is searched');
  assert.equal(Object.keys(r2.catalog.people).length, 3);
});

test('pipeline: quota exhaustion mid-run keeps progress and exits 3', async (t) => {
  const { dir, out, cache } = await tmp();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const args = [...THREE, '--out', out, '--cache', cache];
  const r1 = await run(args, { env: {}, log: quiet, fetchImpl: createMockFetch({ quotaAfterSearches: 1 }).fetchImpl });
  assert.equal(r1.exitCode, 3);
  const searches = JSON.parse(await readFile(join(cache, 'searches.json'), 'utf8'));
  assert.equal(Object.keys(searches).length, 1);

  const r2 = await run(args, { env: {}, log: quiet, fetchImpl: createMockFetch().fetchImpl });
  assert.equal(r2.exitCode, 0);
  assert.equal(r2.catalog.videos.length, 8);
});

test('pipeline: a bad API key exits 2 with nothing searched', async (t) => {
  const { dir, out, cache } = await tmp();
  t.after(() => rm(dir, { recursive: true, force: true }));
  const r = await run(['--people', 'winston-churchill', '--out', out, '--cache', cache], { env: { YOUTUBE_API_KEY: 'nope' }, log: quiet, fetchImpl: createMockFetch({ badKey: true }).fetchImpl });
  assert.equal(r.exitCode, 2);
  assert.equal(r.catalog.videos.length, 0);
});

test('pipeline: a fresh clone (no cache) never loses the committed catalog', async (t) => {
  const { dir, out, cache } = await tmp();
  t.after(() => rm(dir, { recursive: true, force: true }));
  await run(['--people', 'winston-churchill', '--out', out, '--cache', cache], { env: {}, log: quiet, fetchImpl: createMockFetch().fetchImpl });
  await rm(cache, { recursive: true, force: true }); // simulate a fresh clone

  // Bad key: nothing new, but Churchill's videos must survive.
  const bad = await run(['--people', 'mohandas-gandhi', '--out', out, '--cache', cache], { env: {}, log: quiet, fetchImpl: createMockFetch({ badKey: true }).fetchImpl });
  assert.equal(bad.exitCode, 2);
  let cat = JSON.parse(await readFile(out, 'utf8'));
  assert.deepEqual(cat.videos.map((v) => v.id).sort(), ['catsTopTen4', 'churchDoc01', 'churchHitl1']);

  // Good run for Gandhi: Churchill carried over, Gandhi added.
  const ok = await run(['--people', 'mohandas-gandhi', '--out', out, '--cache', cache], { env: {}, log: quiet, fetchImpl: createMockFetch().fetchImpl });
  assert.equal(ok.exitCode, 0);
  cat = JSON.parse(await readFile(out, 'utf8'));
  assert.deepEqual(cat.videos.map((v) => v.id).sort(), ['catsTopTen4', 'churchDoc01', 'churchHitl1', 'gandhiFilm1', 'indiraGand1']);

  // People already in the committed catalog aren't searched again.
  const again = createMockFetch();
  const r = await run(['--people', 'winston-churchill,mohandas-gandhi', '--out', out, '--cache', cache], { env: {}, log: quiet, fetchImpl: again.fetchImpl });
  assert.equal(r.exitCode, 0);
  assert.equal(again.searches, 0);
  assert.equal(r.catalog.videos.length, 5);

  // --fresh drops what isn't cached.
  const fresh = await run(['--people', 'mohandas-gandhi', '--out', out, '--cache', cache, '--offline', '--fresh'], { env: {}, log: quiet });
  assert.deepEqual(fresh.catalog.videos.map((v) => v.id), ['gandhiFilm1', 'indiraGand1']);
});

test('pipeline: rejects unknown people and bad flags', async () => {
  await assert.rejects(run(['--people', 'nobody'], { log: quiet }), /Unknown person/);
  await assert.rejects(run(['--order', 'viewCount'], { log: quiet }), /--order/);
  await assert.rejects(run(['--min-views=-5'], { log: quiet }), /non-negative/);
});

test('buildCatalog with no searches is empty', () => {
  const cat = buildCatalog({ searches: [], videos: {}, channels: {}, generatedAt: null });
  assert.deepEqual(cat.videos, []);
  assert.equal(cat.thresholds.minViews, 500_000);
});
