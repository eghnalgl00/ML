// Turns raw YouTube results into the HumTube catalog. Pure functions, plus one
// small async helper (`fetchPersonLive`) that the browser's live mode uses.
//
// Catalog shape (data/catalog.json):
// {
//   version, generatedAt, thresholds,
//   people: { [personId]: { queries, searchedAt, candidates, accepted, rejected: {reason: n} } },
//   videos: [ { id, title, description, channelId, channelTitle, channelSubscribers,
//               publishedAt, durationSec, views, likes, comments, embeddable,
//               categoryId, topics: [...], personIds: [...], foundFor: [...] } ]
// }

import { PEOPLE, searchesFor } from './people.js';
import { DEFAULT_THRESHOLDS, makeMatcher, videoRejection, channelRejection } from './filters.js';
import { classify } from './topics.js';

export const CATALOG_VERSION = 1;

function groupBy(list, key) {
  const m = new Map();
  for (const item of list) {
    const k = key(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(item);
  }
  return m;
}

const latest = (dates) => dates.filter(Boolean).sort().at(-1) ?? null;

/**
 * Channel IDs worth looking up: only channels of videos that already pass every
 * video-level check for the person they were found for (saves quota).
 */
export function channelIdsToCheck({ people = PEOPLE, searches, videos, thresholds = DEFAULT_THRESHOLDS }) {
  const byPerson = groupBy(searches, (s) => s.personId);
  const ids = new Set();
  for (const p of people) {
    const list = byPerson.get(p.id);
    if (!list) continue;
    const matcher = makeMatcher(p);
    for (const id of new Set(list.flatMap((s) => s.videoIds))) {
      const v = videos[id];
      if (v && !videoRejection(v, matcher, thresholds)) ids.add(v.channelId);
    }
  }
  return [...ids];
}

/**
 * @param {object} args
 * @param {Array}  args.searches  [{ personId, query, order, fetchedAt, videoIds }]
 * @param {object} args.videos    { [videoId]: normalizedVideo }
 * @param {object} args.channels  { [channelId]: normalizedChannel }
 */
export function buildCatalog({
  people = PEOPLE,
  searches,
  videos,
  channels,
  thresholds = DEFAULT_THRESHOLDS,
  generatedAt = new Date().toISOString(),
}) {
  const matchers = new Map(people.map((p) => [p.id, makeMatcher(p)]));
  const byPerson = groupBy(searches, (s) => s.personId);
  const accepted = new Map(); // videoId -> Set(personId it was found for)
  const report = {};

  for (const p of people) {
    const list = byPerson.get(p.id);
    if (!list) continue;
    const ids = [...new Set(list.flatMap((s) => s.videoIds))];
    const rejected = {};
    let ok = 0;
    for (const id of ids) {
      const v = videos[id];
      const reason = !v
        ? 'missing'
        : videoRejection(v, matchers.get(p.id), thresholds) ?? channelRejection(channels[v.channelId], thresholds);
      if (reason) {
        rejected[reason] = (rejected[reason] ?? 0) + 1;
        continue;
      }
      ok += 1;
      if (!accepted.has(id)) accepted.set(id, new Set());
      accepted.get(id).add(p.id);
    }
    report[p.id] = {
      queries: [...new Set(list.map((s) => s.query))],
      searchedAt: latest(list.map((s) => s.fetchedAt)),
      candidates: ids.length,
      accepted: ok,
      rejected,
    };
  }

  const out = [];
  for (const [id, found] of accepted) {
    const v = videos[id];
    const ch = channels[v.channelId];
    const personIds = [...found];
    for (const p of people) {
      if (!found.has(p.id) && matchers.get(p.id).mentions(v.title)) personIds.push(p.id);
    }
    out.push({
      id,
      title: v.title,
      description: (v.description ?? '').slice(0, 280),
      channelId: v.channelId,
      channelTitle: ch?.title || v.channelTitle,
      channelSubscribers: ch?.subscribers ?? null,
      publishedAt: v.publishedAt,
      durationSec: v.durationSec,
      views: v.views,
      likes: v.likes,
      comments: v.comments,
      embeddable: v.embeddable,
      categoryId: v.categoryId,
      topics: classify({ title: v.title, channelTitle: ch?.title || v.channelTitle, categoryId: v.categoryId }),
      personIds,
      foundFor: [...found],
    });
  }
  out.sort((a, b) => b.views - a.views || a.id.localeCompare(b.id));

  return {
    version: CATALOG_VERSION,
    generatedAt,
    thresholds: { ...thresholds },
    people: report,
    videos: out,
  };
}

/** Combine a base catalog with newer fragments (e.g. live-mode results). Later wins. */
export function mergeCatalogs(base, ...fragments) {
  const videos = new Map();
  const people = { ...(base?.people ?? {}) };
  for (const cat of [base, ...fragments]) {
    if (!cat) continue;
    Object.assign(people, cat.people ?? {});
    for (const v of cat.videos ?? []) {
      const prev = videos.get(v.id);
      if (!prev) {
        videos.set(v.id, { ...v });
      } else {
        videos.set(v.id, {
          ...prev,
          ...v,
          personIds: [...new Set([...prev.personIds, ...v.personIds])],
          foundFor: [...new Set([...(prev.foundFor ?? []), ...(v.foundFor ?? [])])],
        });
      }
    }
  }
  return {
    version: CATALOG_VERSION,
    generatedAt: base?.generatedAt ?? null,
    thresholds: base?.thresholds ?? { ...DEFAULT_THRESHOLDS },
    people,
    videos: [...videos.values()].sort((a, b) => b.views - a.views),
  };
}

/**
 * Live mode: search YouTube for one person right now and return a catalog
 * fragment. Costs ~100 quota units per query, plus a few for stats.
 */
export async function fetchPersonLive(person, {
  client,
  thresholds = DEFAULT_THRESHOLDS,
  maxQueries = 2,
  relevanceLanguage = 'en',
}) {
  const searches = [];
  for (const { query, order } of searchesFor(person).slice(0, maxQueries)) {
    const r = await client.search(query, { order, relevanceLanguage });
    searches.push({
      personId: person.id,
      query,
      order,
      fetchedAt: new Date().toISOString(),
      videoIds: r.items.map((i) => i.videoId),
    });
  }
  const ids = [...new Set(searches.flatMap((s) => s.videoIds))];
  const videos = Object.fromEntries((ids.length ? await client.videos(ids) : []).map((v) => [v.id, v]));
  const channelIds = channelIdsToCheck({ people: [person], searches, videos, thresholds });
  const channels = Object.fromEntries(
    (channelIds.length ? await client.channels(channelIds) : []).map((c) => [c.id, c]),
  );
  return buildCatalog({ people: PEOPLE, searches, videos, channels, thresholds });
}
