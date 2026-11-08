// Turns YouTube search results into the HumTube catalog. Pure functions, plus one
// small async helper (`fetchPersonLive`) that the browser's live mode uses.
//
// Each person gets one YouTube search, sorted by YouTube's relevance ranking.
// A video is listed if it passes the bar; it keeps its position in YouTube's
// results (`ranks[personId]`), which is the default order on the site.
//
// Catalog shape (data/catalog.json):
// {
//   version, generatedAt, thresholds,
//   people: { [personId]: { query, order, searchedAt, candidates, accepted, rejected: {reason: n} } },
//   videos: [ { id, title, description, channelId, channelTitle, channelSubscribers,
//               publishedAt, durationSec, views, likes, comments, embeddable,
//               categoryId, topics: [...], personIds: [...], ranks: {personId: n} } ]
// }

import { PEOPLE, queryFor } from './people.js';
import { DEFAULT_THRESHOLDS, videoRejection, channelRejection } from './filters.js';
import { classify } from './topics.js';

export const CATALOG_VERSION = 2;
export const SEARCH_ORDER = 'relevance'; // YouTube's default search ranking

const latest = (dates) => dates.filter(Boolean).sort().at(-1) ?? null;

/** Only channels of videos that already pass the video-level checks (saves quota). */
export function channelIdsToCheck({ searches, videos, thresholds = DEFAULT_THRESHOLDS }) {
  const ids = new Set();
  for (const id of new Set(searches.flatMap((s) => s.videoIds))) {
    const v = videos[id];
    if (v && !videoRejection(v, thresholds)) ids.add(v.channelId);
  }
  return [...ids];
}

/**
 * @param {object} args
 * @param {Array}  args.searches  [{ personId, query, order, fetchedAt, videoIds }] (videoIds in YouTube's order)
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
  const accepted = new Map(); // videoId -> { personIds: [], ranks: {} }
  const report = {};

  for (const p of people) {
    const search = searches.find((s) => s.personId === p.id);
    if (!search) continue;
    const rejected = {};
    let ok = 0;
    search.videoIds.forEach((id, position) => {
      const v = videos[id];
      const reason = !v ? 'missing' : videoRejection(v, thresholds) ?? channelRejection(channels[v.channelId], thresholds);
      if (reason) {
        rejected[reason] = (rejected[reason] ?? 0) + 1;
        return;
      }
      ok += 1;
      if (!accepted.has(id)) accepted.set(id, { personIds: [], ranks: {} });
      const entry = accepted.get(id);
      entry.personIds.push(p.id);
      entry.ranks[p.id] = position + 1;
    });
    report[p.id] = {
      query: search.query,
      order: search.order,
      searchedAt: latest([search.fetchedAt]),
      candidates: search.videoIds.length,
      accepted: ok,
      rejected,
    };
  }

  const out = [];
  for (const [id, { personIds, ranks }] of accepted) {
    const v = videos[id];
    const ch = channels[v.channelId];
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
      ranks,
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
      videos.set(v.id, prev
        ? {
          ...prev,
          ...v,
          personIds: [...new Set([...prev.personIds, ...v.personIds])],
          ranks: { ...(prev.ranks ?? {}), ...(v.ranks ?? {}) },
        }
        : { ...v });
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
 * fragment. Costs ~100 quota units for the search, plus a few for stats.
 */
export async function fetchPersonLive(person, { client, thresholds = DEFAULT_THRESHOLDS, relevanceLanguage = 'en' }) {
  const query = queryFor(person);
  const r = await client.search(query, { order: SEARCH_ORDER, relevanceLanguage });
  const searches = [{
    personId: person.id,
    query,
    order: SEARCH_ORDER,
    fetchedAt: new Date().toISOString(),
    videoIds: r.items.map((i) => i.videoId),
  }];
  const ids = searches[0].videoIds;
  const videos = Object.fromEntries((ids.length ? await client.videos(ids) : []).map((v) => [v.id, v]));
  const channelIds = channelIdsToCheck({ searches, videos, thresholds });
  const channels = Object.fromEntries(
    (channelIds.length ? await client.channels(channelIds) : []).map((c) => [c.id, c]),
  );
  return buildCatalog({ people: [person], searches, videos, channels, thresholds });
}
