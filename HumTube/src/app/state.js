// Loads the catalog, merges live-mode results and indexes everything for the views.

import { PEOPLE, PEOPLE_BY_ID } from '../core/people.js';
import { mergeCatalogs } from '../core/catalog.js';
import { DEFAULT_THRESHOLDS } from '../core/filters.js';
import { VIDEO_ID_RE } from '../core/youtube.js';

const LS = {
  key: 'humtube.apiKey',
  live: 'humtube.live.v1',
  saved: 'humtube.saved.v1',
};
export const LIVE_TTL_MS = 7 * 86_400_000;

function lsGet(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}
function lsSet(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export const settings = {
  get apiKey() {
    return lsGet(LS.key, '') || '';
  },
  set apiKey(v) {
    lsSet(LS.key, v ? String(v).trim() : null);
  },
};

// ── Saved ("Watch later") ───────────────────────────────────────────────
export const saved = {
  list() {
    const l = lsGet(LS.saved, []);
    return Array.isArray(l) ? l.filter((id) => VIDEO_ID_RE.test(id)) : [];
  },
  has(id) {
    return this.list().includes(id);
  },
  toggle(id) {
    const l = this.list();
    const next = l.includes(id) ? l.filter((x) => x !== id) : [id, ...l];
    lsSet(LS.saved, next);
    return next.includes(id);
  },
};

// ── Live-mode cache: { [personId]: { fetchedAt, catalog } } ─────────────
export const live = {
  all() {
    const all = lsGet(LS.live, {});
    const fresh = {};
    for (const [pid, entry] of Object.entries(all ?? {})) {
      if (PEOPLE_BY_ID.has(pid) && entry?.catalog && Date.now() - Date.parse(entry.fetchedAt) < LIVE_TTL_MS) {
        fresh[pid] = entry;
      }
    }
    return fresh;
  },
  put(personId, catalog) {
    const all = this.all();
    all[personId] = { fetchedAt: new Date().toISOString(), catalog };
    return lsSet(LS.live, all);
  },
  clear() {
    lsSet(LS.live, null);
  },
};

function sanitizeCatalog(raw) {
  const videos = Array.isArray(raw?.videos) ? raw.videos : [];
  return {
    ...raw,
    people: raw?.people && typeof raw.people === 'object' ? raw.people : {},
    thresholds: { ...DEFAULT_THRESHOLDS, ...(raw?.thresholds ?? {}) },
    videos: videos
      .filter((v) => v && VIDEO_ID_RE.test(v.id) && Array.isArray(v.personIds))
      .map((v) => ({
        ...v,
        personIds: v.personIds.filter((id) => PEOPLE_BY_ID.has(id)),
        topics: Array.isArray(v.topics) && v.topics.length ? v.topics : ['other'],
        views: Number(v.views) || 0,
        likes: v.likes == null ? null : Number(v.likes),
      })),
  };
}

export async function loadBaseCatalog(url) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { catalog: sanitizeCatalog(await res.json()), error: null };
  } catch (err) {
    return { catalog: sanitizeCatalog({}), error: err.message };
  }
}

/** Build the in-memory index the views read from. */
export function buildIndex(base, extraFragments = []) {
  const liveEntries = live.all();
  const merged = sanitizeCatalog(
    mergeCatalogs(base, ...Object.values(liveEntries).map((e) => e.catalog), ...extraFragments),
  );
  const videos = new Map(merged.videos.map((v) => [v.id, v]));
  const byPerson = new Map(PEOPLE.map((p) => [p.id, []]));
  const byTopic = new Map();
  for (const v of merged.videos) {
    for (const pid of v.personIds) byPerson.get(pid)?.push(v);
    for (const t of v.topics) {
      if (!byTopic.has(t)) byTopic.set(t, []);
      byTopic.get(t).push(v);
    }
  }
  return {
    generatedAt: base.generatedAt ?? null,
    thresholds: merged.thresholds,
    reports: merged.people,
    liveFetchedAt: {
      ...Object.fromEntries(Object.entries(liveEntries).map(([k, e]) => [k, e.fetchedAt])),
      ...Object.fromEntries(extraFragments.flatMap((f) => Object.entries(f.people ?? {}).map(([k, r]) => [k, r.searchedAt]))),
    },
    videos,
    list: merged.videos,
    byPerson,
    byTopic,
  };
}
