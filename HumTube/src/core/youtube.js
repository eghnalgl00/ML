// Minimal YouTube Data API v3 client (search, videos, channels).
// Works in browsers and Node 18+ (global fetch). No dependencies.
//
// Quota cost per call (default project quota is 10,000 units/day):
//   search.list = 100 units, videos.list = 1, channels.list = 1.

import { decodeEntities } from './text.js';

export const API_BASE = 'https://www.googleapis.com/youtube/v3';
export const UNIT_COST = Object.freeze({ search: 100, videos: 1, channels: 1 });
const BATCH = 50;

export class YouTubeError extends Error {
  constructor(message, { status = 0, reason = 'unknown' } = {}) {
    super(message);
    this.name = 'YouTubeError';
    this.status = status;
    this.reason = reason;
  }

  get isQuota() {
    return ['quotaExceeded', 'dailyLimitExceeded', 'rateLimitExceeded'].includes(this.reason);
  }

  get isAuth() {
    return ['keyInvalid', 'forbidden', 'accessNotConfigured', 'API_KEY_INVALID'].includes(this.reason) ||
      this.status === 401;
  }
}

export const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;

export function parseIsoDuration(iso) {
  const m = /^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/.exec(iso ?? '');
  if (!m) return 0;
  const [, d = 0, h = 0, min = 0, s = 0] = m;
  return Number(d) * 86400 + Number(h) * 3600 + Number(min) * 60 + Number(s);
}

const toCount = (v) => (v == null || v === '' ? null : Number(v));

/** Flatten a videos.list item into the shape the filters and catalog use. */
export function normalizeVideo(raw) {
  const s = raw.snippet ?? {};
  const st = raw.statistics ?? {};
  const cd = raw.contentDetails ?? {};
  const status = raw.status ?? {};
  return {
    id: raw.id,
    title: s.title ?? '',
    description: (s.description ?? '').slice(0, 600),
    tags: s.tags ?? [],
    channelId: s.channelId ?? '',
    channelTitle: s.channelTitle ?? '',
    publishedAt: s.publishedAt ?? null,
    categoryId: s.categoryId ?? null,
    lang: s.defaultAudioLanguage ?? s.defaultLanguage ?? null,
    live: s.liveBroadcastContent ?? 'none',
    durationSec: parseIsoDuration(cd.duration),
    views: toCount(st.viewCount) ?? 0,
    likes: toCount(st.likeCount), // null when the uploader hides likes
    comments: toCount(st.commentCount),
    embeddable: status.embeddable !== false,
    privacy: status.privacyStatus ?? 'public',
  };
}

export function normalizeChannel(raw) {
  const s = raw.snippet ?? {};
  const st = raw.statistics ?? {};
  return {
    id: raw.id,
    title: s.title ?? '',
    subscribers: st.hiddenSubscriberCount ? null : toCount(st.subscriberCount),
  };
}

function chunk(list, size) {
  const out = [];
  for (let i = 0; i < list.length; i += size) out.push(list.slice(i, i + size));
  return out;
}

/**
 * @param {object} opts
 * @param {string} [opts.apiKey]  sent as `key=`; may be omitted when a proxy injects
 *                                the X-Goog-Api-Key header for you
 * @param {Function} [opts.fetchImpl]
 * @param {string} [opts.baseUrl]
 * @param {(endpoint: string, units: number) => void} [opts.onRequest]
 */
export function createClient({ apiKey, fetchImpl = globalThis.fetch, baseUrl = API_BASE, onRequest } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('No fetch implementation available');

  async function call(endpoint, params) {
    const url = new URL(`${baseUrl.replace(/\/$/, '')}/${endpoint}`);
    for (const [k, v] of Object.entries(params)) {
      if (v != null && v !== '') url.searchParams.set(k, String(v));
    }
    if (apiKey) url.searchParams.set('key', apiKey);
    onRequest?.(endpoint, UNIT_COST[endpoint] ?? 1);

    let res;
    try {
      res = await fetchImpl(url.toString(), { headers: { Accept: 'application/json' } });
    } catch (err) {
      throw new YouTubeError(`Network error calling YouTube ${endpoint}: ${err.message}`, { reason: 'network' });
    }
    let body = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error page */
    }
    if (!res.ok) {
      const e = body?.error;
      const reason = e?.errors?.[0]?.reason ?? e?.details?.[0]?.reason ?? e?.status ?? `http${res.status}`;
      throw new YouTubeError(e?.message ?? `YouTube ${endpoint} failed with HTTP ${res.status}`, {
        status: res.status,
        reason,
      });
    }
    return body ?? {};
  }

  return {
    /** One search.list page (100 units). */
    async search(q, { maxResults = 50, order = 'viewCount', relevanceLanguage = 'en', pageToken } = {}) {
      const body = await call('search', {
        part: 'snippet',
        type: 'video',
        q,
        maxResults: Math.min(50, maxResults),
        order,
        relevanceLanguage,
        pageToken,
      });
      const items = (body.items ?? [])
        .map((i) => ({
          videoId: i.id?.videoId,
          title: decodeEntities(i.snippet?.title ?? ''),
          channelId: i.snippet?.channelId ?? '',
        }))
        .filter((i) => i.videoId && VIDEO_ID_RE.test(i.videoId));
      return { items, nextPageToken: body.nextPageToken ?? null, totalResults: body.pageInfo?.totalResults ?? null };
    },

    /** videos.list in batches of 50 (1 unit per batch). Returns normalized videos. */
    async videos(ids) {
      const out = [];
      for (const batch of chunk([...new Set(ids)], BATCH)) {
        const body = await call('videos', {
          part: 'snippet,statistics,contentDetails,status',
          id: batch.join(','),
          maxResults: BATCH,
        });
        for (const item of body.items ?? []) out.push(normalizeVideo(item));
      }
      return out;
    },

    /** channels.list in batches of 50 (1 unit per batch). Returns normalized channels. */
    async channels(ids) {
      const out = [];
      for (const batch of chunk([...new Set(ids)], BATCH)) {
        const body = await call('channels', {
          part: 'snippet,statistics',
          id: batch.join(','),
          maxResults: BATCH,
        });
        for (const item of body.items ?? []) out.push(normalizeChannel(item));
      }
      return out;
    },
  };
}
