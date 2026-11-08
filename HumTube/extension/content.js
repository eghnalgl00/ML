// HumTube: runs on youtube.com. YouTube stays YouTube (sign-in, watch page,
// comments, channels, subscriptions); only the videos that appear change:
//   - Home: YouTube's feed is replaced by the HumTube videos.
//   - Everywhere: video tiles that aren't HumTube videos are hidden
//     (recommendations, search results, Shorts shelves).
//   - Watch page: "More about <person>" from HumTube on top of the sidebar.
//   - Search: matching HumTube videos are listed above the results.
// The "HumTube" button in YouTube's top bar turns it on and off.
(() => {
  'use strict';

  const ID_RE = /(?:[?&]v=|\/shorts\/)([A-Za-z0-9_-]{11})/;
  const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
  const CHANNEL_ID = /^UC[A-Za-z0-9_-]{22}$/;
  const TILES = [
    'ytd-rich-item-renderer', 'ytd-video-renderer', 'ytd-compact-video-renderer', 'ytd-grid-video-renderer',
    'ytd-playlist-video-renderer', 'ytd-reel-item-renderer', 'ytd-radio-renderer', 'ytd-compact-radio-renderer',
    'ytd-playlist-renderer', 'ytd-compact-playlist-renderer', 'ytd-movie-renderer', 'yt-lockup-view-model',
    'ytm-shorts-lockup-view-model', 'ytm-shorts-lockup-view-model-v2',
  ].join(',');
  const PAGE = 48;

  const root = document.documentElement;
  let data = null; // { people, topics, videos, order }
  let byId = new Map();
  let byPerson = new Map();
  let enabled = true;
  let topic = 'all';
  let shown = PAGE;
  root.classList.add('humtube-on'); // hide YouTube's feed right away, no flash

  // ── Formatting ──────────────────────────────────────────────────────────
  const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
  const duration = (s) => {
    if (!s) return '';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
  };
  const ago = (iso) => {
    const sec = (Date.now() - Date.parse(iso)) / 1000;
    for (const [n, size] of [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600]]) {
      const k = Math.floor(sec / size);
      if (k >= 1) return `${k} ${n}${k === 1 ? '' : 's'} ago`;
    }
    return 'just now';
  };

  // DOM building without innerHTML (YouTube enforces Trusted Types).
  function el(tag, props = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'onclick') e.addEventListener('click', v);
      else e.setAttribute(k, v);
    }
    for (const c of children) if (c) e.append(c);
    return e;
  }

  function tile(v, cls) {
    const url = `/watch?v=${v.id}`;
    return el('div', { class: cls },
      el('a', { class: 'ht-thumb', href: url },
        el('img', { src: `https://i.ytimg.com/vi/${v.id}/mqdefault.jpg`, alt: '', loading: 'lazy' }),
        v.durationSec ? el('span', { class: 'ht-dur', text: duration(v.durationSec) }) : null),
      el('div', { class: 'ht-meta' },
        el('a', { href: url, title: v.title }, el('h3', { class: 'ht-title', text: v.title })),
        el('p', { class: 'ht-line' }, CHANNEL_ID.test(v.channelId)
          ? el('a', { href: `/channel/${v.channelId}`, text: v.channelTitle })
          : el('span', { text: v.channelTitle })),
        el('p', { class: 'ht-line', text: `${compact.format(v.views)} views${v.publishedAt ? ` · ${ago(v.publishedAt)}` : ''}` })));
  }

  // ── Hide every YouTube video tile that isn't a HumTube video ────────────
  function filterTiles() {
    for (const t of document.querySelectorAll(TILES)) {
      if (t.closest('.humtube-ui')) continue;
      const a = t.querySelector('a[href*="/watch?v="], a[href*="/shorts/"]');
      const m = a && ID_RE.exec(a.getAttribute('href') || '');
      t.classList.toggle('humtube-hidden', !(m && byId.has(m[1])));
    }
  }

  // ── Home: the HumTube feed in place of YouTube's ─────────────────────────
  function renderHome() {
    const browse = document.querySelector('ytd-browse[page-subtype="home"]');
    if (!browse) return;
    const host = browse.querySelector('#primary') || browse;
    let feed = document.getElementById('humtube-feed');
    if (!feed || !host.contains(feed)) {
      feed?.remove();
      feed = el('div', { id: 'humtube-feed', class: 'humtube-ui' });
      host.prepend(feed);
    }
    const key = `${topic}|${shown}`;
    if (feed.dataset.key === key) return;
    feed.dataset.key = key;

    const list = data.order.filter((v) => topic === 'all' || v.topics.includes(topic));
    const present = new Set(data.videos.flatMap((v) => v.topics));
    const chips = el('div', { class: 'ht-chips' },
      ...[['all', 'All'], ...Object.entries(data.topics).filter(([id]) => present.has(id))].map(([id, label]) =>
        el('button', {
          class: `ht-chip${id === topic ? ' is-on' : ''}`,
          type: 'button',
          text: label,
          onclick: () => { topic = id; shown = PAGE; schedule(); },
        })));
    const grid = el('div', { class: 'ht-grid' }, ...list.slice(0, shown).map((v) => tile(v, 'ht-item')));
    const more = list.length > shown
      ? el('p', { class: 'ht-more' }, el('button', { class: 'ht-chip', type: 'button', text: 'Show more', onclick: () => { shown += PAGE; schedule(); } }))
      : null;
    feed.replaceChildren(chips, grid, more ?? '');
  }

  // ── Watch page: HumTube videos at the top of the sidebar ────────────────
  function renderWatch() {
    const id = new URLSearchParams(location.search).get('v');
    const side = document.querySelector('ytd-watch-flexy #secondary-inner') || document.querySelector('ytd-watch-flexy #secondary');
    if (!side) return;
    let box = document.getElementById('humtube-related');
    if (box && box.dataset.key === id && side.contains(box)) return;
    box?.remove();
    const pid = byId.get(id)?.personIds[0];
    const list = (pid ? byPerson.get(pid) : data.order).filter((v) => v.id !== id).slice(0, 20);
    box = el('div', { id: 'humtube-related', class: 'humtube-ui' },
      el('h3', { class: 'ht-heading', text: pid ? `More about ${data.people[pid]?.name ?? ''}` : 'HumTube' }),
      ...list.map((v) => tile(v, 'ht-item')));
    box.dataset.key = id ?? '';
    side.prepend(box);
  }

  // ── Search: matching HumTube videos above YouTube's results ─────────────
  const norm = (s) => String(s ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  function renderSearch() {
    const q = new URLSearchParams(location.search).get('search_query') || '';
    const host = document.querySelector('ytd-search #primary') || document.querySelector('ytd-search');
    if (!host) return;
    let box = document.getElementById('humtube-results');
    if (box && box.dataset.key === q && host.contains(box)) return;
    box?.remove();
    const words = norm(q).split(/\s+/).filter(Boolean);
    const hits = data.order.filter((v) => {
      const text = norm(`${v.title} ${v.channelTitle} ${v.personIds.map((p) => data.people[p]?.name).join(' ')}`);
      return words.every((w) => text.includes(w));
    }).slice(0, 30);
    box = el('div', { id: 'humtube-results', class: 'humtube-ui' },
      el('h3', { class: 'ht-heading', text: hits.length ? `HumTube: ${hits.length} video${hits.length === 1 ? '' : 's'}` : 'No HumTube videos match this search' }),
      ...hits.map((v) => tile(v, 'ht-item')));
    box.dataset.key = q;
    host.prepend(box);
  }

  // ── On/off button in YouTube's top bar ──────────────────────────────────
  function addToggle() {
    const end = document.querySelector('ytd-masthead #end');
    if (!end) return;
    let btn = document.getElementById('humtube-toggle');
    if (!btn) {
      btn = el('button', {
        id: 'humtube-toggle',
        type: 'button',
        onclick: () => {
          enabled = !enabled;
          try { chrome.storage.local.set({ enabled }); } catch { /* storage unavailable */ }
          schedule();
        },
      });
      end.prepend(btn);
    }
    const label = enabled ? 'HumTube on' : 'HumTube off';
    if (btn.textContent !== label) {
      btn.textContent = label;
      btn.title = enabled ? 'Showing only HumTube videos. Click to see all of YouTube.' : 'Showing all of YouTube. Click to show only HumTube videos.';
    }
  }

  function apply() {
    root.classList.toggle('humtube-on', enabled);
    addToggle();
    if (!enabled || !data) return;
    filterTiles();
    if (location.pathname === '/') renderHome();
    else if (location.pathname === '/watch') renderWatch();
    else if (location.pathname === '/results') renderSearch();
  }

  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      apply();
    });
  }

  // ── Boot ────────────────────────────────────────────────────────────────
  try {
    chrome.storage.local.get({ enabled: true }, (r) => { enabled = r.enabled !== false; schedule(); });
    chrome.storage.onChanged.addListener((ch) => { if (ch.enabled) { enabled = ch.enabled.newValue !== false; schedule(); } });
  } catch { /* storage unavailable: stay on */ }

  fetch(chrome.runtime.getURL('catalog.json'))
    .then((r) => r.json())
    .then((c) => {
      const videos = (c.videos ?? []).filter((v) => VIDEO_ID.test(v.id));
      byId = new Map(videos.map((v) => [v.id, v]));
      const order = (c.order ?? []).map((id) => byId.get(id)).filter(Boolean);
      byPerson = new Map();
      for (const v of order) for (const p of v.personIds) {
        if (!byPerson.has(p)) byPerson.set(p, []);
        byPerson.get(p).push(v);
      }
      for (const [p, list] of byPerson) list.sort((a, b) => (a.ranks?.[p] ?? 1e9) - (b.ranks?.[p] ?? 1e9));
      data = { people: c.people ?? {}, topics: c.topics ?? {}, videos, order };
      schedule();
    })
    .catch((err) => console.warn('HumTube: could not load the video list', err));

  new MutationObserver(schedule).observe(root, { childList: true, subtree: true });
  window.addEventListener('yt-navigate-finish', () => { shown = PAGE; schedule(); });
})();
