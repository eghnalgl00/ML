// HTML view renderers. Every dynamic string goes through `esc`.

import {
  PEOPLE, PEOPLE_BY_ID, CATEGORIES, CATEGORY_BY_ID, PERSON_OF_THE_CENTURY, monogram, queriesFor,
} from '../core/people.js';
import { ALL_TOPICS, TOPIC_BY_ID } from '../core/topics.js';
import { REJECTION_LABELS } from '../core/filters.js';
import { fold } from '../core/text.js';
import { esc, short, full, duration, year, date, ago, plural } from './format.js';

// ── Links ───────────────────────────────────────────────────────────────
const qs = (params) => {
  const s = new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== '')).toString();
  return s ? `?${s}` : '';
};
export const href = {
  home: () => '#/',
  category: (id) => `#/c/${id}`,
  person: (id, params = {}) => `#/p/${id}${qs(params)}`,
  topic: (id, params = {}) => `#/t/${id}${qs(params)}`,
  watch: (id) => `#/w/${id}`,
  search: (q) => `#/search/${encodeURIComponent(q)}`,
  saved: () => '#/saved',
  about: () => '#/about',
};
const ytSearch = (q) => `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
const thumb = (id) => `https://i.ytimg.com/vi/${id}/mqdefault.jpg`;
const thumbHq = (id) => `https://i.ytimg.com/vi/${id}/hqdefault.jpg`;

// ── Sorting ─────────────────────────────────────────────────────────────
const ratio = (v) => (v.likes ?? 0) / Math.max(1, v.views);
export const SORTS = [
  { id: 'views', label: 'Most viewed', fn: (a, b) => b.views - a.views },
  { id: 'likes', label: 'Most liked', fn: (a, b) => (b.likes ?? 0) - (a.likes ?? 0) },
  { id: 'loved', label: 'Best like ratio', fn: (a, b) => ratio(b) - ratio(a) },
  { id: 'longest', label: 'Longest first', fn: (a, b) => (b.durationSec ?? 0) - (a.durationSec ?? 0) },
  { id: 'newest', label: 'Newest', fn: (a, b) => String(b.publishedAt ?? '').localeCompare(String(a.publishedAt ?? '')) },
];
const sortBy = (list, id) => [...list].sort((SORTS.find((s) => s.id === id) ?? SORTS[0]).fn);

// ── Small parts ─────────────────────────────────────────────────────────
const icon = {
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>',
  ext: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  bookmark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h12v18l-6-4-6 4z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
  bookmarkOn: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h12v18l-6-4-6 4z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

export function avatar(person, size = 'md') {
  return `<span class="avatar avatar--${size} cat-${person.category}" aria-hidden="true">${esc(monogram(person))}</span>`;
}

function topicTag(id) {
  const t = TOPIC_BY_ID.get(id);
  return t ? `<span class="tag">${esc(t.label)}</span>` : '';
}

export function barNote(t) {
  return `<p class="bar-note">${icon.check}<span>Every video: <b>${short(t.minViews)}+ views</b> · <b>${short(t.minLikes)}+ likes</b> · from channels with <b>${short(t.minSubscribers)}+ subscribers</b></span></p>`;
}

export function personCard(person, count) {
  return `<a class="person-card" href="${href.person(person.id)}">
    ${avatar(person)}
    <span class="pc-text">
      <span class="pc-name">${esc(person.name)}</span>
      <span class="pc-years">${esc(person.years)}</span>
      <span class="pc-blurb">${esc(person.blurb)}</span>
      <span class="pc-count ${count ? '' : 'is-zero'}">${count ? plural(count, 'video') : 'No videos yet'}</span>
    </span>
  </a>`;
}

export function videoCard(v, { showPerson = false, compact = false } = {}) {
  const people = v.personIds.map((id) => PEOPLE_BY_ID.get(id)).filter(Boolean);
  const kicker = showPerson && people.length
    ? `<span class="vc-kicker">${esc(people.slice(0, 2).map((p) => p.name).join(' · '))}${people.length > 2 ? ` +${people.length - 2}` : ''}</span>`
    : '';
  return `<a class="video-card${compact ? ' video-card--compact' : ''}" href="${href.watch(v.id)}">
    <span class="thumb"><img src="${thumb(v.id)}" alt="" loading="lazy" decoding="async" width="320" height="180">${v.durationSec ? `<span class="dur">${duration(v.durationSec)}</span>` : ''}</span>
    <span class="vc-body">
      ${kicker}
      <span class="vc-title">${esc(v.title)}</span>
      <span class="vc-channel">${esc(v.channelTitle)}${v.channelSubscribers ? ` · ${short(v.channelSubscribers)} subs` : ''}</span>
      <span class="vc-stats">${short(v.views)} views · ${short(v.likes)} likes${v.publishedAt ? ` · ${year(v.publishedAt)}` : ''}</span>
      ${compact ? '' : `<span class="vc-tags">${v.topics.slice(0, 3).map(topicTag).join('')}</span>`}
    </span>
  </a>`;
}

function videoGrid(list, opts) {
  return `<div class="video-grid">${list.map((v) => videoCard(v, opts)).join('')}</div>`;
}

function chips(items, active) {
  return `<div class="chips" role="list">${items
    .map((i) => `<a role="listitem" class="chip${i.id === active ? ' is-active' : ''}" href="${i.href}"${i.id === active ? ' aria-current="true"' : ''}>${esc(i.label)}${i.count != null ? ` <span class="chip-count">${i.count}</span>` : ''}</a>`)
    .join('')}</div>`;
}

function sortSelect(active) {
  return `<label class="sort">Sort
    <select data-action="sort">${SORTS.map((s) => `<option value="${s.id}"${s.id === active ? ' selected' : ''}>${s.label}</option>`).join('')}</select>
  </label>`;
}

function topicCounts(list) {
  const m = new Map();
  for (const v of list) for (const t of v.topics) m.set(t, (m.get(t) ?? 0) + 1);
  return m;
}

function emptyCatalog(ctx) {
  return `<section class="notice">
    <h2>The video shelf is still empty</h2>
    <p>HumTube’s catalog is built from the YouTube Data API, and it hasn’t been run yet. Two ways to fill it:</p>
    <ol>
      <li><b>Build the full catalog</b> (recommended): <code>YOUTUBE_API_KEY=… npm run build:catalog</code> — searches all 100 people, applies the quality bar and writes <code>data/catalog.json</code>.</li>
      <li><b>Live mode in this browser</b>: ${ctx.hasKey ? 'your API key is saved — open any person and press <i>Find videos on YouTube</i>.' : '<button type="button" class="link-btn" data-action="open-settings">add your API key</button>, then open any person and press <i>Find videos on YouTube</i>.'}</li>
    </ol>
  </section>`;
}

const pageHead = (title, sub = '') => `<header class="page-head"><h1>${title}</h1>${sub ? `<p class="lede">${sub}</p>` : ''}</header>`;

// ── Sidebar ─────────────────────────────────────────────────────────────
export function sidebar(index, route) {
  const [view, arg] = route.parts;
  const is = (v, a) => view === v && (a == null || arg === a);
  const countFor = (catId) => PEOPLE.filter((p) => p.category === catId).reduce((n, p) => n + (index.byPerson.get(p.id)?.length ? 1 : 0), 0);
  const link = (h, label, active, extra = '') => `<a class="nav-link${active ? ' is-active' : ''}" href="${h}"${active ? ' aria-current="page"' : ''}>${label}${extra}</a>`;
  return `
    <div class="nav-group">
      ${link(href.home(), 'Home', !view)}
      ${link(href.saved(), 'Watch later', is('saved'))}
      ${link(href.about(), 'About HumTube', is('about'))}
    </div>
    <div class="nav-group">
      <h2 class="nav-title">TIME 100 categories</h2>
      ${CATEGORIES.map((c) => link(href.category(c.id), `<span class="dot cat-${c.id}"></span>${esc(c.label)}`, is('c', c.id), `<span class="nav-count" title="People with videos">${countFor(c.id)}/20</span>`)).join('')}
    </div>
    <div class="nav-group">
      <h2 class="nav-title">Topics</h2>
      ${ALL_TOPICS.map((t) => link(href.topic(t.id), `<span class="nav-icon" aria-hidden="true">${t.icon}</span>${esc(t.label)}`, is('t', t.id), `<span class="nav-count">${index.byTopic.get(t.id)?.length ?? 0}</span>`)).join('')}
    </div>
    <p class="nav-foot">${index.generatedAt ? `Catalog built ${esc(date(index.generatedAt))}` : 'Catalog not built yet'} · ${plural(index.list.length, 'video')}</p>`;
}

// ── Home ────────────────────────────────────────────────────────────────
export function home(index, ctx) {
  const einstein = PEOPLE_BY_ID.get(PERSON_OF_THE_CENTURY.id);
  const [r1, r2] = PERSON_OF_THE_CENTURY.runnersUp.map((id) => PEOPLE_BY_ID.get(id));
  const hasVideos = index.list.length > 0;
  const top = index.list.slice(0, 8);

  const hero = `<section class="hero">
    <div class="hero-text">
      <p class="eyebrow">TIME’s Person of the Century · ${esc(PERSON_OF_THE_CENTURY.issue)}</p>
      <h1>Learn the 20th century through the 100 people who shaped it.</h1>
      <p class="lede">Documentaries, war history, speeches, films and interviews about TIME’s 100 most important people of the century, drawn from YouTube’s most-watched and most-liked videos.</p>
      <div class="hero-actions">
        <a class="btn btn--primary" href="${href.person(einstein.id)}">${icon.play}Start with ${esc(einstein.name)}</a>
        <a class="btn" href="${href.person(r1.id)}">${esc(r1.name)}</a>
        <a class="btn" href="${href.person(r2.id)}">${esc(r2.name)}</a>
      </div>
      <p class="hero-foot">Einstein was named Person of the Century; Roosevelt and Gandhi were the runners-up.</p>
    </div>
    <a class="hero-card" href="${href.person(einstein.id)}" aria-label="${esc(einstein.name)}">
      ${avatar(einstein, 'xl')}
      <span class="hero-card-name">${esc(einstein.name)}</span>
      <span class="hero-card-years">${esc(einstein.years)}</span>
      <span class="hero-card-count">${plural(index.byPerson.get(einstein.id)?.length ?? 0, 'video')}</span>
    </a>
  </section>`;

  const topics = `<section class="section">
    <div class="section-head"><h2>Browse by topic</h2></div>
    <div class="topic-tiles">${ALL_TOPICS.map((t) => {
      const n = index.byTopic.get(t.id)?.length ?? 0;
      return `<a class="topic-tile${n ? '' : ' is-empty'}" href="${href.topic(t.id)}"><span class="tt-icon" aria-hidden="true">${t.icon}</span><span class="tt-label">${esc(t.label)}</span><span class="tt-count">${plural(n, 'video')}</span></a>`;
    }).join('')}</div>
  </section>`;

  const mostWatched = hasVideos ? `<section class="section">
    <div class="section-head"><h2>Most watched</h2><a class="see-all" href="${href.topic('all')}">See all ${plural(index.list.length, 'video')}</a></div>
    ${videoGrid(top, { showPerson: true })}
  </section>` : '';

  const rows = CATEGORIES.map((c) => {
    const people = PEOPLE.filter((p) => p.category === c.id);
    return `<section class="section">
      <div class="section-head"><h2><span class="dot cat-${c.id}"></span>${esc(c.label)}</h2><a class="see-all" href="${href.category(c.id)}">All 20</a></div>
      <div class="scroller">${people.map((p) => personCard(p, index.byPerson.get(p.id)?.length ?? 0)).join('')}</div>
    </section>`;
  }).join('');

  return `${hero}${barNote(index.thresholds)}${hasVideos ? '' : emptyCatalog(ctx)}${mostWatched}${topics}${rows}`;
}

// ── Category ────────────────────────────────────────────────────────────
export function category(index, catId) {
  const c = CATEGORY_BY_ID.get(catId);
  if (!c) return notFound();
  const people = PEOPLE.filter((p) => p.category === c.id);
  const ids = new Set(people.map((p) => p.id));
  const vids = index.list.filter((v) => v.personIds.some((id) => ids.has(id)));
  return `<header class="page-head page-head--cat cat-${c.id}">
      <p class="eyebrow">TIME 100 · issue of ${esc(c.issue)}</p>
      <h1>${esc(c.label)}</h1>
      <p class="lede">${esc(c.blurb)}</p>
    </header>
    <div class="person-grid">${people.map((p) => personCard(p, index.byPerson.get(p.id)?.length ?? 0)).join('')}</div>
    ${vids.length ? `<section class="section"><div class="section-head"><h2>Most watched in ${esc(c.short)}</h2></div>${videoGrid(vids.slice(0, 24), { showPerson: true })}</section>` : ''}`;
}

// ── Person ──────────────────────────────────────────────────────────────
export function person(index, personId, params, ctx) {
  const p = PEOPLE_BY_ID.get(personId);
  if (!p) return notFound();
  const c = CATEGORY_BY_ID.get(p.category);
  const all = index.byPerson.get(p.id) ?? [];
  const topic = params.get('topic') || 'all';
  const sort = params.get('sort') || 'views';
  const shown = sortBy(topic === 'all' ? all : all.filter((v) => v.topics.includes(topic)), sort);
  const counts = topicCounts(all);
  const report = index.reports[p.id];
  const busy = ctx.busy.has(p.id);
  const err = ctx.errors.get(p.id);
  const liveAt = index.liveFetchedAt[p.id];
  const isPotc = p.id === PERSON_OF_THE_CENTURY.id;

  const liveBtn = ctx.hasKey
    ? `<button type="button" class="btn${all.length ? '' : ' btn--primary'}" data-action="fetch-live" data-person="${p.id}"${busy ? ' disabled aria-busy="true"' : ''}>${icon.refresh}${busy ? 'Searching YouTube…' : all.length ? 'Refresh from YouTube' : 'Find videos on YouTube'}</button>`
    : `<button type="button" class="btn" data-action="open-settings">${icon.refresh}Use live mode…</button>`;

  let status = '';
  if (report) {
    const rej = Object.entries(report.rejected ?? {}).sort((a, b) => b[1] - a[1]);
    status = `<details class="status">
      <summary>${liveAt ? `Live results from ${esc(ago(liveAt))}` : `Searched ${esc(ago(report.searchedAt))}`} · ${plural(report.candidates, 'video')} checked · ${plural(report.accepted, 'passed', 'passed')} the bar</summary>
      <p>Searches: ${report.queries.map((q) => `“${esc(q)}”`).join(', ')}.</p>
      ${rej.length ? `<ul>${rej.map(([k, n]) => `<li><b>${n}</b> ${esc(REJECTION_LABELS[k] ?? k)}</li>`).join('')}</ul>` : ''}
      <p>Videos about ${esc(p.name)} found while searching for other people are included too.</p>
    </details>`;
  }

  const head = `<header class="person-head">
    ${avatar(p, 'xl')}
    <div class="ph-text">
      <p class="eyebrow"><a href="${href.category(c.id)}"><span class="dot cat-${c.id}"></span>${esc(c.label)}</a>${isPotc ? ' · <b>Person of the Century</b>' : ''}</p>
      <h1>${esc(p.name)}</h1>
      <p class="ph-years">${esc(p.years)}</p>
      <p class="lede">${esc(p.blurb)}</p>
      <div class="ph-actions">
        ${liveBtn}
        <a class="btn btn--ghost" href="${ytSearch(queriesFor(p)[0])}" target="_blank" rel="noopener noreferrer">${icon.ext}Search YouTube</a>
      </div>
      ${err ? `<p class="error" role="alert">${esc(err)}</p>` : ''}
      ${status}
    </div>
  </header>`;

  if (!all.length) {
    const why = report
      ? `<p>${plural(report.candidates, 'video')} turned up, but none met HumTube’s bar of ${short(index.thresholds.minViews)}+ views, ${short(index.thresholds.minLikes)}+ likes and a ${short(index.thresholds.minSubscribers)}+ subscriber channel. Lesser-known figures often have excellent videos that are simply less viral — try <a href="${ytSearch(`${p.name} documentary`)}" target="_blank" rel="noopener noreferrer">searching YouTube directly</a>.</p>`
      : `<p>No one has searched YouTube for ${esc(p.name)} yet. ${ctx.hasKey ? 'Press <i>Find videos on YouTube</i> above (about 100 of your 10,000 daily quota units).' : 'Run the catalog builder, or add an API key in settings to search from this page.'}</p>`;
    return `${head}<section class="empty">${why}</section>`;
  }

  const topicItems = [
    { id: 'all', label: 'All', count: all.length, href: href.person(p.id, { sort: sort === 'views' ? null : sort }) },
    ...ALL_TOPICS.filter((t) => counts.get(t.id)).map((t) => ({ id: t.id, label: t.label, count: counts.get(t.id), href: href.person(p.id, { topic: t.id, sort: sort === 'views' ? null : sort }) })),
  ];

  return `${head}
    <div class="toolbar">${chips(topicItems, topic)}${sortSelect(sort)}</div>
    ${shown.length ? videoGrid(shown) : '<p class="empty">No videos in this topic.</p>'}`;
}

// ── Topic (and "all videos") ────────────────────────────────────────────
export function topic(index, topicId, params) {
  const t = topicId === 'all' ? { id: 'all', label: 'All videos', icon: '📚' } : TOPIC_BY_ID.get(topicId);
  if (!t) return notFound();
  const cat = params.get('cat') || 'all';
  const sort = params.get('sort') || 'views';
  const limit = Math.max(60, Number(params.get('n')) || 60);
  const base = t.id === 'all' ? index.list : index.byTopic.get(t.id) ?? [];
  const inCat = (v, catId) => v.personIds.some((id) => PEOPLE_BY_ID.get(id)?.category === catId);
  const filtered = cat === 'all' ? base : base.filter((v) => inCat(v, cat));
  const shown = sortBy(filtered, sort);
  const keep = { sort: sort === 'views' ? null : sort };
  const catItems = [
    { id: 'all', label: 'All categories', count: base.length, href: href.topic(t.id, keep) },
    ...CATEGORIES.map((c) => ({ id: c.id, label: c.short, count: base.filter((v) => inCat(v, c.id)).length, href: href.topic(t.id, { ...keep, cat: c.id }) })),
  ];
  return `${pageHead(`<span class="h-icon" aria-hidden="true">${t.icon}</span>${esc(t.label)}`, plural(filtered.length, 'video'))}
    <div class="toolbar">${chips(catItems, cat)}${sortSelect(sort)}</div>
    ${shown.length ? videoGrid(shown.slice(0, limit), { showPerson: true }) : '<p class="empty">No videos here yet.</p>'}
    ${shown.length > limit ? `<p class="more"><a class="btn" href="${href.topic(t.id, { ...keep, cat: cat === 'all' ? null : cat, n: limit + 60 })}">Show more</a></p>` : ''}`;
}

// ── Watch ───────────────────────────────────────────────────────────────
export function watch(index, videoId, ctx) {
  const v = index.videos.get(videoId);
  if (!v) return notFound('This video isn’t in HumTube’s catalog.');
  const people = v.personIds.map((id) => PEOPLE_BY_ID.get(id)).filter(Boolean);
  const lead = PEOPLE_BY_ID.get(v.foundFor?.[0]) ?? people[0];
  const isSaved = ctx.saved.has(v.id);
  // Sidebar: the lead person's other videos first, then videos of anyone else in this one.
  const seen = new Set([v.id]);
  const take = (list, max) => list.filter((r) => !seen.has(r.id) && seen.add(r.id)).slice(0, max);
  const more = lead ? take(index.byPerson.get(lead.id) ?? [], 10) : [];
  const others = people.filter((p) => p !== lead);
  const related = take(others.flatMap((p) => index.byPerson.get(p.id) ?? []), Math.max(4, 12 - more.length));
  const ytUrl = `https://www.youtube.com/watch?v=${v.id}`;

  const player = v.embeddable
    ? `<div class="player"><iframe src="https://www.youtube-nocookie.com/embed/${v.id}?rel=0" title="${esc(v.title)}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>`
    : `<a class="player player--poster" href="${ytUrl}" target="_blank" rel="noopener noreferrer"><img src="${thumbHq(v.id)}" alt=""><span class="poster-note">${icon.play}The uploader doesn’t allow embedding — watch on YouTube</span></a>`;

  return `<div class="watch">
    <div class="watch-main">
      ${player}
      <h1 class="watch-title">${esc(v.title)}</h1>
      <div class="watch-meta">
        <div class="channel"><span class="channel-avatar" aria-hidden="true">${esc((v.channelTitle || '?').trim()[0] ?? '?')}</span><span><b>${esc(v.channelTitle)}</b><small>${full(v.channelSubscribers)} subscribers</small></span></div>
        <div class="watch-actions">
          <button type="button" class="btn${isSaved ? ' is-on' : ''}" data-action="save" data-video="${v.id}" aria-pressed="${isSaved}">${isSaved ? icon.bookmarkOn : icon.bookmark}<span>${isSaved ? 'Saved' : 'Watch later'}</span></button>
          <a class="btn" href="${ytUrl}" target="_blank" rel="noopener noreferrer">${icon.ext}Open on YouTube</a>
        </div>
      </div>
      <dl class="stat-row">
        <div><dt>Views</dt><dd>${full(v.views)}</dd></div>
        <div><dt>Likes</dt><dd>${full(v.likes)}</dd></div>
        <div><dt>Length</dt><dd>${duration(v.durationSec) || '—'}</dd></div>
        <div><dt>Published</dt><dd>${esc(date(v.publishedAt)) || '—'}</dd></div>
      </dl>
      <div class="watch-people">
        ${people.map((p) => `<a class="person-chip" href="${href.person(p.id)}">${avatar(p, 'sm')}<span>${esc(p.name)}</span></a>`).join('')}
      </div>
      <p class="watch-topics">${v.topics.map((t) => `<a class="tag tag--link" href="${href.topic(t)}">${esc(TOPIC_BY_ID.get(t)?.label ?? t)}</a>`).join('')}</p>
      ${v.description ? `<p class="watch-desc">${esc(v.description)}${v.description.length >= 280 ? '…' : ''}</p>` : ''}
    </div>
    <aside class="watch-side" aria-label="Related videos">
      ${more.length ? `<h2>More about ${esc(lead.name)}</h2>${more.map((r) => videoCard(r, { compact: true })).join('')}` : ''}
      ${related.length ? `<h2>Also featuring ${esc(others.map((p) => p.name).slice(0, 2).join(' & '))}</h2>${related.map((r) => videoCard(r, { compact: true })).join('')}` : ''}
      ${!more.length && !related.length ? `<h2>More videos</h2><p class="empty">Nothing else yet${lead ? ` for ${esc(lead.name)}` : ''}.</p>` : ''}
    </aside>
  </div>`;
}

// ── Search ──────────────────────────────────────────────────────────────
export function search(index, q) {
  const needle = fold(q).trim();
  if (!needle) return pageHead('Search', 'Type a name, a war, a film or a topic.');
  const words = needle.split(' ');
  const hit = (text) => {
    const f = fold(text);
    return words.every((w) => f.includes(` ${w}`));
  };
  const people = PEOPLE.filter((p) => hit([p.name, p.blurb, ...p.match, CATEGORY_BY_ID.get(p.category).label].join(' ')));
  const topics = ALL_TOPICS.filter((t) => hit(t.label));
  const vids = index.list.filter((v) => hit(`${v.title} ${v.channelTitle} ${v.personIds.map((id) => PEOPLE_BY_ID.get(id)?.name).join(' ')}`));
  const nothing = !people.length && !topics.length && !vids.length;
  return `${pageHead(`Results for “${esc(q)}”`, nothing ? 'Nothing matched. Try a last name, like “Churchill”, or a topic, like “war”.' : '')}
    ${topics.length ? `<div class="chips">${topics.map((t) => `<a class="chip" href="${href.topic(t.id)}">${t.icon} ${esc(t.label)}</a>`).join('')}</div>` : ''}
    ${people.length ? `<section class="section"><div class="section-head"><h2>People</h2></div><div class="person-grid">${people.map((p) => personCard(p, index.byPerson.get(p.id)?.length ?? 0)).join('')}</div></section>` : ''}
    ${vids.length ? `<section class="section"><div class="section-head"><h2>${plural(vids.length, 'video')}</h2></div>${videoGrid(vids.slice(0, 60), { showPerson: true })}</section>` : ''}`;
}

// ── Saved ───────────────────────────────────────────────────────────────
export function savedView(index, ctx) {
  const ids = ctx.saved.list();
  const vids = ids.map((id) => index.videos.get(id)).filter(Boolean);
  return `${pageHead('Watch later', vids.length ? plural(vids.length, 'saved video') : 'Videos you save appear here. They’re stored in this browser only.')}
    ${vids.length ? videoGrid(vids, { showPerson: true }) : ''}`;
}

// ── About ───────────────────────────────────────────────────────────────
export function about(index) {
  const t = index.thresholds;
  return `${pageHead('About HumTube', 'A history classroom made of YouTube’s best-loved videos about the 20th century’s most important people.')}
  <div class="prose">
    <h2>The 100 people</h2>
    <p>In 1998 and 1999 TIME named the 100 most important people of the century in five groups of 20, then chose <a href="${href.person(PERSON_OF_THE_CENTURY.id)}">Albert Einstein</a> as Person of the Century. HumTube follows that list exactly (<a href="https://time.com/archive/6735625/time-100-persons-of-the-century/" target="_blank" rel="noopener noreferrer">TIME’s archive</a>).</p>
    ${CATEGORIES.map((c) => `<h3><span class="dot cat-${c.id}"></span>${esc(c.label)} <small>· ${esc(c.issue)}</small></h3>
      <p class="name-list">${PEOPLE.filter((p) => p.category === c.id).map((p) => `<a href="${href.person(p.id)}">${esc(p.name)}</a>`).join(' · ')}</p>`).join('')}
    <h2>Which videos make it in</h2>
    <ul>
      <li>At least <b>${full(t.minViews)} views</b>.</li>
      <li>At least <b>${full(t.minLikes)} likes</b>. Videos with hidden like counts can’t be checked, so they’re left out.</li>
      <li>From a channel with at least <b>${full(t.minSubscribers)} subscribers</b>. Hidden subscriber counts are left out too.</li>
      <li>Longer than ${t.minDurationSec} seconds, so Shorts are skipped.</li>
      <li>Clearly about the person: their name (or a defining term, like “Tank Man”) must be in the title, tags or opening of the description.</li>
      <li>No memes, parodies, fan edits, rap battles or game footage.</li>
    </ul>
    <p>Each video is labeled with topics (war, documentary, movies & drama, speeches, interviews, biography, archival footage, explainers, music, sports) based on its title, channel and YouTube category.</p>
    <h2>Where the data comes from</h2>
    <p>Video details and statistics come from the YouTube Data API. Videos play through YouTube’s privacy-enhanced embed. HumTube is not affiliated with TIME or YouTube. ${index.generatedAt ? `This catalog was built on ${esc(date(index.generatedAt))}.` : 'The catalog has not been built yet.'}</p>
  </div>`;
}

export function notFound(message = 'That page doesn’t exist.') {
  return `${pageHead('Not found', esc(message))}<p><a class="btn" href="${href.home()}">Back to HumTube</a></p>`;
}
