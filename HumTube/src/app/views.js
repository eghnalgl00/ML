// HTML view renderers. Every dynamic string goes through `esc`.

import {
  PEOPLE, PEOPLE_BY_ID, CATEGORIES, CATEGORY_BY_ID, PERSON_OF_THE_CENTURY, monogram, queriesFor,
} from '../core/people.js';
import { ALL_TOPICS, TOPIC_BY_ID } from '../core/topics.js';
import { fold } from '../core/text.js';
import { esc, short, full, duration, date, timeAgo, plural } from './format.js';

// ── Links ───────────────────────────────────────────────────────────────
const qs = (params) => {
  const s = new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== '')).toString();
  return s ? `?${s}` : '';
};
export const href = {
  home: (params = {}) => `#/${qs(params)}`,
  category: (id) => `#/c/${id}`,
  person: (id, params = {}) => `#/p/${id}${qs(params)}`,
  watch: (id) => `#/w/${id}`,
  search: (q) => `#/search/${encodeURIComponent(q)}`,
  saved: () => '#/saved',
  about: () => '#/about',
};
const ytSearch = (q) => `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
const thumb = (id) => `https://i.ytimg.com/vi/${id}/mqdefault.jpg`;
const thumbHq = (id) => `https://i.ytimg.com/vi/${id}/hqdefault.jpg`;

// ── Sorting ─────────────────────────────────────────────────────────────
const by = (key) => (a, b) => String(b[key] ?? '').localeCompare(String(a[key] ?? ''));
export const SORTS = [
  { id: 'views', label: 'Popular', fn: (a, b) => b.views - a.views },
  { id: 'newest', label: 'Latest', fn: by('publishedAt') },
  { id: 'oldest', label: 'Oldest', fn: (a, b) => by('publishedAt')(b, a) },
  { id: 'likes', label: 'Most liked', fn: (a, b) => (b.likes ?? 0) - (a.likes ?? 0) },
];
const sortBy = (list, id) => [...list].sort((SORTS.find((s) => s.id === id) ?? SORTS[0]).fn);

// ── Icons (24px outline) ────────────────────────────────────────────────
const svg = (d, extra = '') => `<svg viewBox="0 0 24 24" aria-hidden="true" class="i"${extra}>${d}</svg>`;
export const icons = {
  home: svg('<path d="M4 10.5 12 4l8 6.5V20h-5.5v-6h-5v6H4z"/>'),
  later: svg('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>'),
  info: svg('<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.6v.4"/>'),
  leaders: svg('<path d="M5.5 21V4M5.5 4.5h11.5l-2.2 4 2.2 4H5.5"/>'),
  artists: svg('<path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.1 0 1.7-.7 1.7-1.6 0-.5-.2-.8-.5-1.2-.3-.3-.5-.7-.5-1.2 0-.9.7-1.6 1.6-1.6h2.1a4.1 4.1 0 0 0 4.1-4.1C20.5 6.9 16.7 3.5 12 3.5z"/><circle cx="7.6" cy="11.5" r=".9"/><circle cx="10" cy="7.6" r=".9"/><circle cx="14.5" cy="7.6" r=".9"/>'),
  builders: svg('<path d="M3.5 20.5h17M6 20.5V8l6-4 6 4v12.5M9.5 10.5h1M13.5 10.5h1M9.5 14h1M13.5 14h1M10.5 20.5v-3h3v3"/>'),
  scientists: svg('<path d="M9 3.5h6M10.3 3.5v5.4L5.2 18.1a1.6 1.6 0 0 0 1.4 2.4h10.8a1.6 1.6 0 0 0 1.4-2.4L13.7 8.9V3.5M7.4 14.5h9.2"/>'),
  heroes: svg('<path d="m12 3.8 2.5 5.1 5.6.8-4 3.9.9 5.6-5-2.6-5 2.6.9-5.6-4-3.9 5.6-.8z"/>'),
  save: svg('<path d="M6.5 3.5h11v17l-5.5-3.6-5.5 3.6z"/>'),
  saved: svg('<path d="M6.5 3.5h11v17l-5.5-3.6-5.5 3.6z" fill="currentColor"/>'),
  ext: svg('<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
  refresh: svg('<path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6"/>'),
  play: svg('<path d="M8.5 6v12l9.5-6z" fill="currentColor"/>'),
};

// ── Shared parts ────────────────────────────────────────────────────────
export function avatar(person, size = 'md') {
  return `<span class="avatar avatar--${size} cat-${person.category}" aria-hidden="true">${esc(monogram(person))}</span>`;
}

const leadPerson = (v) => PEOPLE_BY_ID.get(v.foundFor?.[0]) ?? PEOPLE_BY_ID.get(v.personIds[0]);
const metaLine = (v) => `${short(v.views)} views${v.publishedAt ? ` · ${timeAgo(v.publishedAt)}` : ''}`;
const videoCount = (index, id) => index.byPerson.get(id)?.length ?? 0;

function thumbLink(v, cls = '') {
  return `<a class="thumb ${cls}" href="${href.watch(v.id)}" tabindex="-1" aria-hidden="true"><img src="${thumb(v.id)}" alt="" loading="lazy" decoding="async" width="320" height="180">${v.durationSec ? `<span class="dur">${duration(v.durationSec)}</span>` : ''}</a>`;
}

/** Grid card: thumbnail, then avatar + title + channel + views · age. */
export function videoCard(v) {
  const p = leadPerson(v);
  return `<div class="vcard">
    ${thumbLink(v)}
    <div class="vcard-meta">
      ${p ? `<a class="vcard-avatar" href="${href.person(p.id)}" title="${esc(p.name)}">${avatar(p)}</a>` : ''}
      <div class="vtext">
        <h3 class="vtitle"><a href="${href.watch(v.id)}" title="${esc(v.title)}">${esc(v.title)}</a></h3>
        <p class="vline">${esc(v.channelTitle)}</p>
        <p class="vline">${metaLine(v)}</p>
      </div>
    </div>
  </div>`;
}

/** Compact row for the watch page's side column. */
function videoRow(v) {
  return `<div class="vrow">
    ${thumbLink(v, 'thumb--sm')}
    <div class="vtext">
      <h3 class="vtitle"><a href="${href.watch(v.id)}" title="${esc(v.title)}">${esc(v.title)}</a></h3>
      <p class="vline">${esc(v.channelTitle)}</p>
      <p class="vline">${metaLine(v)}</p>
    </div>
  </div>`;
}

/** Wide row for search results and Watch later. */
function videoResult(v) {
  const p = leadPerson(v);
  return `<div class="vresult">
    ${thumbLink(v, 'thumb--lg')}
    <div class="vtext">
      <h3 class="vtitle vtitle--lg"><a href="${href.watch(v.id)}">${esc(v.title)}</a></h3>
      <p class="vline">${metaLine(v)}</p>
      <p class="vline vline--channel">${p ? `<a href="${href.person(p.id)}">${avatar(p, 'xs')}</a>` : ''}${esc(v.channelTitle)}</p>
      ${v.description ? `<p class="vline vdesc">${esc(v.description)}</p>` : ''}
    </div>
  </div>`;
}

function personTile(index, p) {
  const n = videoCount(index, p.id);
  return `<a class="ptile" href="${href.person(p.id)}">
    ${avatar(p, 'lg')}
    <span class="ptile-name">${esc(p.name)}</span>
    <span class="ptile-meta">${n ? plural(n, 'video') : 'No videos'}</span>
  </a>`;
}

function personResult(index, p) {
  const c = CATEGORY_BY_ID.get(p.category);
  return `<a class="presult" href="${href.person(p.id)}">
    <span class="presult-avatar">${avatar(p, 'xl')}</span>
    <span class="vtext">
      <span class="vtitle vtitle--lg">${esc(p.name)}</span>
      <span class="vline">${esc(c.label)} · ${esc(p.years)} · ${plural(videoCount(index, p.id), 'video')}</span>
      <span class="vline">${esc(p.blurb)}</span>
    </span>
  </a>`;
}

// Long grids render in pages; main.js appends the rest as you scroll.
export const PAGE = 48;
export const pager = { queue: [], render: videoCard };

function pagedGrid(list, render = videoCard, cls = 'vgrid') {
  pager.queue = list.slice(PAGE);
  pager.render = render;
  return `<div class="${cls}" data-pager>${list.slice(0, PAGE).map(render).join('')}</div>${pager.queue.length ? '<div class="pager-sentinel"></div>' : ''}`;
}

function chipBar(items, active, { sticky = false, extra = '' } = {}) {
  return `<div class="chipbar${sticky ? ' chipbar--sticky' : ''}">
    <div class="chips">${items.map((i) => `<a class="chip${i.id === active ? ' is-active' : ''}" href="${i.href}"${i.id === active ? ' aria-current="true"' : ''}>${esc(i.label)}</a>`).join('')}</div>
    ${extra}
  </div>`;
}

function sortSelect(active) {
  return `<label class="sort"><span class="visually-hidden">Sort by</span>
    <select data-action="sort">${SORTS.map((s) => `<option value="${s.id}"${s.id === active ? ' selected' : ''}>${s.label}</option>`).join('')}</select>
  </label>`;
}

function topicsIn(list) {
  const counts = new Map();
  for (const v of list) for (const t of v.topics) counts.set(t, (counts.get(t) ?? 0) + 1);
  return ALL_TOPICS.filter((t) => counts.get(t.id));
}

// ── Sidebar ─────────────────────────────────────────────────────────────
export function sidebar(index, route) {
  const [view, arg] = route.parts;
  const item = (h, icon, label, active) =>
    `<a class="guide-item${active ? ' is-active' : ''}" href="${h}"${active ? ' aria-current="page"' : ''}>${icon}<span>${esc(label)}</span></a>`;
  return `
    <div class="guide-section">
      ${item(href.home(), icons.home, 'Home', !view || view === 't')}
      ${item(href.saved(), icons.later, 'Watch later', view === 'saved')}
    </div>
    <div class="guide-section">
      <h2 class="guide-title">Categories</h2>
      ${CATEGORIES.map((c) => item(href.category(c.id), icons[c.id], c.short, view === 'c' && arg === c.id)).join('')}
    </div>
    <div class="guide-section">
      ${item(href.about(), icons.info, 'About', view === 'about')}
    </div>
    <p class="guide-foot">Based on TIME’s 100 Persons of the Century. Not affiliated with TIME or YouTube.</p>`;
}

/**
 * Feed order: everyone's most-watched video first, then everyone's second, and
 * so on, so a few very popular people (or music videos) can't fill the page.
 */
export function mixByPerson(list) {
  const groups = new Map();
  for (const v of list) {
    const id = leadPerson(v)?.id ?? '';
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(v);
  }
  const queues = [...groups.values()].map((q) => [...q].sort((a, b) => b.views - a.views));
  queues.sort((a, b) => b[0].views - a[0].views);
  const out = [];
  const rounds = Math.max(0, ...queues.map((q) => q.length));
  for (let i = 0; i < rounds; i += 1) for (const q of queues) if (q[i]) out.push(q[i]);
  return out;
}

// ── Home (and topic pages, which are home with a chip selected) ─────────
export function home(index, topicId, ctx) {
  if (!index.list.length) return emptyHome(index, ctx);
  const topic = TOPIC_BY_ID.has(topicId) ? topicId : 'all';
  const list = topic === 'all' ? index.list : index.byTopic.get(topic) ?? [];
  const items = [
    { id: 'all', label: 'All', href: href.home() },
    ...topicsIn(index.list).map((t) => ({ id: t.id, label: t.label, href: href.home({ topic: t.id }) })),
  ];
  return `${chipBar(items, topic, { sticky: true })}${pagedGrid(mixByPerson(list))}`;
}

function emptyHome(index, ctx) {
  return `<section class="message">
      <h1>No videos yet</h1>
      <p>Videos are added by running the catalog builder with a YouTube Data API key: <code>npm run build:catalog</code>.
      ${ctx.hasKey ? 'Your key is saved, so you can also open a person below and load their videos.' : 'You can also <button type="button" class="link" data-action="open-settings">add a key in Settings</button> and load videos one person at a time.'}</p>
    </section>
    ${CATEGORIES.map((c) => `<section class="shelf">
      <h2 class="shelf-title"><a href="${href.category(c.id)}">${esc(c.label)}</a></h2>
      <div class="ptile-row">${PEOPLE.filter((p) => p.category === c.id).map((p) => personTile(index, p)).join('')}</div>
    </section>`).join('')}`;
}

// ── Category ────────────────────────────────────────────────────────────
export function category(index, catId) {
  const c = CATEGORY_BY_ID.get(catId);
  if (!c) return notFound();
  return `<header class="page-title">
      <h1>${esc(c.label)}</h1>
      <p class="vline">20 people · TIME, ${esc(c.issue)}</p>
    </header>
    <div class="ptile-grid">${PEOPLE.filter((p) => p.category === c.id).map((p) => personTile(index, p)).join('')}</div>`;
}

// ── Person (laid out like a channel page) ───────────────────────────────
export function person(index, personId, params, ctx) {
  const p = PEOPLE_BY_ID.get(personId);
  if (!p) return notFound();
  const c = CATEGORY_BY_ID.get(p.category);
  const all = index.byPerson.get(p.id) ?? [];
  const topic = params.get('topic') || 'all';
  const sort = params.get('sort') || 'views';
  const list = sortBy(topic === 'all' ? all : all.filter((v) => v.topics.includes(topic)), sort);
  const busy = ctx.busy.has(p.id);
  const err = ctx.errors.get(p.id);

  const buttons = [];
  if (ctx.hasKey) {
    buttons.push(`<button type="button" class="pill${all.length ? '' : ' pill--dark'}" data-action="fetch-live" data-person="${p.id}"${busy ? ' disabled aria-busy="true"' : ''}>${icons.refresh}<span>${busy ? 'Loading…' : all.length ? 'Refresh' : 'Load videos'}</span></button>`);
  }
  buttons.push(`<a class="pill" href="${ytSearch(queriesFor(p)[0])}" target="_blank" rel="noopener noreferrer">${icons.ext}<span>Search on YouTube</span></a>`);

  const head = `<header class="channel-head">
    ${avatar(p, 'xxl')}
    <div class="channel-info">
      <h1>${esc(p.name)}</h1>
      <p class="vline"><a href="${href.category(c.id)}">${esc(c.label)}</a> · ${esc(p.years)} · ${plural(all.length, 'video')}</p>
      <p class="channel-desc">${p.id === PERSON_OF_THE_CENTURY.id ? 'TIME’s Person of the Century. ' : ''}${esc(p.blurb)}</p>
      <div class="channel-actions">${buttons.join('')}</div>
      ${err ? `<p class="error" role="alert">${esc(err)}</p>` : ''}
    </div>
  </header>`;

  if (!all.length) {
    const t = index.thresholds;
    const report = index.reports[p.id];
    const msg = report
      ? `None of the ${plural(report.candidates, 'video')} found had at least ${short(t.minViews)} views and ${short(t.minLikes)} likes from a channel with ${short(t.minSubscribers)}+ subscribers.`
      : ctx.hasKey ? 'Not searched yet. Use “Load videos” above.' : 'Not searched yet.';
    return `${head}<div class="tabs"><span class="tab is-active">Videos</span></div><p class="message message--inline">${msg}</p>`;
  }

  const keep = { sort: sort === 'views' ? null : sort };
  const items = [
    { id: 'all', label: 'All', href: href.person(p.id, keep) },
    ...topicsIn(all).map((t) => ({ id: t.id, label: t.label, href: href.person(p.id, { topic: t.id, ...keep }) })),
  ];
  return `${head}
    <div class="tabs"><span class="tab is-active">Videos</span></div>
    ${chipBar(items, topic, { extra: sortSelect(sort) })}
    ${list.length ? pagedGrid(list) : '<p class="message message--inline">No videos.</p>'}`;
}

// ── Watch ───────────────────────────────────────────────────────────────
export function watch(index, videoId, ctx) {
  const v = index.videos.get(videoId);
  if (!v) return notFound('This video isn’t on HumTube.');
  const people = v.personIds.map((id) => PEOPLE_BY_ID.get(id)).filter(Boolean);
  const lead = leadPerson(v);
  const isSaved = ctx.saved.has(v.id);

  // Up next: the lead person's other videos, then anyone else in this one.
  const seen = new Set([v.id]);
  const take = (list) => list.filter((r) => !seen.has(r.id) && seen.add(r.id));
  const upNext = [
    ...take(lead ? index.byPerson.get(lead.id) ?? [] : []).slice(0, 12),
    ...take(people.filter((p) => p !== lead).flatMap((p) => index.byPerson.get(p.id) ?? [])).slice(0, 8),
  ];
  const ytUrl = `https://www.youtube.com/watch?v=${v.id}`;

  const player = v.embeddable
    ? `<div class="player"><iframe src="https://www.youtube-nocookie.com/embed/${v.id}?rel=0" title="${esc(v.title)}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>`
    : `<a class="player player--poster" href="${ytUrl}" target="_blank" rel="noopener noreferrer"><img src="${thumbHq(v.id)}" alt=""><span class="poster-note">${icons.play}Watch on YouTube</span></a>`;

  return `<div class="watch">
    <div class="watch-main">
      ${player}
      <h1 class="watch-title">${esc(v.title)}</h1>
      <div class="watch-bar">
        <div class="owner">
          <span class="owner-avatar" aria-hidden="true">${esc((v.channelTitle || '?').trim()[0] ?? '?')}</span>
          <span class="owner-text"><b>${esc(v.channelTitle)}</b><span>${v.channelSubscribers ? `${short(v.channelSubscribers)} subscribers` : ''}</span></span>
        </div>
        <div class="watch-actions">
          <span class="pill pill--static" title="${full(v.likes)} likes">${svg('<path d="M7 10v10H4V10zM7 10l4-7c1.4 0 2.3 1.2 2 2.5L12.5 9H18a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 16.8 20H7"/>')}<span>${short(v.likes)}</span></span>
          <button type="button" class="pill" data-action="save" data-video="${v.id}" aria-pressed="${isSaved}">${isSaved ? icons.saved : icons.save}<span>${isSaved ? 'Saved' : 'Save'}</span></button>
          <a class="pill" href="${ytUrl}" target="_blank" rel="noopener noreferrer">${icons.ext}<span>YouTube</span></a>
        </div>
      </div>
      <div class="description">
        <p class="description-head"><span>${full(v.views)} views</span><span>${esc(date(v.publishedAt))}</span>${v.topics.filter((t) => t !== 'other').map((t) => `<a href="${href.home({ topic: t })}">#${esc(TOPIC_BY_ID.get(t)?.label.replace(/\s+/g, '') ?? t)}</a>`).join('')}</p>
        ${v.description ? `<p class="description-body">${esc(v.description)}${v.description.length >= 280 ? '…' : ''}</p>` : ''}
        <div class="description-people">
          ${people.map((p) => `<a class="person-link" href="${href.person(p.id)}">${avatar(p)}<span class="person-link-text"><b>${esc(p.name)}</b><span>${esc(CATEGORY_BY_ID.get(p.category).label)}</span></span></a>`).join('')}
        </div>
      </div>
    </div>
    <aside class="watch-side" aria-label="Up next">
      ${upNext.length ? upNext.map(videoRow).join('') : '<p class="vline">No other videos yet.</p>'}
    </aside>
  </div>`;
}

// ── Search ──────────────────────────────────────────────────────────────
export function search(index, q) {
  const needle = fold(q).trim();
  if (!needle) return `<p class="message message--inline">Type a name, a war or a film.</p>`;
  const words = needle.split(' ');
  const hit = (text) => {
    const f = fold(text);
    return words.every((w) => f.includes(` ${w}`));
  };
  const people = PEOPLE.filter((p) => hit([p.name, ...p.match, CATEGORY_BY_ID.get(p.category).label].join(' ')));
  const vids = index.list.filter((v) => hit(`${v.title} ${v.channelTitle} ${v.personIds.map((id) => PEOPLE_BY_ID.get(id)?.name).join(' ')}`));
  if (!people.length && !vids.length) return `<p class="message message--inline">No results for “${esc(q)}”.</p>`;
  return `<div class="results">
    ${people.map((p) => personResult(index, p)).join('')}
    ${vids.length ? pagedGrid(vids, videoResult, 'results-list') : ''}
  </div>`;
}

// ── Watch later ─────────────────────────────────────────────────────────
export function savedView(index, ctx) {
  const vids = ctx.saved.list().map((id) => index.videos.get(id)).filter(Boolean);
  return `<header class="page-title"><h1>Watch later</h1><p class="vline">${plural(vids.length, 'video')}</p></header>
    ${vids.length ? `<div class="results">${vids.map(videoResult).join('')}</div>` : '<p class="message message--inline">Videos you save will show up here.</p>'}`;
}

// ── About ───────────────────────────────────────────────────────────────
export function about(index) {
  const t = index.thresholds;
  return `<article class="prose">
    <h1>About HumTube</h1>
    <p>HumTube collects YouTube videos about the people on TIME’s list of the 100 most important people of the 20th century, published in 1998 and 1999. TIME named Albert Einstein Person of the Century, with Franklin D. Roosevelt and Mohandas Gandhi as runners-up. <a href="https://time.com/archive/6735625/time-100-persons-of-the-century/" target="_blank" rel="noopener noreferrer">TIME’s list</a></p>
    <h2>Which videos are included</h2>
    <ul>
      <li>${full(t.minViews)} or more views</li>
      <li>${full(t.minLikes)} or more likes (videos with hidden likes are left out)</li>
      <li>From a channel with ${full(t.minSubscribers)} or more subscribers (hidden counts are left out)</li>
      <li>Longer than ${t.minDurationSec} seconds</li>
      <li>The person’s name is in the title, tags or the start of the description</li>
      <li>No memes, parodies, fan edits or game videos</li>
    </ul>
    <h2>The 100 people</h2>
    ${CATEGORIES.map((c) => `<h3>${esc(c.label)}</h3>
      <p>${PEOPLE.filter((p) => p.category === c.id).map((p) => `<a href="${href.person(p.id)}">${esc(p.name)}</a>`).join(', ')}</p>`).join('')}
    <h2>Data</h2>
    <p>Video information comes from the YouTube Data API${index.generatedAt ? ` and was last updated ${esc(date(index.generatedAt))}` : ''}. Videos play in YouTube’s privacy-enhanced player. HumTube is not affiliated with TIME or YouTube.</p>
  </article>`;
}

export function notFound(message = 'This page isn’t available.') {
  return `<section class="message"><h1>${esc(message)}</h1><p><a class="pill" href="${href.home()}">Go to home</a></p></section>`;
}
