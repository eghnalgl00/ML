import { PEOPLE_BY_ID, CATEGORY_BY_ID } from '../core/people.js';
import { TOPIC_BY_ID } from '../core/topics.js';
import { createClient, YouTubeError } from '../core/youtube.js';
import { fetchPersonLive } from '../core/catalog.js';
import { loadBaseCatalog, buildIndex, settings, saved, live } from './state.js';
import * as V from './views.js';

// ?catalog=path/to/file.json lets you preview another catalog (same-origin, relative only).
const param = new URLSearchParams(location.search).get('catalog');
const CATALOG_URL = param && /^[\w./-]+\.json$/.test(param) && !param.includes('..') && !param.startsWith('/')
  ? param
  : 'data/catalog.json';

const els = {
  main: document.getElementById('main'),
  nav: document.getElementById('sidebar'),
  search: document.getElementById('search-form'),
  query: document.getElementById('search-input'),
  menu: document.getElementById('menu-btn'),
  settingsBtn: document.getElementById('settings-btn'),
  dialog: document.getElementById('settings'),
  keyInput: document.getElementById('api-key'),
  keyStatus: document.getElementById('key-status'),
};

let base = null;
let index = null;
let loadError = null;
let lastPath = null;
const sessionFragments = new Map(); // live results kept in memory if localStorage is unavailable
const ctx = {
  busy: new Set(),
  errors: new Map(),
  saved,
  get hasKey() {
    return Boolean(settings.apiKey);
  },
};

function reindex() {
  index = buildIndex(base, [...sessionFragments.values()]);
}

function route() {
  const raw = location.hash.replace(/^#/, '') || '/';
  const [path, query = ''] = raw.split('?');
  let parts;
  try {
    parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  } catch {
    parts = ['404'];
  }
  return { path, parts, params: new URLSearchParams(query) };
}

function titleFor(r) {
  const [view, arg] = r.parts;
  const name =
    view === 'p' ? PEOPLE_BY_ID.get(arg)?.name
      : view === 'c' ? CATEGORY_BY_ID.get(arg)?.label
        : view === 't' ? (arg === 'all' ? 'All videos' : TOPIC_BY_ID.get(arg)?.label)
          : view === 'w' ? index.videos.get(arg)?.title
            : view === 'search' ? `Search: ${arg ?? ''}`
              : view === 'saved' ? 'Watch later'
                : view === 'about' ? 'About'
                  : null;
  return name ? `${name} · HumTube` : 'HumTube — the 20th century’s 100 most important people, on video';
}

function render() {
  if (!index) return; // a hash change during boot; the boot render will catch up
  const r = route();
  const [view, arg] = r.parts;
  let html;
  switch (view) {
    case undefined: html = V.home(index, ctx); break;
    case 'c': html = V.category(index, arg); break;
    case 'p': html = V.person(index, arg, r.params, ctx); break;
    case 't': html = V.topic(index, arg, r.params); break;
    case 'w': html = V.watch(index, arg, ctx); break;
    case 'search': html = V.search(index, arg ?? ''); els.query.value = arg ?? ''; break;
    case 'saved': html = V.savedView(index, ctx); break;
    case 'about': html = V.about(index); break;
    default: html = V.notFound();
  }
  if (view !== 'search' && document.activeElement !== els.query) els.query.value = '';
  if (loadError && !view) {
    html = `<p class="error" role="alert">Couldn’t load the catalog (${loadError}). Showing people only.</p>${html}`;
  }
  els.main.innerHTML = html;
  els.nav.innerHTML = V.sidebar(index, r);
  document.title = titleFor(r);
  document.body.classList.remove('nav-open');
  els.menu.setAttribute('aria-expanded', 'false');
  if (r.path !== lastPath) {
    window.scrollTo(0, 0);
    if (lastPath !== null) els.main.focus({ preventScroll: true });
    lastPath = r.path;
  }
}

function setParam(key, value) {
  const r = route();
  const params = r.params;
  if (value == null || value === '' || (key === 'sort' && value === 'views')) params.delete(key);
  else params.set(key, value);
  params.delete('n');
  const s = params.toString();
  location.hash = `${r.path}${s ? `?${s}` : ''}`;
}

function friendlyError(err) {
  if (err instanceof YouTubeError) {
    if (err.isQuota) return 'YouTube’s daily quota for this API key is used up. It resets at midnight Pacific time.';
    if (err.isAuth || err.status === 400) return `YouTube rejected the API key: ${err.message}`;
    if (err.reason === 'network') return 'Couldn’t reach YouTube. Check your connection and try again.';
    return `YouTube error: ${err.message}`;
  }
  return `Something went wrong: ${err.message}`;
}

async function fetchLive(personId) {
  const person = PEOPLE_BY_ID.get(personId);
  if (!person || ctx.busy.has(personId)) return;
  if (!settings.apiKey) {
    openSettings();
    return;
  }
  ctx.busy.add(personId);
  ctx.errors.delete(personId);
  render();
  try {
    const client = createClient({ apiKey: settings.apiKey });
    const fragment = await fetchPersonLive(person, { client, thresholds: index.thresholds });
    if (!live.put(personId, fragment)) sessionFragments.set(personId, fragment);
    reindex();
  } catch (err) {
    ctx.errors.set(personId, friendlyError(err));
  } finally {
    ctx.busy.delete(personId);
    render();
  }
}

// ── Settings dialog ─────────────────────────────────────────────────────
function updateKeyStatus() {
  const k = settings.apiKey;
  els.keyStatus.textContent = k ? `Saved key ending in …${k.slice(-4)}` : 'No key saved.';
}
function openSettings() {
  els.keyInput.value = '';
  updateKeyStatus();
  if (typeof els.dialog.showModal === 'function') els.dialog.showModal();
  else els.dialog.setAttribute('open', '');
  els.keyInput.focus();
}
els.dialog.addEventListener('click', (e) => {
  const action = e.target.closest('[data-dialog]')?.dataset.dialog;
  if (e.target === els.dialog || action === 'close') els.dialog.close();
  if (action === 'save') {
    const v = els.keyInput.value.trim();
    if (v) settings.apiKey = v;
    updateKeyStatus();
    els.dialog.close();
    render();
  }
  if (action === 'forget') {
    settings.apiKey = '';
    updateKeyStatus();
    render();
  }
  if (action === 'clear-live') {
    live.clear();
    sessionFragments.clear();
    reindex();
    els.keyStatus.textContent = 'Live results cleared.';
    render();
  }
});
els.dialog.addEventListener('submit', (e) => e.preventDefault());

// ── Global events ───────────────────────────────────────────────────────
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const action = el.dataset.action;
  if (action === 'fetch-live') fetchLive(el.dataset.person);
  if (action === 'open-settings') openSettings();
  if (action === 'save') {
    // Update in place so a playing video isn't reloaded.
    const on = saved.toggle(el.dataset.video);
    el.classList.toggle('is-on', on);
    el.setAttribute('aria-pressed', String(on));
    el.querySelector('span').textContent = on ? 'Saved' : 'Watch later';
    el.querySelector('svg path').setAttribute('fill', on ? 'currentColor' : 'none');
  }
});
document.addEventListener('change', (e) => {
  if (e.target.matches('select[data-action="sort"]')) setParam('sort', e.target.value);
});
// Broken thumbnails (removed videos, offline) fall back to a styled placeholder.
document.addEventListener('error', (e) => {
  if (e.target.tagName === 'IMG' && e.target.closest('.thumb, .player--poster')) {
    e.target.closest('.thumb, .player--poster').classList.add('is-missing');
  }
}, true);
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && !e.target.closest('input, textarea, select, [contenteditable]')) {
    e.preventDefault();
    els.query.focus();
    els.query.select();
  }
  if (e.key === 'Escape' && document.body.classList.contains('nav-open')) {
    document.body.classList.remove('nav-open');
    els.menu.setAttribute('aria-expanded', 'false');
  }
});
els.search.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = els.query.value.trim();
  if (q) location.hash = V.href.search(q);
});
els.menu.addEventListener('click', () => {
  const open = document.body.classList.toggle('nav-open');
  els.menu.setAttribute('aria-expanded', String(open));
});
document.getElementById('scrim').addEventListener('click', () => {
  document.body.classList.remove('nav-open');
  els.menu.setAttribute('aria-expanded', 'false');
});
els.settingsBtn.addEventListener('click', openSettings);
window.addEventListener('hashchange', render);

// ── Boot ────────────────────────────────────────────────────────────────
const loaded = await loadBaseCatalog(CATALOG_URL);
base = loaded.catalog;
loadError = loaded.error;
reindex();
render();
document.body.classList.add('is-ready');
