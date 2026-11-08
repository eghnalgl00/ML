import { PEOPLE_BY_ID, CATEGORY_BY_ID } from '../core/people.js';
import { createClient, YouTubeError } from '../core/youtube.js';
import { fetchPersonLive } from '../core/catalog.js';
import { loadBaseCatalog, buildIndex, settings, saved, live } from './state.js';
import * as V from './views.js';

// ?catalog=path/to/file.json lets you preview another catalog (same-origin, relative only).
const param = new URLSearchParams(location.search).get('catalog');
const CATALOG_URL = param && /^[\w./-]+\.json$/.test(param) && !param.includes('..') && !param.startsWith('/')
  ? param
  : 'data/catalog.json';

const $ = (id) => document.getElementById(id);
const els = {
  main: $('main'),
  nav: $('sidebar'),
  search: $('search-form'),
  query: $('search-input'),
  menu: $('menu-btn'),
  dialog: $('settings'),
  keyInput: $('api-key'),
  keyStatus: $('key-status'),
};
const narrow = window.matchMedia('(max-width: 1100px)');

let base = null;
let index = null;
let loadError = null;
let lastPath = null;
let observer = null;
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
        : view === 'w' ? index.videos.get(arg)?.title
          : view === 'search' ? arg
            : view === 'saved' ? 'Watch later'
              : view === 'about' ? 'About'
                : null;
  return name ? `${name} - HumTube` : 'HumTube';
}

function render() {
  if (!index) return; // a hash change during boot; the boot render will catch up
  const r = route();
  const [view, arg] = r.parts;
  V.pager.queue = []; // views that page their grid refill this
  let html;
  switch (view) {
    case undefined: html = V.home(index, r.params.get('topic'), ctx); break;
    case 't': html = V.home(index, arg, ctx); break; // old topic links
    case 'c': html = V.category(index, arg); break;
    case 'p': html = V.person(index, arg, r.params, ctx); break;
    case 'w': html = V.watch(index, arg, ctx); break;
    case 'search': html = V.search(index, arg ?? ''); break;
    case 'saved': html = V.savedView(index, ctx); break;
    case 'about': html = V.about(index); break;
    default: html = V.notFound();
  }
  if (loadError && !view) html = `<p class="error" role="alert">Couldn’t load videos (${loadError}).</p>${html}`;

  els.main.innerHTML = html;
  els.nav.innerHTML = V.sidebar(index, r);
  if (view === 'search') els.query.value = arg ?? '';
  else if (document.activeElement !== els.query) els.query.value = '';
  document.title = titleFor(r);
  if (document.body.classList.contains('is-watch') !== (view === 'w')) {
    document.body.classList.add('no-anim');
    document.body.classList.toggle('is-watch', view === 'w');
    requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove('no-anim')));
  }
  closeDrawer();
  closeSearch();
  setupPager();
  if (r.path !== lastPath) {
    window.scrollTo(0, 0);
    if (lastPath !== null) els.main.focus({ preventScroll: true });
    lastPath = r.path;
  }
}

// Long grids: append the next page when the sentinel scrolls into view.
function setupPager() {
  observer?.disconnect();
  const sentinel = els.main.querySelector('.pager-sentinel');
  const grid = els.main.querySelector('[data-pager]');
  if (!sentinel || !grid || !('IntersectionObserver' in window)) return;
  observer = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    const next = V.pager.queue.splice(0, V.PAGE);
    grid.insertAdjacentHTML('beforeend', next.map(V.pager.render).join(''));
    if (!V.pager.queue.length) {
      observer.disconnect();
      sentinel.remove();
    } else {
      // Re-observe so we get called again if the sentinel is still on screen.
      observer.unobserve(sentinel);
      observer.observe(sentinel);
    }
  }, { rootMargin: '1200px 0px' });
  observer.observe(sentinel);
}

function setParam(key, value) {
  const r = route();
  const params = r.params;
  if (value == null || value === '' || (key === 'sort' && value === 'views')) params.delete(key);
  else params.set(key, value);
  const s = params.toString();
  location.hash = `${r.path}${s ? `?${s}` : ''}`;
}

function friendlyError(err) {
  if (err instanceof YouTubeError) {
    if (err.isQuota) return 'The daily YouTube quota for this key is used up. It resets at midnight Pacific time.';
    if (err.isAuth || err.status === 400) return `YouTube didn’t accept the API key: ${err.message}`;
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

// ── Guide (sidebar): collapses on wide screens, slides in on narrow ones ─
const drawerMode = () => narrow.matches || document.body.classList.contains('is-watch');
function syncMenuState() {
  const open = drawerMode()
    ? document.body.classList.contains('drawer-open')
    : !document.body.classList.contains('guide-hidden');
  els.menu.setAttribute('aria-expanded', String(open));
}
function closeDrawer() {
  document.body.classList.remove('drawer-open');
  syncMenuState();
}
els.menu.addEventListener('click', () => {
  if (drawerMode()) document.body.classList.toggle('drawer-open');
  else document.body.classList.toggle('guide-hidden');
  syncMenuState();
});
narrow.addEventListener('change', () => {
  document.body.classList.add('no-anim');
  closeDrawer();
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove('no-anim')));
});
$('scrim').addEventListener('click', closeDrawer);

// ── Search (full-width overlay on phones) ───────────────────────────────
function closeSearch() {
  document.body.classList.remove('search-active');
}
$('search-open').addEventListener('click', () => {
  document.body.classList.add('search-active');
  els.query.focus();
});
$('search-back').addEventListener('click', closeSearch);
els.search.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = els.query.value.trim();
  if (q) location.hash = V.href.search(q);
  els.query.blur();
});

// ── Settings dialog ─────────────────────────────────────────────────────
function updateKeyStatus() {
  const k = settings.apiKey;
  els.keyStatus.textContent = k ? `Saved key ending in ${k.slice(-4)}.` : 'No key saved.';
}
function openSettings() {
  els.keyInput.value = '';
  updateKeyStatus();
  if (typeof els.dialog.showModal === 'function') els.dialog.showModal();
  else els.dialog.setAttribute('open', '');
  els.keyInput.focus();
}
$('settings-btn').addEventListener('click', openSettings);
els.dialog.addEventListener('click', (e) => {
  const action = e.target.closest('[data-dialog]')?.dataset.dialog;
  if (e.target === els.dialog || action === 'close') els.dialog.close();
  if (action === 'save') {
    const v = els.keyInput.value.trim();
    if (v) settings.apiKey = v;
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
    els.keyStatus.textContent = 'Loaded videos cleared.';
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
    el.setAttribute('aria-pressed', String(on));
    el.innerHTML = `${on ? V.icons.saved : V.icons.save}<span>${on ? 'Saved' : 'Save'}</span>`;
  }
});
document.addEventListener('change', (e) => {
  if (e.target.matches('select[data-action="sort"]')) setParam('sort', e.target.value);
});
// Broken thumbnails (removed videos, offline) fall back to a plain placeholder.
document.addEventListener('error', (e) => {
  const box = e.target.tagName === 'IMG' && e.target.closest('.thumb, .player--poster');
  if (box) box.classList.add('is-missing');
}, true);
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && !e.target.closest('input, textarea, select, [contenteditable]')) {
    e.preventDefault();
    if (narrow.matches) document.body.classList.add('search-active');
    els.query.focus();
    els.query.select();
  }
  if (e.key === 'Escape') {
    closeDrawer();
    closeSearch();
  }
});
window.addEventListener('hashchange', render);

// ── Boot ────────────────────────────────────────────────────────────────
syncMenuState();
const loaded = await loadBaseCatalog(CATALOG_URL);
base = loaded.catalog;
loadError = loaded.error;
reindex();
render();
