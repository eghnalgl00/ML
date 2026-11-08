# HumTube

**Learn the 20th century through the 100 people who shaped it.**

HumTube collects YouTube videos about TIME’s
[100 Persons of the Century](https://time.com/archive/6735625/time-100-persons-of-the-century/):
documentaries, war history, speeches, films and trailers, interviews, archival
footage and explainers. Only well-received videos from established channels get in.

## The quality bar

A video is listed only if **all** of these hold:

| Rule | Default |
| --- | --- |
| Video views | **≥ 500,000** |
| Video likes | **≥ 50,000** (hidden like counts can’t be checked, so they fail) |
| Channel subscribers | **≥ 100,000** (hidden counts fail) |
| Length | > 60 s, so YouTube Shorts are skipped |
| Relevance | the person’s name or a defining term (“Tank Man”, “Watson and Crick”) is in the title, tags or the start of the description |
| Language | English (the uploader’s declared language, or the title’s script); `--language any` turns this off |
| Not junk | no memes, parodies, fan edits, rap battles, covers, drawing tutorials, game footage or “…Shorts” channels |

The rules live in [`src/core/filters.js`](src/core/filters.js). The catalog
builder and the in-browser live mode share that file, so both always decide the
same way.

Every video is also tagged with topics, shown as filter chips: **War,
Documentaries, Films, Speeches, Interviews, Biographies, Archive footage,
Explainers, Music, Sports** ([`src/core/topics.js`](src/core/topics.js)).

## The 100 people

The five TIME categories (20 people each) and the Person of the Century (Albert
Einstein; runners-up FDR and Gandhi) are in
[`src/core/people.js`](src/core/people.js). Each entry also holds its search
queries and the match/exclude phrases that keep results on topic. For example,
*Indira* Gandhi is kept off Mohandas Gandhi’s page, “Mandela effect” off
Mandela’s, and “Kennedy Space Center” off the Kennedys’.

## Quick start

Needs Node 18+. No `npm install` needed: there are no dependencies.

```bash
cd HumTube
npm start            # http://localhost:5173
```

The repo ships with an **empty** catalog. Fill it in one of two ways.

### 1. Build the catalog (recommended)

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project,
   enable **YouTube Data API v3**, and create an **API key**.
2. Run:

   ```bash
   YOUTUBE_API_KEY=your_key npm run build:catalog
   ```

This searches YouTube for each person, checks views, likes and subscriber counts,
and writes `data/catalog.json`. Commit that file and the site works for everyone,
with no key needed to browse.

**Searches.** Each person gets two searches: their name sorted by **view count**
(finds the popular videos that clear the bar) and “<name> documentary” sorted by
**relevance** (finds actual documentaries and lectures). Sorting only by view count
let in rap songs, drawing tutorials and quote reels; sorting only by relevance
found good documentaries, but few of them reach 50K likes.

**Quota.** The free quota is 10,000 units a day. A search costs 100 units, and
stats lookups cost 1 unit per 50 videos. Two searches for 100 people is ~20,000
units, so a full build takes **two days**. The builder stops at its budget
(default 9,500), searching everyone’s first query before anyone’s second. Progress
is cached in `.cache/` and in `data/catalog.json` itself, so running the **same
command the next day continues where it left off**, even from a fresh clone.

Useful options (`npm run build:catalog -- --help` lists them all):

```bash
npm run build:catalog -- --category leaders          # one TIME category
npm run build:catalog -- --people winston-churchill,pele
npm run build:catalog -- --queries 1                 # only the by-views search (half the quota)
npm run build:catalog -- --offline --min-likes 20000 # rebuild from cache with a different bar, no API calls
npm run build:catalog -- --dry-run                   # show the plan and quota estimate
```

The run ends with a report listing each person who has no qualifying videos and
why (for example “31 under the view minimum”).

### 2. Live mode (no build step)

Click the ⚙ button, paste your API key, open any person and press **Load
videos**. The browser runs the same search and filters. Results are kept in
that browser for 7 days. The key stays in the browser’s local storage and is
sent only to `googleapis.com`. Restrict it to the YouTube Data API and your site’s
address in Cloud Console.

## What to expect

The 50K-likes rule is strict. A typical video gets roughly 1–3 likes per 100
views, so 50K likes usually means a video with a few million views. The big names
(Hitler, Churchill, the Beatles, Muhammad Ali, Einstein) will have plenty. Some
Builders & Titans (Stephen Bechtel, Charles Merrill, Juan Trippe, Leo Burnett)
may have **no** video that clears the bar. Their pages then say so and link
straight to a YouTube search. To loosen the bar for everyone, rebuild with
`--offline --min-likes …`.

## Project layout

```
HumTube/
├── index.html, styles.css, favicon.svg   static single-page app (hash routing)
├── data/catalog.json                     the generated catalog (one video per line)
├── src/core/                             shared by the browser and Node
│   ├── people.js     TIME's 100 + Person of the Century, queries, match rules
│   ├── filters.js    the quality bar and relevance checks
│   ├── topics.js     topic labels
│   ├── youtube.js    YouTube Data API v3 client (search, videos, channels)
│   ├── catalog.js    builds/merges catalogs; live-mode fetch
│   └── text.js       accent/punctuation-insensitive matching
├── src/app/                              the UI (views, state, formatting)
├── scripts/build-catalog.mjs             the catalog builder CLI
├── scripts/serve.mjs                     tiny static server
└── test/                                 node:test suites + a mock YouTube API
```

The layout follows familiar video-site conventions: a top bar with search, a
collapsible sidebar (Home, Watch later, the five TIME categories), a chip bar to
filter by topic, and a video grid that loads more as you scroll. Each person has a
channel-style page with their videos. The watch page has the embedded player
(youtube-nocookie.com), the uploading channel, views, likes, the people in the
video and an up-next list. Search, Watch later and About pages are included.

## Tests

```bash
npm test
```

The tests run the real filters, client and builder against a mock YouTube API.
They cover threshold edges, hidden likes and subscriber counts, name
disambiguation, cross-tagging, quota budgeting, resuming after quota exhaustion,
bad keys, and the static server’s path restrictions.

## Deploying

Everything is static, so any static host works (GitHub Pages, Netlify,
Cloudflare Pages). Publish the `HumTube/` folder after building the catalog.

HumTube is not affiliated with TIME or YouTube. Video data comes from the
YouTube Data API.
