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
| Not junk | no memes, parodies, fan edits, rap battles or game footage |

The rules live in [`src/core/filters.js`](src/core/filters.js). The catalog
builder and the in-browser live mode share that file, so both always decide the
same way.

Every video is also tagged with topics: **War & Conflict, Documentary, Movies &
Drama, Speeches, Interviews, Biography, Archival Footage, Explainers & Lessons,
Music & Performance, Sports** ([`src/core/topics.js`](src/core/topics.js)).

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

**Quota.** The free quota is 10,000 units a day. One search costs 100 units, and
stats lookups cost 1 unit per 50 videos. One query for each of the 100 people is
~10,000 units, so the builder stops at its budget (default 9,500). All progress
is cached in `.cache/`, so running the **same command the next day continues
where it left off**.

Useful options (`npm run build:catalog -- --help` lists them all):

```bash
npm run build:catalog -- --category leaders          # one TIME category
npm run build:catalog -- --people winston-churchill,pele
npm run build:catalog -- --queries 2                 # also search "<name> documentary" (2× quota)
npm run build:catalog -- --offline --min-likes 20000 # rebuild from cache with a different bar, no API calls
npm run build:catalog -- --dry-run                   # show the plan and quota estimate
```

The run ends with a report listing each person who has no qualifying videos and
why (for example “31 under the view minimum”).

### 2. Live mode (no build step)

Click the ⚙ button, paste your API key, open any person and press **Find videos
on YouTube**. The browser runs the same search and filters. Results are kept in
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

Pages: home (Person of the Century, topics, most watched, all five categories),
category, person (topic filters, sort, search report), topic, watch (embedded
player via youtube-nocookie.com, stats, related), search, watch later, about.

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
