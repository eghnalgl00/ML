// A tiny fake of the YouTube Data API v3 for tests. Only the fields HumTube reads.

const CHANNELS = {
  UCbig: { title: 'History Documentaries', subscriberCount: '2400000' },
  UCsmall: { title: 'Tiny History', subscriberCount: '50000' },
  UChidden: { title: 'Secretive Channel', hidden: true },
  UCmovies: { title: 'Movieclips Trailers', subscriberCount: '9000000' },
};

// id: [title, channel, views, likes|null, duration, extra]
const V = {
  churchDoc01: ['Winston Churchill: The Darkest Hour | Full Documentary', 'UCbig', 2_000_000, 80_000, 'PT45M10S'],
  churchHitl1: ['Churchill vs Hitler — The Battle of Britain', 'UCbig', 900_000, 60_000, 'PT22M'],
  churchSmal2: ['Churchill speech: We Shall Fight on the Beaches', 'UCsmall', 3_000_000, 100_000, 'PT12M'],
  churchEdit3: ['Winston Churchill edit 🔥', 'UCbig', 5_000_000, 400_000, 'PT30S'],
  catsTopTen4: ['Top 10 Cats of All Time', 'UCbig', 10_000_000, 500_000, 'PT8M'],
  churchRare5: ['Churchill rare footage', 'UCbig', 400_000, 30_000, 'PT6M'],
  churchHidL6: ['Churchill interview 1959', 'UCbig', 1_000_000, null, 'PT9M'],
  churchQuik7: ['Churchill in 45 seconds', 'UCbig', 2_000_000, 90_000, 'PT45S'],
  churchGone8: null, // returned by search, missing from videos.list (deleted)
  churchHidS9: ['Churchill and the Blitz', 'UChidden', 1_500_000, 70_000, 'PT15M'],
  indiraGand1: ['Indira Gandhi biography', 'UCbig', 4_000_000, 120_000, 'PT20M'],
  gandhiFilm1: ['Gandhi (1982) Official Trailer', 'UCmovies', 6_000_000, 75_000, 'PT2M30S', { categoryId: '1' }],
  mandelaEff1: ['The Mandela Effect Explained', 'UCbig', 9_000_000, 300_000, 'PT14M'],
  mandelaDoc1: ['Nelson Mandela: Long Walk to Freedom documentary', 'UCbig', 1_200_000, 52_000, 'PT1H5M'],
  edgeExactl1: ['Nelson Mandela speech at his release', 'UCbig', 500_000, 50_000, 'PT1M'],
};

const SEARCH = {
  'Winston Churchill': ['churchDoc01', 'churchHitl1', 'churchSmal2', 'churchEdit3', 'catsTopTen4',
    'churchRare5', 'churchHidL6', 'churchQuik7', 'churchGone8', 'churchHidS9'],
  'Mahatma Gandhi': ['indiraGand1', 'gandhiFilm1'],
  'Nelson Mandela': ['mandelaEff1', 'mandelaDoc1', 'edgeExactl1'],
};

function videoItem(id) {
  const row = V[id];
  if (!row) return null;
  const [title, channelId, views, likes, duration, extra = {}] = row;
  return {
    id,
    snippet: {
      title,
      description: `About ${title}`,
      channelId,
      channelTitle: CHANNELS[channelId].title,
      publishedAt: '2019-05-01T12:00:00Z',
      categoryId: extra.categoryId ?? '27',
      liveBroadcastContent: 'none',
      tags: [],
    },
    statistics: { viewCount: String(views), ...(likes == null ? {} : { likeCount: String(likes) }), commentCount: '100' },
    contentDetails: { duration },
    status: { embeddable: true, privacyStatus: 'public' },
  };
}

const json = (status, body) => ({ ok: status < 400, status, json: async () => body });

/**
 * @param {object} [opts]
 * @param {number} [opts.quotaAfterSearches]  return quotaExceeded after this many searches
 */
export function createMockFetch({ quotaAfterSearches = Infinity, badKey = false } = {}) {
  const calls = [];
  let searches = 0;
  const fetchImpl = async (url) => {
    const u = new URL(url);
    const endpoint = u.pathname.split('/').pop();
    calls.push({ endpoint, params: Object.fromEntries(u.searchParams) });
    if (badKey) {
      return json(400, { error: { code: 400, message: 'API key not valid. Please pass a valid API key.', errors: [{ reason: 'keyInvalid' }] } });
    }
    if (endpoint === 'search') {
      if (searches >= quotaAfterSearches) {
        return json(403, { error: { code: 403, message: 'The request cannot be completed because you have exceeded your quota.', errors: [{ reason: 'quotaExceeded' }] } });
      }
      searches += 1;
      const ids = SEARCH[u.searchParams.get('q')] ?? [];
      return json(200, {
        pageInfo: { totalResults: ids.length },
        items: ids.map((id) => ({ id: { kind: 'youtube#video', videoId: id }, snippet: { title: (V[id]?.[0] ?? 'gone').replace(/'/g, '&#39;'), channelId: V[id]?.[1] ?? 'UCbig' } })),
      });
    }
    if (endpoint === 'videos') {
      const ids = u.searchParams.get('id').split(',');
      return json(200, { items: ids.map(videoItem).filter(Boolean) });
    }
    if (endpoint === 'channels') {
      const ids = u.searchParams.get('id').split(',');
      return json(200, {
        items: ids.filter((id) => CHANNELS[id]).map((id) => ({
          id,
          snippet: { title: CHANNELS[id].title },
          statistics: CHANNELS[id].hidden ? { hiddenSubscriberCount: true } : { subscriberCount: CHANNELS[id].subscriberCount, hiddenSubscriberCount: false },
        })),
      });
    }
    return json(404, { error: { message: 'unknown endpoint' } });
  };
  return { fetchImpl, calls, get searches() { return searches; } };
}
