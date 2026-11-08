// Topic labels ("War & Conflict", "Documentary", "Movies & Drama", ...) assigned
// to each video from its title, channel name and YouTube category. A video can
// carry several topics. Matching is whole-word/phrase after `fold`.

import { fold, hasAnyPhrase } from './text.js';

export const TOPICS = [
  {
    id: 'war', label: 'War & Conflict', icon: '⚔️',
    phrases: ['war', 'wars', 'wwi', 'wwii', 'ww1', 'ww2', 'world war', 'battle', 'battles', 'invasion',
      'military', 'army', 'soldier', 'soldiers', 'troops', 'd day', 'blitz', 'nazi', 'nazis', 'third reich',
      'holocaust', 'genocide', 'combat', 'front line', 'frontline', 'bombing', 'atomic bomb', 'siege',
      'warfare', 'cold war', 'civil war', 'normandy', 'pearl harbor', 'stalingrad', 'dunkirk',
      'luftwaffe', 'wehrmacht', 'vietnam war', 'korean war', 'gulf war', 'resistance fighters', 'guerrilla'],
  },
  {
    id: 'documentary', label: 'Documentary', icon: '🎞️',
    phrases: ['documentary', 'documentaries', 'docuseries', 'full documentary', 'doc', 'the untold story',
      'inside the', 'secret history', 'the real story', 'true story'],
    channelPhrases: ['documentary', 'documentaries', 'real stories', 'timeline', 'free documentary',
      'national geographic', 'smithsonian channel', 'history channel', 'american experience', 'frontline',
      'dw documentary', 'bbc studios', 'channel 4 documentaries'],
  },
  {
    id: 'film', label: 'Movies & Drama', icon: '🎬',
    phrases: ['movie', 'movies', 'film', 'films', 'trailer', 'teaser', 'scene', 'scenes', 'biopic',
      'full movie', 'official clip', 'movie clip', 'drama', 'miniseries', 'tv series',
      'netflix', 'hbo', 'the crown', 'oscar', 'oscars'],
    categoryIds: ['30', '44'],
  },
  {
    id: 'speech', label: 'Speeches', icon: '🎙️',
    phrases: ['speech', 'speeches', 'address', 'addresses', 'i have a dream', 'fireside chat',
      'we shall fight', 'inaugural', 'keynote', 'commencement', 'sermon', 'rally', 'oration',
      'declaration of war', 'tear down this wall', 'last speech', 'famous speech'],
  },
  {
    id: 'interview', label: 'Interviews', icon: '💬',
    phrases: ['interview', 'interviews', 'interviewed', 'conversation', 'in his own words',
      'in her own words', 'in their own words', '60 minutes', 'q a', 'talks about', 'talk show',
      'podcast', 'dick cavett', 'michael parkinson', 'press conference', 'oprah show', 'the tonight show'],
  },
  {
    id: 'biography', label: 'Biography', icon: '📜',
    phrases: ['biography', 'biographical', 'life story', 'life of', 'story of', 'who was', 'who is',
      'rise and fall', 'early life', 'the man who', 'the woman who', 'legacy', 'last days',
      'final days', 'death of', 'how he became', 'how she became', 'the life and', 'untold story',
      'tragic life', 'life and death', 'remembering'],
  },
  {
    id: 'archival', label: 'Archival Footage', icon: '📽️',
    phrases: ['footage', 'archive', 'archival', 'rare', 'newsreel', 'colorized', 'colourized',
      'in color', 'in colour', 'restored', 'original recording', 'real voice', 'actual footage',
      'british pathe', 'pathe', 'unseen', 'never before seen', 'home movies', 'audio recording',
      '1900s', '1910s', '1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s'],
  },
  {
    id: 'explainer', label: 'Explainers & Lessons', icon: '🧠',
    phrases: ['explained', 'explainer', 'history of', 'in minutes', 'in 5 minutes', 'in 10 minutes',
      'animated', 'animation', 'lesson', 'lecture', 'ted ed', 'ted talk', 'tedx', 'crash course',
      'how did', 'why did', 'what if', 'the truth about', 'facts', 'everything you need to know',
      'timeline', 'summary', 'theory', 'how it works', 'what is', 'why', 'how'],
    categoryIds: ['27', '28'],
  },
  {
    id: 'music', label: 'Music & Performance', icon: '🎵',
    phrases: ['live', 'concert', 'performance', 'performs', 'song', 'songs', 'official video',
      'music video', 'official music video', 'lyrics', 'album', 'sings', 'singing', 'full concert',
      'remastered', 'ballet', 'dance', 'broadway', 'musical', 'jazz', 'symphony', 'opera'],
    categoryIds: ['10'],
  },
  {
    id: 'sports', label: 'Sports', icon: '🏅',
    phrases: ['fight of the century', 'knockout', 'knockouts', 'goals', 'skills', 'highlights', 'boxing',
      'bout', 'championship', 'world cup', 'olympics', 'nfl', 'super bowl', 'baseball', 'football',
      'soccer', 'kung fu', 'martial arts', 'mlb', 'heavyweight', 'rumble in the jungle', 'thrilla'],
    categoryIds: ['17'],
  },
];

export const OTHER_TOPIC = { id: 'other', label: 'General History', icon: '🏛️' };
export const ALL_TOPICS = [...TOPICS, OTHER_TOPIC];
export const TOPIC_BY_ID = new Map(ALL_TOPICS.map((t) => [t.id, t]));

const COMPILED = TOPICS.map((t) => ({
  id: t.id,
  phrases: t.phrases.map(fold),
  channelPhrases: (t.channelPhrases ?? []).map(fold),
  categoryIds: new Set(t.categoryIds ?? []),
}));

/** @param {{title?: string, channelTitle?: string, categoryId?: string}} video */
export function classify(video) {
  const title = fold(video.title);
  const channel = fold(video.channelTitle);
  const found = [];
  for (const t of COMPILED) {
    if (
      hasAnyPhrase(title, t.phrases) ||
      hasAnyPhrase(channel, t.channelPhrases) ||
      t.categoryIds.has(String(video.categoryId ?? ''))
    ) {
      found.push(t.id);
    }
  }
  return found.length ? found : [OTHER_TOPIC.id];
}
