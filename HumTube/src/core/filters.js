// HumTube's quality bar and relevance rules, shared by the catalog builder
// (Node) and live mode (browser) so both make exactly the same decisions.

import { fold, hasAnyPhrase } from './text.js';

export const DEFAULT_THRESHOLDS = Object.freeze({
  minViews: 500_000, // video views
  minLikes: 50_000, // video likes (videos with hidden likes fail)
  minSubscribers: 100_000, // channel subscribers (hidden counts fail)
  minDurationSec: 60, // drops YouTube Shorts-style clips; 0 disables
  language: 'en', // drops videos in other languages; '' disables
});

// Titles containing any of these are skipped for everyone: they are popular
// but not useful for learning history (memes, fan edits, game footage...).
export const GLOBAL_EXCLUDE = [
  'parody', 'parodies', 'meme', 'memes', 'edit', 'edits', 'ytp', 'shorts', 'tiktok',
  'funny moments', 'prank', 'roblox', 'minecraft', 'fortnite', 'gta', 'hoi4',
  'hearts of iron', 'rap battle', 'rap battles', 'asmr', 'ai voice', 'ai cover',
  'sings ai', 'try not to laugh', 'drawing', 'how to draw', 'acoustic cover',
  'piano cover', 'guitar cover', 'drum cover', 'cover by', 'karaoke',
  'subtitulada', 'subtitulado', 'legendado', 'en espanol', 'sub espanol',
  'audition', 'x factor', 'got talent', 'the voice', 'mashup', 'remix', 'kids song',
  'nursery rhyme', 'metal detecting',
  // exam-prep lessons and other-language re-uploads with English-looking titles
  'hindi', 'urdu', 'class 9', 'class 10', 'class 11', 'class 12', 'ctet', 'dsssb', 'kvs',
  'reet', 'cbse', 'ncert', 'upsc',
].map(fold);

const SHORT_HASHTAGS = ['#fyp', '#viral', '#funfact', '#trending', '#foryou'];

// Channels whose names mark them as clip/short-form outlets.
const CHANNEL_EXCLUDE = ['shorts', 'tiktok', 'memes'].map(fold);

const DESCRIPTION_HEAD = 300;

/**
 * Build the relevance matcher for one person.
 *  - `own(video)`: may a video found by this person's own search be listed under them?
 *  - `mentions(title)`: does a title clearly name them? (used to cross-tag videos
 *    found for other people, e.g. "Churchill vs Hitler" lands on both pages)
 */
export function makeMatcher(person) {
  const strong = person.match.map(fold);
  const weak = (person.primaryMatch ?? []).map(fold);
  const exclude = (person.exclude ?? []).map(fold);

  // Excluded when the title or the channel name has an exclude phrase, or when
  // it's a YouTube "Music" video about a non-musician (a song using their name).
  const rejected = (video) =>
    hasAnyPhrase(fold(video.title), exclude) ||
    hasAnyPhrase(fold(video.channelTitle ?? ''), exclude) ||
    (!person.music && String(video.categoryId ?? '') === '10');

  return {
    own(video) {
      if (rejected(video)) return false;
      const title = fold(video.title);
      if (hasAnyPhrase(title, strong) || hasAnyPhrase(title, weak)) return true;
      if (person.titleOnly) return false;
      const tags = fold((video.tags ?? []).join(' | '));
      if (hasAnyPhrase(tags, strong) || hasAnyPhrase(tags, weak)) return true;
      const head = fold((video.description ?? '').slice(0, DESCRIPTION_HEAD));
      return hasAnyPhrase(head, strong);
    },
    /** Accepts a video ({title, channelTitle, categoryId}) or just a title. */
    mentions(videoOrTitle) {
      const video = typeof videoOrTitle === 'string' ? { title: videoOrTitle } : videoOrTitle;
      return !rejected(video) && hasAnyPhrase(fold(video.title), strong);
    },
  };
}

export function isGloballyExcluded(title, channelTitle = '') {
  const raw = String(title ?? '');
  const t = fold(raw);
  if (hasAnyPhrase(t, GLOBAL_EXCLUDE) || hasAnyPhrase(fold(channelTitle), CHANNEL_EXCLUDE)) return true;
  // Song covers ("... (Frank Sinatra cover)"), but not a "cover-up".
  if (t.includes(' cover ') && !t.includes(' cover up ')) return true;
  // Hashtag-stuffed titles are short-form clips.
  const hashtags = raw.match(/#[\p{L}\p{N}_]+/gu) ?? [];
  return hashtags.length >= 2 || hashtags.some((h) => SHORT_HASHTAGS.includes(h.toLowerCase()));
}

// Common words in Spanish, Portuguese, French, Italian and German titles that
// aren't English words.
const FOREIGN_WORDS = new Set(('el la los las del que y con por para una le les des du et est pour dans ' +
  'une o os da do com uma não nao como mais il di della che per sono der die das und mit ist ein eine ' +
  'nicht auf historia história histoire storia geschichte vida muerte guerra guerre krieg').split(' '));
const ENGLISH_WORDS = new Set(('the of and to in is was how why what who with for on from his her their ' +
  'this that it at by as an are be you your').split(' '));

/**
 * Language check. Uses YouTube's declared audio language when the uploader set
 * one; otherwise rejects titles written mostly in a non-Latin script.
 */
export function isWrongLanguage(video, language) {
  if (!language) return false;
  if (video.lang && !String(video.lang).toLowerCase().startsWith(language.toLowerCase())) return true;
  if (language.toLowerCase() !== 'en') return false;
  // Uploaders often mislabel the language, so also look at the title itself.
  // Count letters and combining marks (Indic vowel signs are marks).
  const title = String(video.title ?? '');
  const chars = title.match(/[\p{L}\p{M}]/gu) ?? [];
  const other = chars.filter((c) => !/[\p{Script=Latin}\p{Script=Common}\p{Script=Inherited}]/u.test(c)).length;
  if (other >= 8 || (chars.length >= 6 && other / chars.length > 0.3)) return true;
  const words = fold(title).trim().split(' ');
  const foreign = words.filter((w) => FOREIGN_WORDS.has(w)).length;
  const english = words.filter((w) => ENGLISH_WORDS.has(w)).length;
  return foreign >= 2 && english === 0;
}

/**
 * Checks that need only the video itself (no channel lookup). Returns the
 * first failing reason, or null when the video passes.
 * Reasons: unavailable | excluded | language | relevance | views | likes | duration
 */
export function videoRejection(video, matcher, thresholds = DEFAULT_THRESHOLDS) {
  if (!video || video.privacy !== 'public' || video.live !== 'none') return 'unavailable';
  if (isGloballyExcluded(video.title, video.channelTitle)) return 'excluded';
  if (isWrongLanguage(video, thresholds.language)) return 'language';
  if (matcher && !matcher.own(video)) return 'relevance';
  if (!(video.views >= thresholds.minViews)) return 'views';
  if (!(video.likes != null && video.likes >= thresholds.minLikes)) return 'likes';
  if (thresholds.minDurationSec > 0 && !(video.durationSec >= thresholds.minDurationSec)) return 'duration';
  return null;
}

/** Returns 'subscribers' when the channel fails the bar (or is unknown/hidden), else null. */
export function channelRejection(channel, thresholds = DEFAULT_THRESHOLDS) {
  if (!channel || channel.subscribers == null) return 'subscribers';
  return channel.subscribers >= thresholds.minSubscribers ? null : 'subscribers';
}

export const REJECTION_LABELS = {
  missing: 'details unavailable',
  unavailable: 'private, live or removed',
  excluded: 'memes, edits, covers, tutorials or game footage',
  language: 'not in English',
  relevance: 'not clearly about this person',
  views: 'under the view minimum',
  likes: 'under the like minimum (or likes hidden)',
  duration: 'too short (Shorts)',
  subscribers: 'channel under the subscriber minimum',
};
