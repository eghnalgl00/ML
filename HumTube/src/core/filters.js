// HumTube's quality bar and relevance rules, shared by the catalog builder
// (Node) and live mode (browser) so both make exactly the same decisions.

import { fold, hasAnyPhrase } from './text.js';

export const DEFAULT_THRESHOLDS = Object.freeze({
  minViews: 500_000, // video views
  minLikes: 50_000, // video likes (videos with hidden likes fail)
  minSubscribers: 100_000, // channel subscribers (hidden counts fail)
  minDurationSec: 60, // drops YouTube Shorts-style clips; 0 disables
});

// Titles containing any of these are skipped for everyone: they are popular
// but not useful for learning history (memes, fan edits, game footage...).
export const GLOBAL_EXCLUDE = [
  'parody', 'parodies', 'meme', 'memes', 'edit', 'edits', 'ytp', 'shorts', 'tiktok',
  'funny moments', 'prank', 'roblox', 'minecraft', 'fortnite', 'gta', 'hoi4',
  'hearts of iron', 'rap battle', 'rap battles', 'asmr', 'ai voice', 'ai cover',
  'sings ai', 'try not to laugh',
].map(fold);

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

  const excluded = (foldedTitle) => hasAnyPhrase(foldedTitle, exclude);

  return {
    own(video) {
      const title = fold(video.title);
      if (excluded(title)) return false;
      const tags = fold((video.tags ?? []).join(' | '));
      if (hasAnyPhrase(title, strong) || hasAnyPhrase(tags, strong)) return true;
      if (hasAnyPhrase(title, weak) || hasAnyPhrase(tags, weak)) return true;
      const head = fold((video.description ?? '').slice(0, DESCRIPTION_HEAD));
      return hasAnyPhrase(head, strong);
    },
    mentions(title) {
      const t = fold(title);
      return !excluded(t) && hasAnyPhrase(t, strong);
    },
  };
}

export function isGloballyExcluded(title) {
  return hasAnyPhrase(fold(title), GLOBAL_EXCLUDE);
}

/**
 * Checks that need only the video itself (no channel lookup). Returns the
 * first failing reason, or null when the video passes.
 * Reasons: unavailable | excluded | relevance | views | likes | duration
 */
export function videoRejection(video, matcher, thresholds = DEFAULT_THRESHOLDS) {
  if (!video || video.privacy !== 'public' || video.live !== 'none') return 'unavailable';
  if (isGloballyExcluded(video.title)) return 'excluded';
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
  excluded: 'memes, edits or game footage',
  relevance: 'not clearly about this person',
  views: 'under the view minimum',
  likes: 'under the like minimum (or likes hidden)',
  duration: 'too short (Shorts)',
  subscribers: 'channel under the subscriber minimum',
};
