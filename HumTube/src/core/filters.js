// HumTube's quality bar, shared by the catalog builder (Node) and live mode
// (browser). YouTube's own search ranking decides what's relevant; this only
// applies the minimums.

export const DEFAULT_THRESHOLDS = Object.freeze({
  minViews: 500_000, // video views
  minLikes: 50_000, // video likes (videos with hidden likes fail)
  minSubscribers: 100_000, // channel subscribers (hidden counts fail)
  minDurationSec: 60, // skips YouTube Shorts; 0 turns this off
});

/**
 * Checks that need only the video itself (no channel lookup). Returns the
 * first failing reason, or null when the video passes.
 * Reasons: unavailable | views | likes | duration
 */
export function videoRejection(video, thresholds = DEFAULT_THRESHOLDS) {
  if (!video || video.privacy !== 'public' || video.live !== 'none') return 'unavailable';
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
  views: 'under the view minimum',
  likes: 'under the like minimum (or likes hidden)',
  duration: 'Shorts',
  subscribers: 'channel under the subscriber minimum',
};
