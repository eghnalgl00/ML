import { test } from 'node:test';
import assert from 'node:assert/strict';

import { fold, hasAnyPhrase, decodeEntities } from '../src/core/text.js';
import { PEOPLE, PEOPLE_BY_ID, CATEGORIES, PERSON_OF_THE_CENTURY, queryFor, monogram } from '../src/core/people.js';
import { videoRejection, channelRejection, DEFAULT_THRESHOLDS } from '../src/core/filters.js';
import { classify } from '../src/core/topics.js';
import { parseIsoDuration } from '../src/core/youtube.js';
import { youtubeOrder } from '../src/app/views.js';

const person = (id) => PEOPLE_BY_ID.get(id);
const video = (over = {}) => ({
  id: 'abcdefghijk', title: 'Anything', privacy: 'public', live: 'none',
  views: 1_000_000, likes: 60_000, durationSec: 600, channelId: 'UC1', ...over,
});

test('the TIME 100 list: 100 unique people, 20 per category', () => {
  assert.equal(PEOPLE.length, 100);
  assert.equal(new Set(PEOPLE.map((p) => p.id)).size, 100);
  for (const c of CATEGORIES) assert.equal(PEOPLE.filter((p) => p.category === c.id).length, 20, c.id);
  assert.ok(person(PERSON_OF_THE_CENTURY.id));
  for (const id of PERSON_OF_THE_CENTURY.runnersUp) assert.ok(person(id), id);
  for (const p of PEOPLE) {
    assert.ok(queryFor(p).length > 2, `${p.id} has a search phrase`);
    assert.ok(Array.isArray(p.aliases), `${p.id} has aliases`);
    assert.ok(monogram(p).length >= 1);
  }
  assert.equal(queryFor(person('winston-churchill')), 'Winston Churchill', 'the name by default');
  assert.equal(queryFor(person('unknown-rebel')), 'Tiananmen Square Tank Man', 'a phrase when the name is ambiguous');
});

test('fold: case, accents, punctuation, special letters', () => {
  assert.equal(fold("Lech Wałęsa's speech"), ' lech walesa s speech ');
  assert.equal(fold('Pelé — O Rei!'), ' pele o rei ');
  assert.equal(fold('Kurt Gödel'), ' kurt godel ');
  assert.equal(fold('Franklin D. Roosevelt'), ' franklin d roosevelt ');
  assert.equal(fold(''), ' ');
  assert.ok(hasAnyPhrase(fold('The Hitler Youth'), [fold('Hitler')]));
  assert.ok(!hasAnyPhrase(fold('Hitlerism'), [fold('Hitler')]), 'whole words only');
});

test('decodeEntities handles search.list escaping', () => {
  assert.equal(decodeEntities('Churchill&#39;s &quot;Finest Hour&quot; &amp; more'), 'Churchill\'s "Finest Hour" & more');
  assert.equal(decodeEntities('&#x2014;'), '—');
});

test('quality bar: views, likes, subscribers, Shorts, availability — nothing else', () => {
  assert.equal(videoRejection(video()), null);
  assert.equal(videoRejection(video({ title: 'Top 10 Cats of All Time' })), null, 'relevance is YouTube’s job');
  assert.equal(videoRejection(video({ views: 500_000, likes: 50_000 })), null, 'thresholds are inclusive');
  assert.equal(videoRejection(video({ views: 499_999 })), 'views');
  assert.equal(videoRejection(video({ likes: 49_999 })), 'likes');
  assert.equal(videoRejection(video({ likes: null })), 'likes', 'hidden likes fail');
  assert.equal(videoRejection(video({ durationSec: 59 })), 'duration');
  assert.equal(videoRejection(video({ durationSec: 30 }), { ...DEFAULT_THRESHOLDS, minDurationSec: 0 }), null);
  assert.equal(videoRejection(video({ privacy: 'private' })), 'unavailable');
  assert.equal(videoRejection(video({ live: 'upcoming' })), 'unavailable');

  assert.equal(channelRejection({ subscribers: 100_000 }), null);
  assert.equal(channelRejection({ subscribers: 99_999 }), 'subscribers');
  assert.equal(channelRejection({ subscribers: null }), 'subscribers', 'hidden counts fail');
  assert.equal(channelRejection(undefined), 'subscribers');
});

test('topic labels for the chips', () => {
  assert.deepEqual(classify({ title: 'Winston Churchill | Full Documentary' }), ['documentary']);
  assert.ok(classify({ title: 'Churchill vs Hitler: The Battle of Britain' }).includes('war'));
  assert.ok(classify({ title: 'Gandhi (1982) Official Trailer' }).includes('film'));
  assert.ok(classify({ title: 'I Have a Dream speech' }).includes('speech'));
  assert.ok(classify({ title: 'Bruce Lee interview 1971' }).includes('interview'));
  assert.ok(classify({ title: 'Rare color footage of 1940s London' }).includes('archival'));
  assert.ok(classify({ title: 'Something', categoryId: '10' }).includes('music'));
  assert.ok(classify({ title: 'Pelé best goals' }).includes('sports'));
  assert.ok(classify({ title: 'Anything at all', channelTitle: 'Timeline - World History Documentaries' }).includes('documentary'));
  assert.deepEqual(classify({ title: 'Mother Teresa' }), ['other']);
});

test('ISO 8601 durations', () => {
  assert.equal(parseIsoDuration('PT45M10S'), 2710);
  assert.equal(parseIsoDuration('PT1H5M'), 3900);
  assert.equal(parseIsoDuration('PT30S'), 30);
  assert.equal(parseIsoDuration('P1DT2H'), 93600);
  assert.equal(parseIsoDuration('P0D'), 0);
  assert.equal(parseIsoDuration(undefined), 0);
});

test('home feed keeps YouTube’s order: everyone’s #1 result, then everyone’s #2', () => {
  const v = (id, pid, rank, views) => ({ id, views, personIds: [pid], ranks: { [pid]: rank } });
  const list = [
    v('b2', 'the-beatles', 2, 900), v('b1', 'the-beatles', 1, 10), v('b3', 'the-beatles', 3, 800),
    v('c1', 'winston-churchill', 1, 50), v('c2', 'winston-churchill', 4, 5),
  ];
  // Round 1: b1 (10 views) and c1 (50) → by views: c1, b1. Round 2: b2, c2 → b2, c2. Round 3: b3.
  assert.deepEqual(youtubeOrder(list).map((x) => x.id), ['c1', 'b1', 'b2', 'c2', 'b3']);
});
