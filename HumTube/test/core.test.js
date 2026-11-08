import { test } from 'node:test';
import assert from 'node:assert/strict';

import { fold, hasAnyPhrase, decodeEntities } from '../src/core/text.js';
import { PEOPLE, PEOPLE_BY_ID, CATEGORIES, PERSON_OF_THE_CENTURY, queriesFor, monogram } from '../src/core/people.js';
import { makeMatcher, videoRejection, channelRejection, isGloballyExcluded, DEFAULT_THRESHOLDS } from '../src/core/filters.js';
import { classify } from '../src/core/topics.js';
import { parseIsoDuration } from '../src/core/youtube.js';

const person = (id) => PEOPLE_BY_ID.get(id);
const video = (over = {}) => ({
  id: 'abcdefghijk', title: '', tags: [], description: '', privacy: 'public', live: 'none',
  views: 1_000_000, likes: 60_000, durationSec: 600, channelId: 'UC1', ...over,
});

test('the TIME 100 list: 100 unique people, 20 per category', () => {
  assert.equal(PEOPLE.length, 100);
  assert.equal(new Set(PEOPLE.map((p) => p.id)).size, 100);
  for (const c of CATEGORIES) assert.equal(PEOPLE.filter((p) => p.category === c.id).length, 20, c.id);
  assert.ok(person(PERSON_OF_THE_CENTURY.id));
  for (const id of PERSON_OF_THE_CENTURY.runnersUp) assert.ok(person(id), id);
  for (const p of PEOPLE) {
    assert.ok(p.match.length, `${p.id} has match phrases`);
    assert.ok(queriesFor(p).length >= 2, `${p.id} has queries`);
    assert.ok(monogram(p).length >= 1);
    // Each person's own name (or a match phrase) must be findable by their matcher.
    assert.ok(makeMatcher(p).mentions(p.match[0]), `${p.id} matches its first phrase`);
  }
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

test('matchers: disambiguation and exclusions', () => {
  const gandhi = makeMatcher(person('mohandas-gandhi'));
  assert.ok(gandhi.own(video({ title: 'Gandhi (1982) Official Trailer' })));
  assert.ok(!gandhi.own(video({ title: 'Indira Gandhi biography' })));
  assert.ok(!gandhi.mentions('Rahul Gandhi press conference'));

  const mandela = makeMatcher(person('nelson-mandela'));
  assert.ok(!mandela.own(video({ title: 'The Mandela Effect Explained' })));
  assert.ok(mandela.own(video({ title: "Mandela's release from prison, 1990" })));

  const fdr = makeMatcher(person('franklin-d-roosevelt'));
  assert.ok(fdr.mentions('Franklin D. Roosevelt: Day of Infamy speech'));
  assert.ok(fdr.mentions('FDR and the New Deal'));
  assert.ok(!fdr.mentions('Theodore Roosevelt and the Rough Riders'));

  assert.ok(makeMatcher(person('lech-walesa')).mentions('Lech Wałęsa and Solidarność'));
  assert.ok(makeMatcher(person('pele')).mentions('PELÉ - Best Goals Ever'));
  assert.ok(!makeMatcher(person('the-kennedys')).mentions('Kennedy Space Center launch'));
  assert.ok(makeMatcher(person('the-kennedys')).mentions('JFK assassination: the full story'));
  assert.ok(!makeMatcher(person('enrico-fermi')).own(video({ title: 'The Fermi Paradox' })));
  assert.ok(!makeMatcher(person('ho-chi-minh')).own(video({ title: 'Ho Chi Minh City street food tour' })));
  assert.ok(!makeMatcher(person('princess-diana')).own(video({ title: 'Diana and Roma play with toys' })));

  // Weak (primary-only) phrases count for own-search results but don't cross-tag.
  const rh = makeMatcher(person('rodgers-hammerstein'));
  assert.ok(rh.own(video({ title: 'The Sound of Music - Do-Re-Mi' })));
  assert.ok(!rh.mentions('The Sound of Music - Do-Re-Mi'));

  // Strong phrases in the opening of the description count; weak ones don't.
  const churchill = makeMatcher(person('winston-churchill'));
  assert.ok(churchill.own(video({ title: 'Darkest Hour', description: 'Winston Churchill faces his greatest test.' })));
  assert.ok(!churchill.own(video({ title: 'Darkest Hour', description: `${'x'.repeat(400)} Churchill` })));
  assert.ok(churchill.own(video({ title: 'Darkest Hour', tags: ['churchill', 'ww2'] })));
});

test('language filter', () => {
  const m = makeMatcher(person('mohandas-gandhi'));
  const ok = video({ title: 'Mahatma Gandhi documentary' });
  assert.equal(videoRejection({ ...ok, lang: 'en-GB' }, m), null);
  assert.equal(videoRejection({ ...ok, lang: 'hi' }, m), 'language');
  assert.equal(videoRejection({ ...ok, title: 'ऐसे हुई थी Mahatma Gandhi की हत्या... (3D Animation)' }, m), 'language');
  assert.equal(videoRejection({ ...ok, lang: 'hi' }, m, { ...DEFAULT_THRESHOLDS, language: '' }), null, 'language check can be disabled');
  assert.equal(videoRejection({ ...ok, lang: null }, m), null, 'no declared language + Latin title passes');
});

test('global exclusions', () => {
  assert.ok(isGloballyExcluded('Hitler edit 🔥 #shorts'));
  assert.ok(isGloballyExcluded('Downfall parody: Hitler finds out'));
  assert.ok(isGloballyExcluded('Gandhi vs Martin Luther King Jr. Epic Rap Battles of History'));
  assert.ok(!isGloballyExcluded('Edited footage of Churchill restored')); // "edited" is not "edit"
  assert.ok(!isGloballyExcluded('Winston Churchill documentary'));
  assert.ok(isGloballyExcluded('Mahatma Gandhi Drawing Portrait Very Easy'));
  assert.ok(isGloballyExcluded('Let It Be - The Beatles (Boyce Avenue acoustic cover)'));
  assert.ok(isGloballyExcluded('She Had A Higher IQ Than Albert Einstein!', 'MrBallen Shorts'));
  assert.ok(!makeMatcher(person('winston-churchill')).own(video({ title: 'Jack Harlow - Churchill Downs feat. Drake' })));
});

test('quality bar: views, likes, subscribers, duration, availability', () => {
  const m = makeMatcher(person('winston-churchill'));
  const ok = video({ title: 'Churchill documentary' });
  assert.equal(videoRejection(ok, m), null);
  assert.equal(videoRejection({ ...ok, views: 500_000, likes: 50_000 }, m), null, 'thresholds are inclusive');
  assert.equal(videoRejection({ ...ok, views: 499_999 }, m), 'views');
  assert.equal(videoRejection({ ...ok, likes: 49_999 }, m), 'likes');
  assert.equal(videoRejection({ ...ok, likes: null }, m), 'likes', 'hidden likes fail');
  assert.equal(videoRejection({ ...ok, durationSec: 59 }, m), 'duration');
  assert.equal(videoRejection({ ...ok, durationSec: 30 }, m, { ...DEFAULT_THRESHOLDS, minDurationSec: 0 }), null);
  assert.equal(videoRejection({ ...ok, privacy: 'private' }, m), 'unavailable');
  assert.equal(videoRejection({ ...ok, live: 'upcoming' }, m), 'unavailable');
  assert.equal(videoRejection({ ...ok, title: 'Cats compilation' }, m), 'relevance');
  assert.equal(videoRejection({ ...ok, title: 'Churchill meme' }, m), 'excluded');

  assert.equal(channelRejection({ subscribers: 100_000 }), null);
  assert.equal(channelRejection({ subscribers: 99_999 }), 'subscribers');
  assert.equal(channelRejection({ subscribers: null }), 'subscribers', 'hidden counts fail');
  assert.equal(channelRejection(undefined), 'subscribers');
});

test('topic classification', () => {
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
  assert.ok(!classify({ title: 'Ted Kennedy remembered' }).includes('explainer'));
});

test('ISO 8601 durations', () => {
  assert.equal(parseIsoDuration('PT45M10S'), 2710);
  assert.equal(parseIsoDuration('PT1H5M'), 3900);
  assert.equal(parseIsoDuration('PT30S'), 30);
  assert.equal(parseIsoDuration('P1DT2H'), 93600);
  assert.equal(parseIsoDuration('P0D'), 0);
  assert.equal(parseIsoDuration(undefined), 0);
});

test('home feed mixes people instead of stacking one person’s videos', async () => {
  const { mixByPerson } = await import('../src/app/views.js');
  const v = (id, pid, views) => ({ id, views, personIds: [pid], foundFor: [pid] });
  const list = [v('b1', 'the-beatles', 900), v('b2', 'the-beatles', 800), v('b3', 'the-beatles', 700), v('c1', 'winston-churchill', 50), v('g1', 'mohandas-gandhi', 20), v('c2', 'winston-churchill', 10)];
  assert.deepEqual(mixByPerson(list).map((x) => x.id), ['b1', 'c1', 'g1', 'b2', 'c2', 'b3']);
});
