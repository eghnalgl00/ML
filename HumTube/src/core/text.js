// Text helpers shared by the browser app and the Node catalog builder.
//
// `fold` turns any string into a lowercase, accent-free, punctuation-free form
// padded with single spaces, so whole-word / whole-phrase matching becomes a
// plain substring check:  fold("Lech Wałęsa's speech") === " lech walesa s speech "

const SPECIAL = {
  ł: 'l', Ł: 'l', ø: 'o', Ø: 'o', ß: 'ss', æ: 'ae', Æ: 'ae',
  œ: 'oe', Œ: 'oe', đ: 'd', Đ: 'd', ı: 'i', þ: 'th', Þ: 'th',
};

export function fold(text = '') {
  const core = String(text)
    .replace(/[łŁøØßæÆœŒđĐıþÞ]/g, (c) => SPECIAL[c])
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  return core ? ` ${core} ` : ' ';
}

/** True when the folded `haystack` contains any of the folded `phrases`. */
export function hasAnyPhrase(foldedHaystack, foldedPhrases) {
  for (const p of foldedPhrases) {
    if (p.trim() && foldedHaystack.includes(p)) return true;
  }
  return false;
}

export function slugify(text) {
  return fold(text).trim().replace(/ /g, '-');
}

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", '#39': "'" };

/** search.list titles come HTML-escaped; videos.list titles do not. */
export function decodeEntities(text = '') {
  return String(text).replace(/&(#x?[0-9a-f]+|[a-z]+|#39);/gi, (m, code) => {
    const key = code.toLowerCase();
    if (key in ENTITIES) return ENTITIES[key];
    if (key.startsWith('#x')) return String.fromCodePoint(parseInt(key.slice(2), 16));
    if (key.startsWith('#')) return String.fromCodePoint(parseInt(key.slice(1), 10));
    return m;
  });
}
