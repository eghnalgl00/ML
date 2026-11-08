import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHumTubeServer } from '../scripts/serve.mjs';

test('static server serves the app and nothing else', async (t) => {
  const server = createHumTubeServer().listen(0);
  t.after(() => server.close());
  await new Promise((r) => server.once('listening', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const get = (p) => fetch(base + p).then((r) => ({ status: r.status, type: r.headers.get('content-type') }));

  assert.deepEqual(await get('/'), { status: 200, type: 'text/html; charset=utf-8' });
  assert.equal((await get('/src/app/main.js')).type, 'text/javascript; charset=utf-8');
  assert.equal((await get('/data/catalog.json')).status, 200);
  assert.equal((await get('/.cache/searches.json')).status, 404);
  assert.equal((await get('/scripts/build-catalog.mjs')).status, 404);
  assert.equal((await get('/package.json')).status, 404);
  assert.equal((await get('/%2e%2e/%2e%2e/etc/passwd')).status, 404);
  assert.equal((await get('/src/../../README.md')).status, 404);
});
