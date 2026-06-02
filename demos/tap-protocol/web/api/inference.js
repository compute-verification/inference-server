// POST /api/inference  — Vercel Node runtime
//
// Proxies {prompt, max_tokens?} to the vast Gateway's POST /request.
// GATEWAY_URL is held in env; never sent to the browser.

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method not allowed' });
    return;
  }

  const target = process.env.GATEWAY_URL;
  if (!target) {
    res.status(503).json({ error: 'GATEWAY_URL not configured (cluster is offline)' });
    return;
  }

  // Vercel parses JSON bodies automatically when Content-Type: application/json.
  const body = (req.body && typeof req.body === 'object') ? req.body : (() => {
    try { return JSON.parse(req.body || '{}'); } catch { return null; }
  })();
  if (!body) {
    res.status(400).json({ error: 'bad JSON body' });
    return;
  }

  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';
  if (!prompt) {
    res.status(400).json({ error: 'prompt required' });
    return;
  }
  const maxTokens = Math.max(1, Math.min(256, body.max_tokens | 0 || 128));

  // 60-second budget. Hobby tier function limit is 60s anyway.
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 55 * 1000);

  try {
    const upstream = await fetch(`${target.replace(/\/$/, '')}/request`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: maxTokens }),
      signal: controller.signal,
    });
    clearTimeout(t);

    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader('Content-Type', upstream.headers.get('Content-Type') || 'application/json');
    res.send(text);
  } catch (e) {
    clearTimeout(t);
    const reason = e.name === 'AbortError' ? 'gateway timeout' : `gateway unreachable: ${e.message}`;
    res.status(502).json({ error: reason });
  }
};
