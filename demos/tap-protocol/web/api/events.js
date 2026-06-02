// GET /api/events  — Vercel Node runtime
//
// SSE proxy: streams the Tap's /events feed (re-exposed via the patched
// gateway.py). TAP_URL is held in env; never sent to the browser.
//
// CRITICAL: we MUST NOT interleave a heartbeat write into the middle of a
// data event. The upstream sends each event as `data: {…json…}\n\n`, but
// `reader.read()` returns whatever TCP chunk happens to be ready — a large
// event may arrive in 2-3 reads. If a heartbeat timer fires between two
// such reads, the browser sees a corrupted line like
//   data: {"actual_sha256": "sha256:e216e60: hb
// and `JSON.parse` throws. To prevent this we parse the upstream stream
// into complete SSE messages (split on `\n\n`) and emit each message in
// one `res.write`. Heartbeats are then safe because the writer is always
// between messages, never inside one.

function writeMessage(res, msg) {
  // msg already ends in \n\n; pass through as one syscall.
  try { res.write(msg); } catch (_) {}
}

function offlineEvent(reason) {
  return `data: ${JSON.stringify({ ts: Date.now() / 1000, type: 'offline', id: 0, reason })}\n\n`;
}

module.exports = async function handler(req, res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  const target = process.env.TAP_URL;
  if (!target) {
    writeMessage(res, offlineEvent('TAP_URL not configured'));
    res.end();
    return;
  }

  // The browser disconnecting should kill our upstream read.
  const controller = new AbortController();
  req.on?.('close', () => { try { controller.abort(); } catch (_) {} });

  let upstream;
  try {
    upstream = await fetch(`${target.replace(/\/$/, '')}/events`, {
      headers: { Accept: 'text/event-stream' },
      signal: controller.signal,
    });
  } catch (e) {
    writeMessage(res, offlineEvent(`tap unreachable: ${e.message}`));
    res.end();
    return;
  }

  if (!upstream.ok || !upstream.body) {
    writeMessage(res, offlineEvent(`tap returned HTTP ${upstream.status}`));
    res.end();
    return;
  }

  // ─── Buffered SSE forwarder ───────────────────────────────────────────
  // We track a tail buffer; whenever we see `\n\n` we flush every complete
  // message up to and including that boundary as a single write.
  let buffer = '';
  const decoder = new TextDecoder('utf-8');

  // Heartbeat fires between messages only. We set a flag while we're
  // flushing so the heartbeat timer skips if it happens to fire mid-flush
  // (the timer runs on the same event loop turn as the flush so this is
  // really belt-and-braces — Node IS single-threaded, but Promise
  // microtasks between awaits could in theory open a gap).
  let flushing = false;
  const heartbeat = setInterval(() => {
    if (flushing) return;
    // ": " prefix makes this an SSE comment, which clients ignore but which
    // keeps middleboxes from idling the connection.
    try { res.write(': hb\n\n'); } catch (_) { clearInterval(heartbeat); }
  }, 15000);

  const flushCompleteMessages = () => {
    flushing = true;
    try {
      // Pull complete messages off the front of the buffer (each ending in \n\n).
      let lastEnd = -1;
      let searchFrom = 0;
      while (true) {
        const idx = buffer.indexOf('\n\n', searchFrom);
        if (idx === -1) break;
        lastEnd = idx + 2;
        searchFrom = lastEnd;
      }
      if (lastEnd > 0) {
        const toFlush = buffer.slice(0, lastEnd);
        buffer = buffer.slice(lastEnd);
        try { res.write(toFlush); } catch (_) {}
      }
    } finally {
      flushing = false;
    }
  };

  const reader = upstream.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      flushCompleteMessages();
    }
    // Drain final chunk (incomplete message at EOF — emit it anyway so
    // a trailing message without \n\n still reaches the client).
    buffer += decoder.decode();
    if (buffer.length) {
      try { res.write(buffer); } catch (_) {}
    }
  } catch (_) {
    // upstream dropped or aborted
  } finally {
    clearInterval(heartbeat);
    try { res.end(); } catch (_) {}
  }
};
