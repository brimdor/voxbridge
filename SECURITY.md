# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| >= 0.2.0| ✅ Active          |
| 0.1.x   | ⚠️ Best-effort     |

## Reporting a Vulnerability

**Please do not open a public issue.**

If you discover a security vulnerability in VoxBridge, please report it privately via GitHub's **Private Vulnerability Reporting** feature (enabled on this repository under **Settings → Security → Code security and analysis → Private vulnerability reporting → Enable**).

Alternatively, email the maintainer with:
- A clear description of the vulnerability
- Steps to reproduce
- Your assessment of impact
- Any suggested fix (optional)

We will acknowledge receipt within 48 hours and provide an initial assessment within 7 days.

## Security Considerations for the HTTP Server

VoxBridge ships an optional FastAPI server (`voxbridge serve`). It is **designed for local/offline use** — the security middleware is a safety layer, not a perimeter. If you expose it beyond localhost:

1. **Rate limiting**: Enable `RateLimiter` via middleware. Default is generous (60 req/min). Tune it.
2. **CORS**: Default origins are `localhost` and `127.0.0.1` only. No wildcard ports. Use `VOXBRIDGE_CORS_ALLOW_ORIGINS` to customize.
3. **Path traversal**: CLI `--output` validates for `..` and escapes. The server does not accept file paths from clients for this reason.
4. **Request / audio size**: 10MB body limit and ~50MB audio output cap by default. Tune via code.
5. **Signal-based timeout**: Synthesis has a 60s default alarm; set `VOXBRIDGE_MAX_SYNTH_SECONDS=0` to disable if your hardware is slow.
6. **Single-threaded synthesis**: The server queues requests and processes them one at a time. There is no DoS vector from concurrent synthesis, but the queue itself can grow without bound until OOM. Monitor `queue_depth` in health responses.

## Hardening Checklist for Production Use

If you are NOT using VoxBridge as a purely local desktop tool, review:

- [ ] `VOXBRIDGE_MAX_SYNTH_SECONDS` capped appropriately for your hardware
- [ ] `RateLimiter` max_requests tuned to actual traffic
- [ ] `CORS` origins set to your exact domains, not `*`
- [ ] Reverse proxy terminating TLS (Cloudflare, Nginx, etc.)
- [ ] `X-Forwarded-For` / `X-Real-IP` headers correctly set by proxy so `RateLimiter` tracks real IPs
