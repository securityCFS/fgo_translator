# GitHub-Hosted Translation Cache Design

Date: 2026-05-13

## Goal

Reduce repeated LLM token usage by reusing trusted script translations across runs and across deployment modes.

The cache must work for both branches:

- `master`: Flask-backed app. It can generate trusted translations on the server and publish cache entries.
- `static-migration`: pure static app. It can read public cache entries from GitHub, but it must not upload browser-generated translations.

The cache database must be hostable on GitHub. GitHub Pages will serve immutable JSON files as a static read path. Trusted server deployments will write JSON files into the cache repository through the GitHub Contents API.

## Non-Goals

- No user-submitted translation review workflow in the first version.
- No public write endpoint for browser clients.
- No caching of client-side translations from `static-migration`.
- No attempt to merge partial line-level translations from different models.
- No server-side database process is required for the first version.

## Repository Model

Create a separate repository, for example `fgo-translation-cache`, published with GitHub Pages.

The repository stores one JSON file per script/model/language/source hash:

```text
v1/{source_region}/{script_id}/{source_hash}/{target_language}/{provider}/{model}/{prompt_version}.json
```

Example:

```text
v1/JP/0400041440/9c2f...a31d/zh-CN/gemini/gemini-2.5-flash/fgo-v1.json
```

This gives static clients a deterministic URL:

```text
{CACHE_BASE_URL}/v1/JP/0400041440/9c2f...a31d/zh-CN/gemini/gemini-2.5-flash/fgo-v1.json
```

A 200 response is a cache hit. A 404 is a miss.

## Cache Key

The cache key consists of:

- `script_id`: Atlas script ID.
- `source_region`: usually `JP`, normalized to Atlas region names.
- `source_hash`: SHA-256 of the canonical extracted source dialogues.
- `target_language`: normalized BCP-47-like label such as `zh-CN`, `zh-TW`, `en`.
- `provider`: normalized provider family, for example `gemini`, `openai`, `deepseek`, `qwen`, `claude`, or `custom`.
- `model`: sanitized model name.
- `prompt_version`: explicit prompt/parser contract version, initially `fgo-v1`.

`source_hash` is computed from canonical JSON:

```json
[
  {"speaker":"...", "content":"..."},
  {"speaker":"...", "content":"..."}
]
```

Rules:

- Preserve dialogue order.
- Normalize line endings to `\n`.
- Trim only the same way the existing extractor trims.
- Use `ensure_ascii=false` for semantic stability before hashing if implemented in Python, with sorted keys and compact separators.

The server must compute `source_hash` from its own extraction, not from client-supplied text.

## Cache Entry Schema

```json
{
  "schema_version": 1,
  "script_id": "0400041440",
  "source_region": "JP",
  "source_hash": "sha256...",
  "target_language": "zh-CN",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "prompt_version": "fgo-v1",
  "dialogue_count": 123,
  "trusted_generation": true,
  "generator": {
    "app": "fgo_translator",
    "branch_mode": "server",
    "generated_at": "2026-05-13T00:00:00Z"
  },
  "translations": [
    {
      "speaker": "...",
      "translated_content": "..."
    }
  ]
}
```

Validation rules:

- `schema_version` must be `1`.
- `trusted_generation` must be `true`.
- `dialogue_count` must equal `translations.length`.
- `dialogue_count` must equal the current extracted source dialogue count.
- No translation may contain `[Translation Error:`.
- Cache reader must ignore files that fail validation.

## `master` Server Flow

The Flask `/translate` route should accept additional metadata:

- `script_ids`
- `script_dialogue_counts`
- `source_region`
- `target_language`
- `translation_method`
- `session_id`

Flow:

1. Frontend extracts dialogues as it does today and sends the combined list plus script metadata to `/translate`.
2. Server resolves provider/model from saved preferences or environment variables.
3. Server re-extracts each requested script using `loader.extract_dialogues(script_id, source_region)`.
4. Server computes a cache key for each script.
5. Server checks GitHub Pages cache for each script.
6. Cache hits are validated and inserted into the response.
7. Cache misses are translated by the server LLM client.
8. Server validates translated script slices.
9. Server writes successful cache misses to the cache repository using a GitHub token.
10. Server returns a combined `translated_dialogues` list in the original script order.

The existing progress Socket.IO updates can remain batch-based. Cache hits should still advance progress so the UI does not appear stuck.

If GitHub cache read fails because of network or rate limits, translation should continue normally. If cache write fails, translation should still return to the user and log the write failure.

## `static-migration` Flow

The static app cannot be trusted to upload translations because API keys and request bodies live in the browser.

Flow:

1. Static app extracts dialogues using `js/api.js`.
2. Static app computes the same canonical `source_hash` in JavaScript.
3. Static app builds deterministic GitHub Pages cache URLs.
4. Cache hits are used directly.
5. Cache misses are translated in the browser with `Translator.translateDialogues()`.
6. Browser-generated translations are displayed locally and passed to gaming mode.
7. Browser-generated translations are not uploaded to GitHub.

This gives static deployments token savings when the central cache has entries, without opening a moderation or trust problem.

## Publishing And Secrets

Server deployments use environment variables:

- `TRANSLATION_CACHE_BASE_URL`: public GitHub Pages base URL.
- `TRANSLATION_CACHE_REPO`: `owner/repo` for writes.
- `TRANSLATION_CACHE_BRANCH`: publish branch, default `main`.
- `TRANSLATION_CACHE_TOKEN`: GitHub token with contents write access.
- `TRANSLATION_CACHE_PROMPT_VERSION`: default `fgo-v1`.
- `TRANSLATION_CACHE_ENABLED`: default enabled when base URL is configured.
- `TRANSLATION_CACHE_WRITE_ENABLED`: default enabled only when token and repo are configured.

The static app only needs:

- cache base URL, either hardcoded in `js/api.js`/`index.html` config or stored in local preferences.

No GitHub token is exposed to static clients.

## Collision And Path Safety

Path segments must be sanitized:

- Keep alphanumeric, `.`, `_`, and `-`.
- Replace `/`, `:`, spaces, and other characters with `_`.
- Use the full SHA-256 source hash or at least a 32-character prefix.

Before upload, the server should request the target file. If it already exists and validates, do not overwrite. If it exists but does not validate, log and skip overwrite in the first version.

## Branch Synchronization Plan

Implement shared concepts in both branches:

- Same cache key fields.
- Same source hash algorithm.
- Same entry schema.
- Same cache URL layout.
- Same validation rules.

Branch-specific integration:

- `master`: add Python cache client and wire it into `/translate`.
- `static-migration`: add JavaScript cache lookup and validation before browser translation.
- Flask template files in `static-migration` should follow the same server behavior as `master` where applicable.

Because `static-migration` has diverged significantly, merge by porting the cache feature rather than attempting a broad branch merge.

## Testing

Unit tests:

- Python source hash is stable for equivalent dialogue lists.
- Python cache path sanitizes model names.
- Python cache reader rejects count mismatch and error placeholders.
- Python `/translate` uses cache hits before model calls.
- Python `/translate` uploads only server-generated successful misses.

JavaScript tests or lightweight browser checks:

- Static source hash matches Python for the same fixture.
- Static app uses cache hits.
- Static app falls back to browser translation on 404.
- Static app never attempts a write request.

Manual verification:

- Start Flask app on `master`.
- Translate a known script once and confirm JSON is written to the cache repo.
- Translate the same script again and confirm no model call is made.
- Open `static-migration` static page, translate the same script, and confirm it reads the GitHub cache.
- Confirm a missing static cache entry still translates locally and does not upload.

## Risks

- Public cache entries may contain game text and translated dialogue. A private cache repository avoids public redistribution but cannot be read by pure GitHub Pages clients.
- GitHub Pages CDN propagation can delay new cache entries. The server should return freshly generated translations directly even if the static cache is not immediately visible.
- GitHub API rate limits can affect bulk writes. Server should batch checks and tolerate write failures.
- Prompt changes can make older translations stylistically inconsistent. `prompt_version` prevents accidental reuse across incompatible prompt contracts.

## First Implementation Slice

Build the smallest useful version:

1. Add Python cache key/hash/schema utilities.
2. Add Python GitHub Pages read and GitHub Contents API write client.
3. Wire cache into `master` `/translate` for LLM translations only.
4. Port read-only cache lookup to `static-migration`.
5. Add tests/fixtures for cross-language hash parity.

Atlas/Rayshift and Free Engine paths remain unchanged in the first slice.
