/**
 * translate.js — Browser-side translation using Gemini (or OpenAI-compatible) API
 * API key and settings are stored in localStorage.
 */
const Translator = (() => {

    // ── Preferences ─────────────────────────────────────────────────────────

    const PREFS_KEY = 'fgo_translator_prefs';

    function loadPrefs() {
        try {
            return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}');
        } catch { return {}; }
    }

    function savePrefs(obj) {
        const cur = loadPrefs();
        localStorage.setItem(PREFS_KEY, JSON.stringify({ ...cur, ...obj }));
    }

    function getPref(key, fallback = '') {
        return loadPrefs()[key] ?? fallback;
    }

    function _sanitizePathSegment(value) {
        const safe = String(value || '').trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '');
        return safe || 'unknown';
    }

    function _normalizeTargetLanguage(value) {
        const raw = String(value || '').trim();
        const lowered = raw.toLowerCase();
        const mapping = {
            'chinese': 'zh-CN',
            'chinese simplified': 'zh-CN',
            'chinese (simplified)': 'zh-CN',
            'simplified chinese': 'zh-CN',
            'zh-cn': 'zh-CN',
            'chinese traditional': 'zh-TW',
            'chinese (traditional)': 'zh-TW',
            'traditional chinese': 'zh-TW',
            'zh-tw': 'zh-TW',
            'english': 'en',
            'en': 'en',
            'japanese': 'ja',
            'ja': 'ja',
            'korean': 'ko',
            'ko': 'ko',
        };
        return mapping[lowered] || _sanitizePathSegment(raw);
    }
    function _normalizeProvider(apiType, model, method) {
        if (method === 'free') return 'google-translate';
        const raw = String(apiType || 'openai').trim().toLowerCase();
        const modelLower = String(model || '').toLowerCase();
        if (raw === 'gemini' || modelLower.includes('gemini')) return 'gemini';
        if (modelLower.includes('deepseek')) return 'deepseek';
        if (modelLower.includes('qwen') || modelLower.includes('qwq')) return 'qwen';
        if (modelLower.includes('claude')) return 'claude';
        if (raw === 'openai' || raw === 'custom') return raw;
        return _sanitizePathSegment(raw);
    }

    async function _sha256Hex(value) {
        const bytes = new TextEncoder().encode(value);
        const hash = await crypto.subtle.digest('SHA-256', bytes);
        return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
    }

    async function _canonicalSourceHash(dialogues) {
        const payload = (dialogues || []).map(item => ({
            content: String(item.content || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
            speaker: String(item.speaker || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
        }));
        return _sha256Hex(JSON.stringify(payload));
    }

    function _cacheRelativePath(key) {
        return [
            'v1',
            _sanitizePathSegment(String(key.sourceRegion || 'JP').toUpperCase()),
            _sanitizePathSegment(key.scriptId),
            _sanitizePathSegment(key.sourceHash),
            _sanitizePathSegment(key.targetLanguage),
            _sanitizePathSegment(key.provider),
            _sanitizePathSegment(key.model),
            `${_sanitizePathSegment(key.promptVersion || 'fgo-v1')}.json`,
        ].join('/');
    }

    function _base64Utf8(value) {
        const bytes = new TextEncoder().encode(value);
        let binary = '';
        bytes.forEach(byte => { binary += String.fromCharCode(byte); });
        return btoa(binary);
    }

    function _hasTranslationErrors(translations) {
        return (translations || []).some(item => String(item.translated_content || '').includes('[Translation Error:'));
    }

    async function saveTranslationCache(opts = {}) {
        const prefs = loadPrefs();
        const repo = String(prefs.githubCacheRepo || prefs.github_cache_repo || prefs.cacheRepo || '').trim();
        const token = String(prefs.githubCacheToken || prefs.github_cache_token || prefs.cacheToken || '').trim();
        const branch = String(prefs.githubCacheBranch || prefs.github_cache_branch || prefs.cacheBranch || 'main').trim() || 'main';
        const promptVersion = String(prefs.translationCachePromptVersion || prefs.translation_cache_prompt_version || prefs.cachePromptVersion || 'fgo-v1').trim() || 'fgo-v1';
        if (!repo || !token) {
            return { configured: false, written: 0, skipped: opts.scriptIds?.length || 0, errors: [] };
        }

        const scriptIds = (opts.scriptIds || []).map(String).filter(Boolean);
        const counts = opts.scriptDialogueCounts || [];
        const sources = opts.sourceDialogues || [];
        const translations = opts.translatedDialogues || [];
        const provider = _normalizeProvider(opts.apiType, opts.baseModel, opts.method);
        const model = opts.method === 'free' ? 'free-engine' : (opts.baseModel || (opts.apiType === 'gemini' ? 'gemini-2.5-flash' : 'unknown'));
        const targetLanguage = _normalizeTargetLanguage(opts.targetLanguage || 'Chinese Simplified');
        const sourceRegion = String(opts.sourceRegion || 'JP').toUpperCase();

        const headers = {
            'Accept': 'application/vnd.github+json',
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            'X-GitHub-Api-Version': '2022-11-28',
        };
        let offset = 0, written = 0, skipped = 0;
        const errors = [];

        for (let i = 0; i < scriptIds.length; i++) {
            const scriptId = scriptIds[i];
            const count = Number(counts[i] || 0);
            const sourceSlice = sources.slice(offset, offset + count);
            const translatedSlice = translations.slice(offset, offset + count);
            offset += count;
            if (!count || translatedSlice.length !== sourceSlice.length || _hasTranslationErrors(translatedSlice)) {
                skipped += 1;
                continue;
            }

            const sourceHash = await _canonicalSourceHash(sourceSlice);
            const key = { scriptId, sourceRegion, sourceHash, targetLanguage, provider, model, promptVersion };
            const path = _cacheRelativePath(key);
            const url = `https://api.github.com/repos/${repo}/contents/${encodeURIComponent(path).replace(/%2F/g, '/')}`;
            try {
                const existing = await fetch(`${url}?ref=${encodeURIComponent(branch)}`, { headers });
                if (existing.ok) {
                    skipped += 1;
                    continue;
                }
                if (existing.status !== 404) {
                    errors.push(`${scriptId}: existence check HTTP ${existing.status}`);
                    continue;
                }
                const body = {
                    schema_version: 1,
                    script_id: scriptId,
                    source_region: sourceRegion,
                    source_hash: sourceHash,
                    target_language: targetLanguage,
                    provider,
                    model,
                    prompt_version: promptVersion,
                    dialogue_count: count,
                    trusted_generation: true,
                    generator: {
                        app: 'fgo_translator',
                        branch_mode: 'static-browser',
                        generated_at: new Date().toISOString(),
                    },
                    translations: translatedSlice.map(item => ({
                        speaker: String(item.speaker || ''),
                        translated_content: String(item.translated_content || ''),
                    })),
                };
                const put = await fetch(url, {
                    method: 'PUT',
                    headers,
                    body: JSON.stringify({
                        message: `Add translation cache ${scriptId} ${targetLanguage} ${model}`,
                        content: _base64Utf8(JSON.stringify(body, null, 2)),
                        branch,
                    }),
                });
                if (put.ok) written += 1;
                else errors.push(`${scriptId}: upload HTTP ${put.status} ${(await put.text()).slice(0, 160)}`);
            } catch (e) {
                errors.push(`${scriptId}: ${e.message}`);
            }
        }
        return { configured: true, written, skipped, errors };
    }

    // ── Core translation ─────────────────────────────────────────────────────

    /**
     * Translate an array of {speaker, content} objects.
     * Returns array of {speaker, translated_content}.
     * Progress callback: (current, total, speaker) => void.
     */
    async function translateDialogues(dialogues, opts = {}, onProgress = null) {
        const {
            targetLanguage = getPref('targetLanguage', 'Chinese Simplified'),
            apiType       = getPref('apiType', 'gemini'),
            apiKey        = getPref('apiKey', ''),
            apiBase       = getPref('apiBase', ''),
            baseModel     = getPref('baseModel', ''),
            method        = 'gpt',
        } = opts;

        // Free engine: Google Translate public endpoint (no API key, CORS-enabled)
        if (method === 'free') {
            const results = [];
            for (let i = 0; i < dialogues.length; i++) {
                if (onProgress) onProgress(i, dialogues.length, dialogues[i].speaker || '');
                const t = await _freeTranslateOne(dialogues[i], targetLanguage);
                results.push(t);
            }
            if (onProgress) onProgress(dialogues.length, dialogues.length, '');
            return results;
        }

        if (!apiKey) throw new Error('API key not set. Please configure it in Settings.');

        const CHUNK = 30; // lines per API request
        const results = [];

        for (let start = 0; start < dialogues.length; start += CHUNK) {
            const chunk = dialogues.slice(start, start + CHUNK);
            if (onProgress) onProgress(start, dialogues.length, chunk[0]?.speaker || '');

            const translated = await _translateChunk(chunk, targetLanguage, apiType, apiKey, apiBase, baseModel);
            results.push(...translated);
        }
        if (onProgress) onProgress(dialogues.length, dialogues.length, '');
        return results;
    }

    // ── Formatting-tag protection ────────────────────────────────────────────
    // FGO scripts contain bracketed control tags (e.g. [image berserkerLang09],
    // [line 3], [align center], [f xxl], [/f], [r]) that LLMs love to "translate"
    // — turning [image berserkerLang09] into [image beserker_language_9] etc.,
    // which then 404s when gaming.html builds the asset URL.
    //
    // We strip those tags out before the LLM sees the text and splice them back
    // into the translated string at the same relative positions. Tags that
    // contain readable text (e.g. ruby furigana [#漢字:かな]) are left as-is so
    // the LLM can still translate the visible characters.
    const _TAG_STRIP_RE = /\[image\s+[\w-]+\]|\[line\s+\d+\]|\[align(?:\s+\w+)?\]|\[f\s+[\w-]+\]|\[\/f\]|\[r\]/gi;
    function _stripFormatTags(text) {
        const tags = [];
        const stripped = (text || '').replace(_TAG_STRIP_RE, m => {
            const i = tags.length;
            tags.push(m);
            // Use a token unlikely to be touched by translation models. Keep it
            // ASCII so byte-level tokenisers don't split it; surround with a
            // unicode bracket pair that LLMs reliably copy verbatim.
            return `〘FT${i}〙`;
        });
        return { stripped, tags };
    }
    function _restoreFormatTags(translated, tags) {
        if (!tags.length) return translated;
        let out = translated || '';
        // Restore using a forgiving regex: tolerate stray spaces, half-width
        // brackets, or case shifts the LLM may introduce.
        for (let i = 0; i < tags.length; i++) {
            const re = new RegExp(`[〘\\[【]\\s*F\\s*T\\s*${i}\\s*[〙\\]】]`, 'i');
            if (re.test(out)) {
                out = out.replace(re, tags[i]);
            } else {
                // Token went missing — append at end so the asset still loads.
                out += tags[i];
            }
        }
        return out;
    }

    // Map UI language names → Google Translate codes
    const _LANG_MAP = {
        'chinese': 'zh-CN', 'chinese simplified': 'zh-CN', '简体中文': 'zh-CN', '中文': 'zh-CN',
        'chinese traditional': 'zh-TW', '繁體中文': 'zh-TW', '繁体中文': 'zh-TW',
        'english': 'en', 'japanese': 'ja', 'korean': 'ko',
        'french': 'fr', 'german': 'de', 'spanish': 'es', 'russian': 'ru',
    };

    // Translate a single Japanese string via Google Translate's public endpoint.
    // Returns the translated text, or the original on failure.
    async function _freeTranslateString(text, tl) {
        const t = (text || '').trim();
        if (!t) return text || '';
        const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=ja&tl=${tl}&dt=t&q=${encodeURIComponent(text)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        return (data?.[0] || []).map(seg => seg[0] || '').join('') || text;
    }

    async function _freeTranslateOne(dialogue, targetLanguage) {
        const speaker = dialogue.speaker || '';
        const content = dialogue.content || '';
        if (speaker === 'System' || !content.trim()) {
            return { speaker, content, translated_content: content };
        }
        const tl = _LANG_MAP[(targetLanguage || '').toLowerCase()] || 'zh-CN';
        try {
            // Split the dialogue on FGO formatting tags so Google Translate only
            // ever sees plain Japanese text segments. The tags themselves
            // (e.g. [image berserkerLang09], [line 3], [r], [/f]) are passed
            // through untouched; placeholder-based protection isn't reliable
            // against Google Translate because it freely "translates"/splits
            // CJK bracket pairs and ASCII tokens alike.
            //
            // We DO translate the visible reading half of ruby tags
            // [#漢字:かな] by splitting them into their parts.
            const segments = _splitForFreeTranslate(content);
            const out = [];
            for (const seg of segments) {
                if (seg.type === 'tag') {
                    out.push(seg.value);
                } else {
                    out.push(await _freeTranslateString(seg.value, tl));
                }
            }
            return { speaker, content, translated_content: out.join('') || content };
        } catch (e) {
            return { speaker, content, translated_content: `[Translation Error: ${e.message}]` };
        }
    }

    // Tokenise a dialogue body into [{type:'text'|'tag', value}] segments. Tags
    // matched by _TAG_STRIP_RE (and the [#kanji:reading] ruby tag, where we
    // keep only the visible kanji) are emitted as 'tag' segments and sent
    // through the translator verbatim. Everything else is translatable text.
    function _splitForFreeTranslate(text) {
        const RE = /\[image\s+[\w-]+\]|\[line\s+\d+\]|\[align(?:\s+\w+)?\]|\[f\s+[\w-]+\]|\[\/f\]|\[r\]|\[#([^\]:]+):[^\]]*\]/gi;
        const out = [];
        let last = 0, m;
        while ((m = RE.exec(text)) !== null) {
            if (m.index > last) out.push({ type: 'text', value: text.slice(last, m.index) });
            if (m[0].startsWith('[#') && m[1]) {
                // Ruby furigana: keep the visible kanji as plain text so it
                // gets translated; drop the reading.
                out.push({ type: 'text', value: m[1] });
            } else {
                out.push({ type: 'tag', value: m[0] });
            }
            last = m.index + m[0].length;
        }
        if (last < text.length) out.push({ type: 'text', value: text.slice(last) });
        return out;
    }

    async function _translateChunk(dialogues, lang, apiType, apiKey, apiBase, baseModel) {
        // Pre-strip formatting tags from each dialogue so the LLM doesn't mangle
        // them (e.g. rewriting [image berserkerLang09] → [image beserker_language_9]
        // and breaking the asset URL). Tags are spliced back into each
        // translated_content via _restoreFormatTags after parsing.
        const tagLists = dialogues.map(d => _stripFormatTags(d.content || ''));
        const sanitized = dialogues.map((d, i) => ({ speaker: d.speaker || '', content: tagLists[i].stripped }));

        // Build canonical numbered translation prompt (matches Python backend)
        const systemPrompt = `You are a professional translator for game dialogue.\nTranslate Japanese text from the game "Fate/Grand Order" into ${lang}.\nPreserve tone, character speech style, and terminology. Use standard transliterations for names (e.g., キリエライト → Mash Kyrielight/玛修·基列莱特, 藤丸立香 → Ritsuka Fujimaru/藤丸立香).\nOnly return the translated sentence in ${lang}, no extra text or formatting.\n`;

        let dialoguePrompt = `Please translate the following dialogues into ${lang}. You MUST follow these rules:\n1. Translate ALL dialogues\n2. For each dialogue, write its number followed by a colon (e.g., "1:", "2:", etc.)\n3. Write the translation on the next line\n4. Keep the translations in the same order as the original dialogues\n5. Translate both the speaker's name and their dialogue content\n6. For choices, translate both the choice number and content\n7. For system messages, keep them as is\n8. Use standard transliterations for character names (e.g., ライネス → Lainess/莱尼斯, グレイ → Gray/格雷)\n9. For katakana words (e.g., オーディール・コール), translate them to English (e.g., Order Call) and keep the English in the translation\n10. Tokens that look like 〘FT0〙, 〘FT1〙, … are placeholders for game formatting tags. Copy each one verbatim into the translation at the equivalent position. Do NOT translate, renumber, remove, or alter the digits inside.\n\nExample format:\n1:\n[Translated Speaker Name]: [Translation of first dialogue]\n2:\n[Translated Speaker Name]: [Translation of second dialogue]\n\nHere are the dialogues to translate:\n\n`;
        sanitized.forEach((d, i) => {
            dialoguePrompt += `${i + 1}:\nSpeaker: ${d.speaker || ''}\nContent: ${d.content || ''}\n\n`;
        });
        dialoguePrompt += '\nRemember to translate ALL dialogues and maintain the exact format shown in the example.';

        let rawTranslated = '';

        if (apiType === 'gemini') {
            const model = baseModel || 'gemini-2.5-flash';
            const base = apiBase || 'https://generativelanguage.googleapis.com/v1beta';
            const url = `${base.replace(/\/+$/, '')}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
            const body = {
                systemInstruction: { parts: [{ text: systemPrompt }] },
                contents: [{ role: 'user', parts: [{ text: dialoguePrompt }] }],
                generationConfig: { temperature: 0.7 },
            };
            let r;
            try {
                r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            } catch (e) {
                throw new Error(`Network/CORS error calling Gemini at ${base}. Check your API Base URL in Settings (should be https://generativelanguage.googleapis.com/v1beta) and your network. Original: ${e.message}`);
            }
            if (!r.ok) {
                const txt = await r.text();
                throw new Error(`Gemini error ${r.status}: ${txt.slice(0, 300)}`);
            }
            const data = await r.json();
            const cand = data.candidates?.[0];
            rawTranslated = (cand?.content?.parts || []).map(p => p.text || '').join('');
            if (!rawTranslated) {
                const block = data.promptFeedback?.blockReason;
                throw new Error(`Gemini returned no text${block ? ` (blocked: ${block})` : ''}`);
            }
        } else {
            // OpenAI-compatible
            const base = apiBase || 'https://api.openai.com';
            const model = baseModel || 'gpt-4o-mini';
            // Build chat completions URL; tolerate apiBase that already includes "/v1"
            const trimmed = base.replace(/\/+$/, '');
            const url = /\/v\d+$/.test(trimmed)
                ? `${trimmed}/chat/completions`
                : `${trimmed}/v1/chat/completions`;
            const authHeader = `Bearer ${apiKey}`;
            const body = {
                model,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: dialoguePrompt },
                ],
                temperature: 0.7,
            };
            let r;
            try {
                r = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': authHeader },
                    body: JSON.stringify(body),
                });
            } catch (e) {
                throw new Error(
                    `Network/CORS error calling ${url}. ` +
                    `Most OpenAI-compatible endpoints (DashScope/通义千问, OpenAI, DeepSeek, Moonshot, etc.) block browser requests (no CORS headers). ` +
                    `Workarounds:\n` +
                    `  1. Use the "Gemini" engine — Google's API has CORS enabled and works directly from browsers.\n` +
                    `  2. Use the "Free Engine" — Google Translate public endpoint, no key needed.\n` +
                    `  3. Self-host a tiny CORS proxy (e.g. Cloudflare Worker) and point API Base at it.\n` +
                    `  4. Run the original Flask version locally (python app.py) when you need DashScope/etc.\n` +
                    `Original error: ${e.message}`
                );
            }
            if (!r.ok) {
                const txt = await r.text();
                throw new Error(`API error ${r.status}: ${txt.slice(0, 300)}`);
            }
            const data = await r.json();
            rawTranslated = data.choices?.[0]?.message?.content || '';
        }

        // Parse numbered translations (matches Python _parse_numbered_translation_response)
        const parsed = _parseNumberedTranslations(rawTranslated, dialogues.length);
        return dialogues.map((d, i) => ({
            speaker: d.speaker || '',
            content: d.content || '',
            translated_content: _restoreFormatTags(
                parsed[i] || '[Translation Error: Missing translation]',
                tagLists[i].tags,
            ),
        }));
    }

    /**
     * Parse a numbered LLM response of the form:
     *   1:
     *   [Speaker]: translated text
     *   2:
     *   [Speaker]: translated text
     * Returns an array of length expectedCount (padded/truncated).
     */
    function _parseNumberedTranslations(response, expectedCount) {
        const translations = [];
        let current = [];
        let currentNum = null;
        for (let line of response.split('\n')) {
            line = line.trim();
            if (!line) continue;
            const m = line.match(/^(\d+):\s*(.*)$/);
            if (m) {
                if (current.length) {
                    translations.push(current.join('\n'));
                    current = [];
                }
                currentNum = parseInt(m[1], 10);
                if (m[2].trim()) current.push(m[2].trim());
            } else if (currentNum !== null) {
                current.push(line);
            }
        }
        if (current.length) translations.push(current.join('\n'));

        if (translations.length !== expectedCount) {
            console.warn(`Translation count mismatch: got ${translations.length}, expected ${expectedCount}`);
            if (!translations.length) {
                // Fallback: take all non-numbered lines
                const lines = response.split('\n').map(l => l.trim()).filter(l => l && !/^\d+:/.test(l));
                translations.push(...lines);
            }
            while (translations.length < expectedCount) translations.push('[Translation Error: Missing translation]');
            translations.length = expectedCount;
        }
        return translations;
    }

    return { loadPrefs, savePrefs, getPref, translateDialogues, saveTranslationCache };
})();
