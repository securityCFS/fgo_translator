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

    // Map UI language names → Google Translate codes
    const _LANG_MAP = {
        'chinese': 'zh-CN', 'chinese simplified': 'zh-CN', '简体中文': 'zh-CN', '中文': 'zh-CN',
        'chinese traditional': 'zh-TW', '繁體中文': 'zh-TW', '繁体中文': 'zh-TW',
        'english': 'en', 'japanese': 'ja', 'korean': 'ko',
        'french': 'fr', 'german': 'de', 'spanish': 'es', 'russian': 'ru',
    };

    async function _freeTranslateOne(dialogue, targetLanguage) {
        const speaker = dialogue.speaker || '';
        const content = dialogue.content || '';
        if (speaker === 'System' || !content.trim()) {
            return { speaker, translated_content: content };
        }
        const tl = _LANG_MAP[(targetLanguage || '').toLowerCase()] || 'zh-CN';
        const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=ja&tl=${tl}&dt=t&q=${encodeURIComponent(content)}`;
        try {
            const r = await fetch(url);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const data = await r.json();
            // data[0] is array of [translatedSegment, originalSegment, ...]
            const translated = (data?.[0] || []).map(seg => seg[0] || '').join('');
            return { speaker, translated_content: translated || content };
        } catch (e) {
            return { speaker, translated_content: `[Translation Error: ${e.message}]` };
        }
    }

    async function _translateChunk(dialogues, lang, apiType, apiKey, apiBase, baseModel) {
        const lines = dialogues.map(d =>
            d.speaker ? `${d.speaker}：${d.content}` : d.content
        ).join('\n');

        const systemPrompt = `You are a professional translator for the mobile game Fate/Grand Order. Translate the following Japanese dialogue lines into ${lang}. Preserve speaker names, honorifics, and game-specific terms. Output ONLY the translated lines in the same order, one per line, in the format "Speaker：Translation" or just "Translation" if there is no speaker. Do not add explanations.`;

        const userPrompt = lines;

        let rawTranslated = '';

        if (apiType === 'gemini') {
            const model = baseModel || 'gemini-2.0-flash';
            const base = apiBase || 'https://generativelanguage.googleapis.com/v1beta';
            const url = `${base}/models/${model}:generateContent?key=${apiKey}`;
            const body = {
                contents: [{ parts: [{ text: `${systemPrompt}\n\n${userPrompt}` }] }],
                generationConfig: { temperature: 0.3, maxOutputTokens: 4096 },
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
            rawTranslated = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
        } else {
            // OpenAI-compatible
            const base = apiBase || 'https://api.openai.com';
            const model = baseModel || 'gpt-4o-mini';
            const url = `${base.replace(/\/+$/, '')}/v1/chat/completions`;
            const authHeader = `Bearer ${apiKey}`;
            const body = {
                model,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt },
                ],
                temperature: 0.3,
                max_tokens: 4096,
            };
            let r;
            try {
                r = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': authHeader },
                    body: JSON.stringify(body),
                });
            } catch (e) {
                throw new Error(`Network/CORS error calling ${url}. Many OpenAI-compatible endpoints block browser requests (no CORS). Try Gemini instead, or use a CORS-enabled endpoint. Original: ${e.message}`);
            }
            if (!r.ok) {
                const txt = await r.text();
                throw new Error(`API error ${r.status}: ${txt.slice(0, 300)}`);
            }
            const data = await r.json();
            rawTranslated = data.choices?.[0]?.message?.content || '';
        }

        // Parse translated lines back to {speaker, translated_content}
        const outLines = rawTranslated.split('\n').filter(l => l.trim());
        return dialogues.map((d, i) => {
            const tLine = outLines[i] || '';
            const colonIdx = tLine.indexOf('：');
            if (colonIdx > 0) {
                return { speaker: tLine.slice(0, colonIdx).trim(), translated_content: tLine.slice(colonIdx + 1).trim() };
            }
            return { speaker: d.speaker || '', translated_content: tLine.trim() || d.content };
        });
    }

    return { loadPrefs, savePrefs, getPref, translateDialogues };
})();
