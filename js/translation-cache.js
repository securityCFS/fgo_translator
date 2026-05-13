window.TranslationCache = (() => {
    const DEFAULT_PROMPT_VERSION = 'fgo-v1';

    function loadPrefs() {
        try {
            return JSON.parse(localStorage.getItem('fgo_translator_prefs') || '{}');
        } catch {
            return {};
        }
    }

    function getBaseUrl() {
        const prefs = loadPrefs();
        return String(
            window.TRANSLATION_CACHE_BASE_URL ||
            prefs.translationCacheBaseUrl ||
            prefs.translation_cache_base_url ||
            ''
        ).replace(/\/+$/, '');
    }

    function sanitizePathSegment(value) {
        const safe = String(value || '')
            .trim()
            .replace(/[^A-Za-z0-9._-]+/g, '_')
            .replace(/^[._]+|[._]+$/g, '');
        return safe || 'unknown';
    }

    function normalizeTargetLanguage(value) {
        const raw = String(value || '').trim();
        const lowered = raw.toLowerCase();
        const mapping = {
            'chinese': 'zh-CN',
            'chinese simplified': 'zh-CN',
            'chinese (simplified)': 'zh-CN',
            'simplified chinese': 'zh-CN',
            'zh-cn': 'zh-CN',
            '中文': 'zh-CN',
            '简体中文': 'zh-CN',
            'chinese traditional': 'zh-TW',
            'chinese (traditional)': 'zh-TW',
            'traditional chinese': 'zh-TW',
            'zh-tw': 'zh-TW',
            '繁體中文': 'zh-TW',
            '繁体中文': 'zh-TW',
            'english': 'en',
            'en': 'en',
            'japanese': 'ja',
            'ja': 'ja',
            'korean': 'ko',
            'ko': 'ko',
        };
        return mapping[lowered] || sanitizePathSegment(raw);
    }

    function normalizeProvider(apiType, model) {
        const raw = String(apiType || 'openai').toLowerCase();
        const modelLower = String(model || '').toLowerCase();
        if (raw === 'gemini' || modelLower.includes('gemini')) return 'gemini';
        if (modelLower.includes('deepseek')) return 'deepseek';
        if (modelLower.includes('qwen') || modelLower.includes('qwq')) return 'qwen';
        if (modelLower.includes('claude')) return 'claude';
        if (raw === 'openai' || raw === 'custom') return raw;
        return sanitizePathSegment(raw);
    }

    async function sha256Hex(text) {
        const bytes = new TextEncoder().encode(text);
        const hash = await crypto.subtle.digest('SHA-256', bytes);
        return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
    }

    async function canonicalSourceHash(dialogues) {
        const payload = (dialogues || []).map(d => ({
            content: String(d.content || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
            speaker: String(d.speaker || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
        }));
        return sha256Hex(JSON.stringify(payload));
    }

    function relativePath(key) {
        return [
            'v1',
            sanitizePathSegment(String(key.sourceRegion || 'JP').toUpperCase()),
            sanitizePathSegment(key.scriptId),
            sanitizePathSegment(key.sourceHash),
            sanitizePathSegment(key.targetLanguage),
            sanitizePathSegment(key.provider),
            sanitizePathSegment(key.model),
            `${sanitizePathSegment(key.promptVersion || DEFAULT_PROMPT_VERSION)}.json`,
        ].join('/');
    }

    function validate(data, key, expectedDialogueCount) {
        if (!data || data.schema_version !== 1 || data.trusted_generation !== true) return null;
        if (String(data.script_id) !== String(key.scriptId)) return null;
        if (String(data.source_region) !== String(key.sourceRegion)) return null;
        if (String(data.source_hash) !== String(key.sourceHash)) return null;
        if (String(data.target_language) !== String(key.targetLanguage)) return null;
        if (String(data.provider) !== String(key.provider)) return null;
        if (String(data.model) !== String(key.model)) return null;
        if (String(data.prompt_version) !== String(key.promptVersion || DEFAULT_PROMPT_VERSION)) return null;
        if (!Array.isArray(data.translations)) return null;
        if (Number(data.dialogue_count) !== expectedDialogueCount || data.translations.length !== expectedDialogueCount) return null;
        if (data.translations.some(t => String(t.translated_content || '').includes('[Translation Error:'))) return null;
        return data.translations;
    }

    async function readScript({scriptId, sourceRegion, dialogues, targetLanguage, apiType, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        const baseUrl = getBaseUrl();
        if (!baseUrl) return null;
        const sourceHash = await canonicalSourceHash(dialogues);
        const key = {
            scriptId: String(scriptId),
            sourceRegion: String(sourceRegion || 'JP').toUpperCase(),
            sourceHash,
            targetLanguage: normalizeTargetLanguage(targetLanguage),
            provider: normalizeProvider(apiType, model),
            model: model || 'unknown',
            promptVersion,
        };
        try {
            const response = await fetch(`${baseUrl}/${relativePath(key)}`, {cache: 'force-cache'});
            if (!response.ok) return null;
            return validate(await response.json(), key, dialogues.length);
        } catch {
            return null;
        }
    }

    return {
        canonicalSourceHash,
        normalizeProvider,
        normalizeTargetLanguage,
        readScript,
    };
})();
