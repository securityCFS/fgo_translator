window.TranslationCache = (() => {
    const DEFAULT_PROMPT_VERSION = 'fgo-v1';
    const DEFAULT_UPLOAD_URL = 'https://fgo-translator-cache.710244487.workers.dev/upload';

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

    function getUploadUrl() {
        const prefs = loadPrefs();
        return String(
            window.TRANSLATION_CACHE_UPLOAD_URL ||
            prefs.translationCacheUploadUrl ||
            prefs.translation_cache_upload_url ||
            DEFAULT_UPLOAD_URL
        ).trim();
    }

    function getRepoInfo() {
        const prefs = loadPrefs();
        const configuredRepo = String(
            window.TRANSLATION_CACHE_REPO ||
            prefs.translationCacheRepo ||
            prefs.translation_cache_repo ||
            ''
        ).trim();
        const configuredBranch = String(
            window.TRANSLATION_CACHE_BRANCH ||
            prefs.translationCacheBranch ||
            prefs.translation_cache_branch ||
            'main'
        ).trim() || 'main';
        if (configuredRepo) {
            return { repo: configuredRepo, branch: configuredBranch };
        }

        const match = getBaseUrl().match(/^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)(?:\/|$)/);
        if (!match) return null;
        return {
            repo: `${match[1]}/${match[2]}`,
            branch: decodeURIComponent(match[3]),
        };
    }

    function encodeGitHubPath(path) {
        return String(path || '').split('/').map(part => encodeURIComponent(part)).join('/');
    }

    async function listGitHubDirectory(path) {
        const repoInfo = getRepoInfo();
        if (!repoInfo) return [];
        try {
            const response = await fetch(
                `https://api.github.com/repos/${repoInfo.repo}/contents/${encodeGitHubPath(path)}?ref=${encodeURIComponent(repoInfo.branch)}`,
                { cache: 'force-cache' }
            );
            if (!response.ok) return [];
            const data = await response.json();
            return Array.isArray(data) ? data.filter(item => item && typeof item === 'object') : [];
        } catch {
            return [];
        }
    }

    async function readJsonUrl(url) {
        if (!url) return null;
        try {
            const response = await fetch(url, { cache: 'force-cache' });
            if (!response.ok) return null;
            const data = await response.json();
            return data && typeof data === 'object' ? data : null;
        } catch {
            return null;
        }
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

    async function readByKey(key, expectedDialogueCount) {
        const baseUrl = getBaseUrl();
        if (!baseUrl) return null;
        try {
            const response = await fetch(`${baseUrl}/${relativePath(key)}`, {cache: 'force-cache'});
            if (!response.ok) return null;
            return validate(await response.json(), key, expectedDialogueCount);
        } catch {
            return null;
        }
    }

    async function readScript({scriptId, sourceRegion, dialogues, targetLanguage, apiType, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        return readScriptVersion({
            scriptId,
            sourceRegion,
            dialogues,
            targetLanguage,
            provider: normalizeProvider(apiType, model),
            model: model || 'unknown',
            promptVersion,
        });
    }

    async function readScriptVersion({scriptId, sourceRegion, dialogues, targetLanguage, provider, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        if (!Array.isArray(dialogues)) return null;
        const key = {
            scriptId: String(scriptId),
            sourceRegion: String(sourceRegion || 'JP').toUpperCase(),
            sourceHash: await canonicalSourceHash(dialogues),
            targetLanguage: normalizeTargetLanguage(targetLanguage),
            provider: provider || 'unknown',
            model: model || 'unknown',
            promptVersion,
        };
        return readByKey(key, dialogues.length);
    }

    async function listOptions({scriptId, sourceRegion, dialogues, targetLanguage, promptVersion}) {
        if (!Array.isArray(dialogues) || !dialogues.length) return [];
        const sourceHash = await canonicalSourceHash(dialogues);
        const normalizedTarget = normalizeTargetLanguage(targetLanguage);
        const basePath = [
            'v1',
            sanitizePathSegment(String(sourceRegion || 'JP').toUpperCase()),
            sanitizePathSegment(scriptId),
            sanitizePathSegment(sourceHash),
            sanitizePathSegment(normalizedTarget),
        ].join('/');
        const options = [];

        for (const providerItem of await listGitHubDirectory(basePath)) {
            if (providerItem.type !== 'dir') continue;
            const providerDir = providerItem.name || '';
            const providerPath = `${basePath}/${providerDir}`;
            for (const modelItem of await listGitHubDirectory(providerPath)) {
                if (modelItem.type !== 'dir') continue;
                const modelDir = modelItem.name || '';
                const modelPath = `${providerPath}/${modelDir}`;
                for (const promptItem of await listGitHubDirectory(modelPath)) {
                    const name = String(promptItem.name || '');
                    if (promptItem.type !== 'file' || !name.endsWith('.json')) continue;
                    const payload = await readJsonUrl(promptItem.download_url);
                    if (!payload) continue;
                    const provider = String(payload.provider || providerDir);
                    const model = String(payload.model || modelDir);
                    const resolvedPrompt = String(payload.prompt_version || name.slice(0, -5));
                    if (promptVersion && resolvedPrompt !== String(promptVersion)) continue;
                    const key = {
                        scriptId: String(scriptId),
                        sourceRegion: String(sourceRegion || 'JP').toUpperCase(),
                        sourceHash,
                        targetLanguage: normalizedTarget,
                        provider,
                        model,
                        promptVersion: resolvedPrompt,
                    };
                    const translations = validate(payload, key, dialogues.length);
                    if (!translations) continue;
                    options.push({
                        id: `${provider}||${model}||${resolvedPrompt}`,
                        provider,
                        model,
                        promptVersion: resolvedPrompt,
                        label: `${provider} / ${model} / ${resolvedPrompt}`,
                        dialogueCount: translations.length,
                        generatedAt: payload.generator?.generated_at || '',
                    });
                }
            }
        }

        return options.sort((a, b) => a.label.localeCompare(b.label));
    }

    function normalizeTranslations(translations, sourceDialogues) {
        if (!Array.isArray(translations) || translations.length !== sourceDialogues.length) return null;
        const cleaned = [];
        for (let i = 0; i < translations.length; i++) {
            const item = translations[i] || {};
            const text = String(item.translated_content || '').trim();
            if (!text || text.includes('[Translation Error:')) return null;
            cleaned.push({
                speaker: String(item.speaker || sourceDialogues[i]?.speaker || '').trim(),
                translated_content: text,
            });
        }
        return cleaned;
    }

    async function uploadScript({scriptId, sourceRegion, dialogues, translations, targetLanguage, apiType, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        const uploadUrl = getUploadUrl();
        if (!uploadUrl || !Array.isArray(dialogues) || !dialogues.length || !model) return null;
        const cleanedTranslations = normalizeTranslations(translations, dialogues);
        if (!cleanedTranslations) return null;
        const sourceHash = await canonicalSourceHash(dialogues);
        const payload = {
            script_id: String(scriptId),
            source_region: String(sourceRegion || 'JP').toUpperCase(),
            source_hash: sourceHash,
            target_language: targetLanguage,
            api_type: apiType || 'openai',
            model,
            prompt_version: promptVersion,
            translations: cleanedTranslations,
        };
        try {
            const response = await fetch(uploadUrl, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        }
    }

    async function uploadScripts({scriptIds, scriptDialogueCounts, allDialogues, translatedDialogues, sourceRegion, targetLanguage, apiType, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        if (!Array.isArray(scriptIds) || !Array.isArray(scriptDialogueCounts)) return [];
        const results = [];
        let offset = 0;
        for (let i = 0; i < scriptIds.length; i++) {
            const count = Number(scriptDialogueCounts[i] || 0);
            const source = (allDialogues || []).slice(offset, offset + count);
            const translated = (translatedDialogues || []).slice(offset, offset + count);
            offset += count;
            const result = await uploadScript({
                scriptId: scriptIds[i],
                sourceRegion,
                dialogues: source,
                translations: translated,
                targetLanguage,
                apiType,
                model,
                promptVersion,
            });
            results.push({scriptId: String(scriptIds[i]), ok: !!result, result});
        }
        return results;
    }

    return {
        canonicalSourceHash,
        normalizeProvider,
        normalizeTargetLanguage,
        listOptions,
        readScript,
        readScriptVersion,
        uploadScript,
        uploadScripts,
    };
})();
