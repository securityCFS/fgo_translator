/**
 * api.js — Atlas Academy API helpers (all CORS-friendly, no proxy needed)
 */
const AA = (() => {
    const BASE = 'https://api.atlasacademy.io';
    const CDN  = 'https://static.atlasacademy.io';
    const DEFAULT_TIMEOUT_MS = 30000;
    const BULK_EXPORT_TIMEOUT_MS = 90000;
    const DETAIL_TIMEOUT_MS = 45000;
    const ACTIVITY_ENRICH_LIMIT = 12;

    const REGIONS = ['JP','NA','CN','TW','KR'];
    function norm(r) { return REGIONS.includes((r||'').toUpperCase()) ? r.toUpperCase() : 'JP'; }

    // Simple in-memory cache keyed by URL
    const _cache = new Map();
    async function get(url, opts = {}) {
        if (_cache.has(url)) return _cache.get(url);
        const timeoutMs = opts.timeoutMs || DEFAULT_TIMEOUT_MS;
        const ctrl = new AbortController();
        let abortFromCaller;
        const timeoutId = window.setTimeout(() => ctrl.abort(), timeoutMs);
        if (opts.signal) {
            abortFromCaller = () => ctrl.abort();
            if (opts.signal.aborted) ctrl.abort();
            else opts.signal.addEventListener('abort', abortFromCaller, { once: true });
        }
        try {
            const r = await fetch(url, { signal: ctrl.signal });
            if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
            const data = await r.json();
            _cache.set(url, data);
            return data;
        } catch (e) {
            if (e && e.name === 'AbortError') {
                const err = new Error(`Atlas Academy request timed out after ${Math.round(timeoutMs / 1000)}s.`);
                err.code = 'ATLAS_TIMEOUT';
                err.url = url;
                err.timeoutMs = timeoutMs;
                throw err;
            }
            throw e;
        } finally {
            window.clearTimeout(timeoutId);
            if (opts.signal && abortFromCaller) opts.signal.removeEventListener('abort', abortFromCaller);
        }
    }

    // ── Search ──────────────────────────────────────────────────────────────

    /**
     * Search wars by name or ID.
     * Uses the bulk export endpoint (same as Flask backend) and filters locally.
     */
    async function searchWar(name = '', region = 'JP', limit = 50) {
        region = norm(region);
        limit = Math.min(Math.max(limit, 1), 200);
        name = (name || '').trim();
        try {
            // If numeric ID, fetch directly
            if (/^\d+$/.test(name)) {
                const w = await get(`${BASE}/nice/${region}/war/${name}`);
                return w ? [_normalizeWar(w, region)] : [];
            }
            const wars = await get(`${BASE}/export/${region}/nice_war.json`);
            if (!Array.isArray(wars)) return [];
            const nl = name.toLowerCase();
            const filtered = nl
                ? wars.filter(w => (w.name || '').toLowerCase().includes(nl)
                    || (w.longName || '').toLowerCase().includes(nl)
                    || (w.eventName || '').toLowerCase().includes(nl))
                : wars;
            // Sort by eventId startedAt desc, fallback war id desc
            let evStart = {};
            try {
                const evs = await get(`${BASE}/export/${region}/basic_event.json`);
                if (Array.isArray(evs)) evs.forEach(e => { if (e.id && e.startedAt) evStart[e.id] = e.startedAt; });
            } catch {}
            const sorted = [...filtered].sort((a, b) => {
                const ta = evStart[a.eventId] || a.startedAt || a.id || 0;
                const tb = evStart[b.eventId] || b.startedAt || b.id || 0;
                return tb - ta;
            });
            return sorted.slice(0, limit).map(w => _normalizeWar(w, region));
        } catch (e) { console.error('searchWar error', e); return []; }
    }

    function _normalizeWar(w, region) {
        return {
            id: String(w.id), name: w.name || w.longName || '',
            longName: w.longName || w.name || '',
            banner: w.banner || w.headerImage || w.icon || '',
            eventName: w.eventName || '', age: w.age || '',
            startedAt: w.startedAt, endedAt: w.endedAt,
            region, itemKind: 'war',
        };
    }

    async function _eventTimeline(region) {
        const started = {};
        const ended = {};
        try {
            const evs = await get(`${BASE}/export/${region}/basic_event.json`, { timeoutMs: BULK_EXPORT_TIMEOUT_MS });
            if (Array.isArray(evs)) {
                evs.forEach(e => {
                    if (e.id && (e.startedAt || e.noticeAt)) started[e.id] = e.startedAt || e.noticeAt;
                    if (e.id && e.endedAt) ended[e.id] = e.endedAt;
                });
            }
        } catch {}
        return { started, ended };
    }

    function _applyWarTimeline(war, timeline) {
        const evId = war.eventId || 0;
        if (!evId) return { ...war };
        return {
            ...war,
            startedAt: war.startedAt || timeline.started[evId],
            endedAt: war.endedAt || timeline.ended[evId],
        };
    }

    async function _mapLimit(items, limit, mapper) {
        const results = [];
        for (let i = 0; i < items.length; i += limit) {
            const chunk = items.slice(i, i + limit);
            results.push(...await Promise.all(chunk.map(mapper)));
        }
        return results;
    }

    async function _enrichWarForDisplay(war, region) {
        let enriched = war;
        try {
            const detail = await get(`${BASE}/nice/${region}/war/${war.id}`, { timeoutMs: DETAIL_TIMEOUT_MS });
            enriched = {
                ...war,
                ...detail,
                eventName: war.eventName || detail.eventName,
                startedAt: war.startedAt || detail.startedAt,
                endedAt: war.endedAt || detail.endedAt,
            };
        } catch (e) {
            console.warn('war detail enrichment failed:', war.id, e);
        }
        if (!enriched.banner && !enriched.noticeBanner && war.eventId) {
            try {
                const event = await get(`${BASE}/nice/${region}/event/${war.eventId}`, { timeoutMs: DETAIL_TIMEOUT_MS });
                enriched = {
                    ...enriched,
                    eventName: enriched.eventName || event.name,
                    banner: event.banner || event.noticeBanner || enriched.banner || '',
                    noticeBanner: event.noticeBanner || enriched.noticeBanner || '',
                    startedAt: enriched.startedAt || event.startedAt || event.noticeAt,
                    endedAt: enriched.endedAt || event.endedAt,
                };
            } catch (e) {
                console.warn('event detail enrichment failed:', war.eventId, e);
            }
        }
        return enriched;
    }

    /**
     * Search events by name or ID.
     * Uses the bulk export endpoint and filters locally.
     */
    async function searchEvent(name = '', region = 'JP', limit = 50) {
        region = norm(region);
        limit = Math.min(Math.max(limit, 1), 200);
        name = (name || '').trim();
        try {
            if (/^\d+$/.test(name)) {
                const ev = await get(`${BASE}/nice/${region}/event/${name}`);
                return ev ? [_normalizeEvent(ev, region)] : [];
            }
            const events = await get(`${BASE}/export/${region}/nice_event.json`);
            if (!Array.isArray(events)) return [];
            const nl = name.toLowerCase();
            let filtered = nl
                ? events.filter(e => (e.name || '').toLowerCase().includes(nl))
                : events.filter(e => e.warIds && e.warIds.length > 0); // prefer events with war content
            filtered = [...filtered].sort((a, b) =>
                ((b.startedAt || b.noticeAt || b.id || 0) - (a.startedAt || a.noticeAt || a.id || 0)));
            return filtered.slice(0, limit).map(e => _normalizeEvent(e, region));
        } catch (e) { console.error('searchEvent error', e); return []; }
    }

    function _normalizeEvent(e, region) {
        return {
            id: String(e.id), name: e.name || '',
            banner: e.banner || e.noticeBanner || '',
            noticeBanner: e.noticeBanner || '',
            type: e.type || '', startedAt: e.startedAt, endedAt: e.endedAt,
            warIds: e.warIds || [],
            region, itemKind: 'event',
        };
    }

    /**
     * Get latest wars (uses nice_war.json export, sorted by event start desc).
     */
    async function latestWars(region = 'JP', limit = 50) {
        region = norm(region);
        try {
            const wars = await get(`${BASE}/export/${region}/basic_war.json`, { timeoutMs: BULK_EXPORT_TIMEOUT_MS });
            if (!Array.isArray(wars)) return [];
            const timeline = await _eventTimeline(region);
            const sorted = wars.map(w => _applyWarTimeline(w, timeline)).sort((a, b) => {
                const ta = a.startedAt || a.id || 0;
                const tb = b.startedAt || b.id || 0;
                return tb - ta;
            });
            const selected = sorted.slice(0, limit);
            const enriched = await Promise.all(selected.map((war, idx) =>
                idx < ACTIVITY_ENRICH_LIMIT ? _enrichWarForDisplay(war, region) : war
            ));
            return enriched.map(w => _normalizeWar(w, region));
        } catch (e) {
            console.warn('latestWars export failed:', e);
            throw e;
        }
    }

    /**
     * Get latest events (uses nice_event.json export, sorted by startedAt desc).
     */
    async function latestEvents(region = 'JP', limit = 50) {
        region = norm(region);
        try {
            const events = await get(`${BASE}/export/${region}/nice_event.json`);
            if (!Array.isArray(events)) return [];
            const sorted = [...events]
                .filter(e => e.warIds && e.warIds.length > 0)
                .sort((a, b) => ((b.startedAt || b.id || 0) - (a.startedAt || a.id || 0)));
            return sorted.slice(0, limit).map(e => _normalizeEvent(e, region));
        } catch (e) {
            console.warn('latestEvents export failed:', e);
            throw e;
        }
    }

    function _normalizePhaseScripts(phaseScripts = []) {
        return (phaseScripts || []).map(ps => ({
            phase: ps.phase,
            scripts: (ps.scripts || []).map(s => ({
                scriptId: String(s.scriptId || s.id || s),
                script: s.script || '',
            })).filter(s => s.scriptId && s.scriptId !== '0'),
        })).filter(ps => ps.scripts.length > 0);
    }

    function _scriptIdsFromPhaseScripts(phaseScripts = []) {
        const seen = new Set();
        const ids = [];
        for (const ps of phaseScripts || []) {
            for (const script of ps.scripts || []) {
                const id = String(script.scriptId || script.id || script || '');
                if (!id || id === '0' || seen.has(id)) continue;
                seen.add(id);
                ids.push(id);
            }
        }
        return ids;
    }

    function _normalizeQuest(q, spot, war, region, mapById = new Map()) {
        const phaseScripts = _normalizePhaseScripts(q.phaseScripts || []);
        const scriptIds = _scriptIdsFromPhaseScripts(phaseScripts);
        const mapId = spot && spot.mapId != null ? String(spot.mapId) : '';
        const map = mapById.get(mapId) || {};
        return {
            id: String(q.id),
            name: q.name || '',
            type: q.type || '',
            flags: q.flags || [],
            phases: q.phases || [],
            phasesNoBattle: q.phasesNoBattle || [],
            phasesWithEnemies: q.phasesWithEnemies || [],
            phaseScripts,
            scriptIds,
            scriptCount: scriptIds.length,
            hasDialogueScript: scriptIds.length > 0,
            spotId: String(q.spotId || (spot && spot.id) || ''),
            spotName: q.spotName || (spot && spot.name) || '',
            mapId,
            mapImage: map.mapImage || '',
            spotX: spot && spot.x,
            spotY: spot && spot.y,
            openedAt: q.openedAt,
            closedAt: q.closedAt,
            consumeType: q.consumeType || '',
            consume: q.consume,
            region,
            warId: String(q.warId || war.id || ''),
            warName: war.name || war.longName || '',
            warLongName: q.warLongName || war.longName || war.name || '',
        };
    }

    /**
     * Latest quests (used as "Tasks" tab). Fetches recent wars and flattens their quests.
     */
    async function latestTasks(region = 'JP', limit = 50) {
        region = norm(region);
        try {
            limit = Math.min(Math.max(limit || 50, 1), 200);
            const latestRows = await get(`${BASE}/basic/${region}/quest/phase/latestEnemyData`);
            const uniqueRows = [];
            const seenQuestIds = new Set();
            for (const row of Array.isArray(latestRows) ? latestRows : []) {
                const id = String(row && row.id || '');
                if (!id || seenQuestIds.has(id)) continue;
                seenQuestIds.add(id);
                uniqueRows.push(row);
            }

            const warIds = [...new Set(uniqueRows.map(row => String(row.warId || '')).filter(Boolean))];
            const warQuestById = new Map();
            await Promise.all(warIds.map(async warId => {
                try {
                    const { quests } = await getWarQuests(warId, region);
                    quests.forEach(q => warQuestById.set(String(q.id), q));
                } catch (e) { console.warn('latestTasks war lookup failed:', warId, e); }
            }));

            let hiddenNoScriptCount = 0;
            const tasks = [];
            for (const row of uniqueRows) {
                const quest = warQuestById.get(String(row.id));
                if (!quest || !quest.hasDialogueScript) {
                    hiddenNoScriptCount += 1;
                    continue;
                }
                tasks.push({
                    ...quest,
                    itemKind: 'task',
                    latestPhase: row.phase,
                    latestOpenedAt: row.openedAt,
                    openedAt: quest.openedAt || row.openedAt,
                });
            }

            if (tasks.length < limit) {
                const seen = new Set(tasks.map(q => String(q.id)));
                const wars = await latestWars(region, 20);
                for (const war of wars) {
                    if (tasks.length >= limit) break;
                    try {
                        const { quests } = await getWarQuests(war.id, region);
                        quests
                            .filter(q => q.hasDialogueScript && !seen.has(String(q.id)))
                            .sort((a, b) => (b.openedAt || 0) - (a.openedAt || 0))
                            .forEach(q => {
                                if (tasks.length >= limit) return;
                                seen.add(String(q.id));
                                tasks.push({ ...q, itemKind: 'task' });
                            });
                    } catch (e) { console.warn('latestTasks fill failed:', war.id, e); }
                }
            }

            return {
                tasks: tasks.slice(0, limit),
                hiddenNoScriptCount,
                scannedCount: uniqueRows.length,
            };
        } catch (e) {
            console.error('latestTasks', e);
            throw e;
        }
    }

    // ── Quest detail ─────────────────────────────────────────────────────────

    /**
     * Get quests belonging to a war.
     */
    async function getWarQuests(warId, region = 'JP') {
        region = norm(region);
        // Use nice war which has full quest list and phase info
        const url = `${BASE}/nice/${region}/war/${warId}`;
        let war;
        try {
            war = await get(url);
        } catch (e) {
            console.warn('nice war lookup failed, trying latest quest fallback:', warId, e);
            return getWarQuestsFromLatestRows(warId, region, e);
        }
        const maps = war.maps || [];
        const mapById = new Map(maps.map(m => [String(m.id), m]));
        const quests = (war.spots || []).flatMap(sp => (sp.quests || []).map(q =>
            _normalizeQuest(q, sp, war, region, mapById)
        ));
        return {
            quests,
            warInfo: {
                id: String(warId),
                name: war.name || '',
                longName: war.longName || '',
                banner: war.banner || '',
                mapImage: (maps[0] && maps[0].mapImage) || '',
            }
        };
    }

    async function getWarQuestsFromLatestRows(warId, region = 'JP', originalError = null) {
        const warKey = String(warId);
        let basicWar = {};
        try {
            basicWar = await get(`${BASE}/basic/${region}/war/${encodeURIComponent(warKey)}`, { timeoutMs: DETAIL_TIMEOUT_MS });
        } catch (e) {
            console.warn('basic war fallback failed:', warId, e);
        }
        const latestRows = await get(`${BASE}/basic/${region}/quest/phase/latestEnemyData`, { timeoutMs: BULK_EXPORT_TIMEOUT_MS });
        const byQuestId = new Map();
        for (const row of Array.isArray(latestRows) ? latestRows : []) {
            if (String(row && row.warId || '') !== warKey) continue;
            const questId = String(row.id || '');
            if (!questId || byQuestId.has(questId)) continue;
            byQuestId.set(questId, row);
        }
        if (!byQuestId.size) {
            throw originalError || new Error(`No fallback quests found for war ${warKey}`);
        }

        const fallbackWar = {
            id: basicWar.id || warKey,
            name: basicWar.name || '',
            longName: basicWar.longName || basicWar.name || '',
            banner: basicWar.banner || '',
            eventName: basicWar.eventName || '',
        };
        const quests = [];
        await _mapLimit([...byQuestId.entries()], 6, async ([questId, row]) => {
            try {
                const quest = await get(`${BASE}/nice/${region}/quest/${encodeURIComponent(questId)}`, { timeoutMs: DETAIL_TIMEOUT_MS });
                quests.push(_normalizeQuest(quest, {
                    id: row.spotId,
                    name: row.spotName,
                    mapId: row.mapId,
                }, fallbackWar, region, new Map()));
            } catch (e) {
                console.warn('latest quest fallback detail failed:', questId, e);
            }
        });
        quests.sort((a, b) => (b.openedAt || 0) - (a.openedAt || 0) || Number(b.id) - Number(a.id));
        return {
            quests,
            warInfo: {
                id: warKey,
                name: fallbackWar.name || `War ${warKey}`,
                longName: fallbackWar.longName || fallbackWar.name || '',
                banner: fallbackWar.banner || '',
                mapImage: '',
            }
        };
    }

    /**
     * Get quests belonging to an event (resolves event → warIds → quests).
     */
    async function getEventQuests(eventId, region = 'JP') {
        region = norm(region);
        let ev;
        try {
            ev = await get(`${BASE}/nice/${region}/event/${eventId}`, { timeoutMs: DETAIL_TIMEOUT_MS });
        } catch (e) {
            console.warn('nice event lookup failed, trying basic event fallback:', eventId, e);
            ev = await get(`${BASE}/basic/${region}/event/${eventId}`, { timeoutMs: DETAIL_TIMEOUT_MS });
        }
        const warIds = ev.warIds || [];
        const allQuests = [], wars = [];
        await Promise.all(warIds.map(async wid => {
            try {
                const { quests, warInfo } = await getWarQuests(wid, region);
                allQuests.push(...quests);
                wars.push(warInfo);
            } catch (e) { console.warn('getEventQuests war', wid, e); }
        }));
        return {
            quests: allQuests,
            wars,
            activity: {
                kind: 'event',
                id: String(ev.id),
                name: ev.name || '',
                banner: ev.banner || ev.noticeBanner || '',
                startedAt: ev.startedAt,
                endedAt: ev.endedAt,
            },
        };
    }

    /**
     * Get phases and script IDs for a quest.
     * Returns { name, phaseScripts: [{phase, scripts:[{scriptId,script}]}], phasesNoBattle, phasesWithEnemies }
     */
    async function getQuestScripts(questId, region = 'JP') {
        region = norm(region);
        const quest = await get(`${BASE}/nice/${region}/quest/${questId}`);
        if (Array.isArray(quest.phaseScripts)) {
            return {
                name: quest.name || '',
                questId: String(questId),
                phaseScripts: _normalizePhaseScripts(quest.phaseScripts),
                phasesNoBattle: quest.phasesNoBattle || [],
                phasesWithEnemies: quest.phasesWithEnemies || [],
            };
        }
        const phases = quest.phases || [];
        const phaseScripts = [];
        const phasesNoBattle = [];
        const phasesWithEnemies = [];

        await Promise.all(phases.map(async phase => {
            try {
                const pd = await get(`${BASE}/nice/${region}/quest/${questId}/${phase}`);
                const scripts = (pd.scripts || []).map(s => ({
                    scriptId: String(s.scriptId || s.id || s),
                    script: s.script || '',
                }));
                phaseScripts.push({ phase, scripts });
                const hasBattle = (pd.stages || []).some(st => (st.enemies || []).length > 0 || (st.wave || []).length > 0);
                if (hasBattle) phasesWithEnemies.push(phase);
                else phasesNoBattle.push(phase);
            } catch (e) {
                phaseScripts.push({ phase, scripts: [] });
                phasesNoBattle.push(phase);
            }
        }));

        phaseScripts.sort((a, b) => a.phase - b.phase);
        return { name: quest.name || '', questId: String(questId), phaseScripts, phasesNoBattle, phasesWithEnemies };
    }

    /**
     * Extract dialogues from a script (raw FGO script text).
     * Returns { dialogues: [{speaker, content, scriptId}] }.
     */
    async function extractDialogues(scriptId, region = 'JP') {
        region = norm(region);
        if (region === 'NA') {
            const rayshift = await tryRayshiftDialogues(scriptId);
            if (rayshift) return { dialogues: rayshift };
        }
        const { raw } = await fetchScriptText(scriptId, region);
        return { dialogues: parseDialogues(raw, scriptId) };
    }

    async function fetchScriptText(scriptId, region = 'JP') {
        region = norm(region);
        const meta = await get(`${BASE}/nice/${region}/script/${scriptId}`);
        const rawUrl = meta.script;
        if (!rawUrl) throw new Error(`No script URL for ${scriptId}`);
        const rawResp = await fetch(rawUrl);
        if (!rawResp.ok) throw new Error(`Script text HTTP ${rawResp.status}: ${rawUrl}`);
        const raw = await rawResp.text();
        return { meta, raw, scriptUrl: rawUrl };
    }

    function cleanTranslatedScriptText(text) {
        return String(text || '')
            .replace(/\[%1\]/g, '\u85e4\u4e38\u7acb\u9999')
            .replace(/\[line 3\]/g, '\u2014')
            .replace(/\[line 6\]/g, '\u2014')
            .replace(/\[line 18\]/g, '\u2014');
    }

    async function rayshiftAvailable(scriptId) {
        const url = `https://rayshift.io/api/v1/translate/check-ingame/${encodeURIComponent(scriptId)}`;
        try {
            const head = await fetch(url, { method: 'HEAD' });
            if (head.ok) return true;
        } catch {}
        try {
            const resp = await fetch(url);
            return resp.ok;
        } catch {
            return false;
        }
    }

    async function tryRayshiftDialogues(scriptId) {
        try {
            if (!await rayshiftAvailable(scriptId)) return null;
            const jp = await fetchScriptText(scriptId, 'JP');
            const rsUrl = `https://rayshift.io/api/v1/translate/script-ingame/${encodeURIComponent(scriptId)}`;
            const rsResp = await fetch(rsUrl);
            if (!rsResp.ok) return null;
            const rsRaw = await rsResp.text();
            const jpDialogues = parseDialogues(cleanTranslatedScriptText(jp.raw), scriptId);
            const rsDialogues = parseDialogues(cleanTranslatedScriptText(rsRaw), scriptId);
            if (!jpDialogues.length || !rsDialogues.length) return null;
            return jpDialogues.map((jpLine, idx) => ({
                speaker: jpLine.speaker || '',
                content: jpLine.content || '',
                scriptId: String(scriptId),
                translated_content: (rsDialogues[idx] && rsDialogues[idx].content) || '',
                rayshift: true,
            }));
        } catch (e) {
            console.warn('Rayshift fetch failed:', scriptId, e);
            return null;
        }
    }

    async function checkAtlasTranslations(questId) {
        questId = String(questId || '');
        if (!questId) throw new Error('quest_id required');
        const regions = ['NA', 'CN', 'TW', 'KR'];
        const availability = {};
        await Promise.all(regions.map(async region => {
            try {
                const resp = await fetch(`${BASE}/basic/${region}/quest/${encodeURIComponent(questId)}`);
                if (!resp.ok) {
                    availability[region] = false;
                    return;
                }
                const data = await resp.json();
                availability[region] = !!(data && data.id);
            } catch {
                availability[region] = false;
            }
        }));

        let hasRayshift = false;
        try {
            const phase = await get(`${BASE}/nice/JP/quest/${encodeURIComponent(questId)}/1`);
            const firstScriptId = String((phase.scripts || [])[0]?.scriptId || '');
            if (firstScriptId) hasRayshift = await rayshiftAvailable(firstScriptId);
        } catch {
            hasRayshift = false;
        }
        if (hasRayshift) availability.NA = true;
        availability.rayshift = hasRayshift;
        return availability;
    }

    async function getAtlasDialogues(scriptIds = [], targetRegion = 'NA') {
        const allowed = new Set(['NA', 'CN', 'TW', 'KR']);
        targetRegion = String(targetRegion || 'NA').toUpperCase();
        if (!Array.isArray(scriptIds) || scriptIds.length === 0) throw new Error('script_ids required');
        if (!allowed.has(targetRegion)) throw new Error('target_region must be one of CN, KR, NA, TW');

        const allDialogues = [];
        for (const scriptId of scriptIds) {
            const data = await extractDialogues(String(scriptId), targetRegion);
            allDialogues.push(...(data.dialogues || []));
        }
        const translated = allDialogues.map(d => ({
            speaker: d.speaker || '',
            translated_content: d.translated_content || d.content || '',
            rayshift: !!d.rayshift,
        }));
        return {
            translated_dialogues: translated,
            source_region: targetRegion,
            rayshift: allDialogues.some(d => d.rayshift),
        };
    }

    /**
     * Parse raw script text into dialogue lines (plain extraction, no visual framing).
     * IMPORTANT: This must produce the same number of entries that
     * parseScriptVisual counts via dialogueIdx, otherwise gaming.html's choice
     * popup will read mismatched translations. The shared rule (mirrors Python
     * extract_dialogues regex) is: trim + [r]->\n; if non-empty, count/push.
     * Tag-only content (e.g. [se voice]) is NOT stripped here — it is still
     * counted, even though its cleaned form is empty.
     */
    function parseDialogues(raw, scriptId = '') {
        const lines = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
        const dialogues = [];
        const SPEAKER_RE = /^＠(.*)/;          // allow empty speaker
        const CHOICE_RE  = /^？(\d+)：(.+)/;
        const END_RE     = /^？！/;
        let lastChoiceNum = null;
        let i = 0;
        while (i < lines.length) {
            const line = lines[i].trim();
            i++;
            if (!line) continue;
            const sm = SPEAKER_RE.exec(line);
            if (sm) {
                const speakerRaw = sm[1].trim();
                const slotM = /^([A-Z])：(.+)$/.exec(speakerRaw);
                const speaker = slotM ? slotM[2].trim() : (speakerRaw || 'Narrator');
                const parts = [];
                while (i < lines.length) {
                    const cl = lines[i].trim(); i++;
                    if (cl.includes('[k]')) {
                        const pre = cl.slice(0, cl.indexOf('[k]')).trim();
                        if (pre) parts.push(pre);
                        break;
                    }
                    if (cl) parts.push(cl);
                }
                // Use raw-trim (with [r]->\n) for emptiness — matches the
                // Python extract_dialogues regex semantics. Tag content is
                // preserved for the LLM prompt.
                const rawContent = parts.join('\n').replace(/\[r\]/g, '\n').trim();
                if (rawContent) dialogues.push({ speaker, content: rawContent, scriptId });
                continue;
            }
            const cm = CHOICE_RE.exec(line);
            if (cm) {
                const txt = cm[2].trim().replace(/\[r\]/g, '\n');
                if (txt) {
                    lastChoiceNum = cm[1];
                    dialogues.push({ speaker: '藤丸立香', content: `Choice ${cm[1]}: ${txt}`, scriptId });
                }
                continue;
            }
            if (END_RE.test(line)) {
                dialogues.push({
                    speaker: 'System',
                    content: lastChoiceNum ? `Choice ${lastChoiceNum} Ending` : 'Choice Ending',
                    scriptId,
                });
            }
        }
        return dialogues;
    }

    /**
     * Parse raw script into visual frames for gaming mode.
     * Faithful port of Python _parse_fgo_script in app.py.
     * Notable:
     *  - Two-pass: pre-scan choice groups so popup is emitted BEFORE branches
     *  - Each branch dialogue tagged with branchId; choice frame carries endDialogueIdx
     *  - dialogueIdx is incremented for ＠..[k], ？N：text, AND ？！
     *  - clean_text PRESERVES formatting tags ([#base:reading], [align ...],
     *    [line N], [f xxx]/[/f], [image xxx]) for the frontend to render.
     */
    function parseScriptVisual(raw, region = 'JP') {
        region = norm(region);
        const BG_BASE = id => `${CDN}/${region}/Back/back${id}.png`;
        const FIG_BASE = eid => `${CDN}/${region}/CharaFigure/${eid}/${eid}.png`;

        let text = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        text = text.replace(/\[\s*\n\s*/g, '[');
        text = text.replace(/\n\s*(?=[^\[＠？\n])/g, ' ');
        text = text.replace(/\[%1\]/g, '藤丸立香').replace(/\[r\]/g, '\n');
        // NOTE: do NOT pre-replace [line N] here; gaming.html renders it.
        const lines = text.split('\n');

        // Strip every bracketed command except those the frontend renders.
        const _PRESERVE = /\[(?:#[^\[\]:]+:[^\[\]]+|align(?:\s+\w+)?|line\s+\d+|f\s+[\w-]+|\/f|image\s+[\w-]+)\]/gi;
        function cleanText(s) {
            // [#text] (no reading) -> text  (ruby with no reading: unwrap)
            s = s.replace(/\[#([^\[\]:]+)\](?!:)/g, '$1');
            const out = [];
            let last = 0;
            _PRESERVE.lastIndex = 0;
            let m;
            while ((m = _PRESERVE.exec(s)) !== null) {
                let chunk = s.slice(last, m.index);
                chunk = chunk.replace(/\[[^\[\]]+\]/g, '');
                out.push(chunk);
                out.push(m[0]);
                last = m.index + m[0].length;
            }
            let tail = s.slice(last);
            tail = tail.replace(/\[[^\[\]]+\]/g, '');
            out.push(tail);
            return out.join('').trim();
        }

        // ---- pre-scan choice groups to know dialogueIdx of every ？N and ？！ ----
        // The dialogueIdx must align with parseDialogues' indices, so we use
        // the SAME emptiness rule: raw content (post [r]->\n + trim) is non-empty.
        function rawNonEmpty(s) {
            return s.replace(/\[r\]/g, '\n').trim() !== '';
        }
        function scanChoiceGroups() {
            let idx = 0;
            const groups = [];
            let cur = null;
            let j = 0;
            while (j < lines.length) {
                const ln = lines[j].trim();
                if (ln.startsWith('＠')) {
                    // Collect content lines through [k] just like the main pass,
                    // so we can decide whether this ＠ block contributes to idx.
                    const rawAfter = ln.slice(1);
                    let initial = '';
                    if (rawAfter && rawAfter[0] !== ' ') {
                        const sp = rawAfter.trim().split(/\s+/, 2);
                        if (sp.length === 2) initial = rawAfter.trim().slice(sp[0].length).trim();
                    } else {
                        initial = rawAfter.trim();
                    }
                    const parts = [];
                    let consumedK = false;
                    if (initial) {
                        if (initial.includes('[k]')) {
                            const pre = initial.slice(0, initial.indexOf('[k]')).trim();
                            if (pre) parts.push(pre);
                            consumedK = true;
                        } else {
                            parts.push(initial);
                        }
                    }
                    let jj = j + 1;
                    if (!consumedK) {
                        while (jj < lines.length) {
                            const cl = lines[jj].trim();
                            if (cl.includes('[k]')) {
                                const pre = cl.slice(0, cl.indexOf('[k]')).trim();
                                if (pre) parts.push(pre);
                                jj++;
                                break;
                            }
                            if (cl) parts.push(cl);
                            jj++;
                        }
                    }
                    if (rawNonEmpty(parts.join('\n'))) idx++;
                    j = consumedK ? j + 1 : jj;
                    continue;
                }
                const cm = /^？(\d+)：(.+)/.exec(ln);
                if (cm) {
                    const txt = cm[2].trim().replace(/\[r\]/g, '\n');
                    if (cur === null) cur = { firstLine: j, choices: [], endDialogueIdx: null, endLine: null };
                    if (txt) {
                        cur.choices.push({
                            num: parseInt(cm[1], 10),
                            text: cleanText(cm[2].trim()),
                            dialogueIdx: idx,
                        });
                        idx++;
                    }
                    j++;
                    continue;
                }
                if (ln.startsWith('？！')) {
                    if (cur !== null) {
                        cur.endDialogueIdx = idx;
                        cur.endLine = j;
                        groups.push(cur);
                        cur = null;
                    }
                    idx++;
                    j++;
                    continue;
                }
                j++;
            }
            if (cur !== null) {
                cur.endDialogueIdx = idx;
                cur.endLine = lines.length;
                groups.push(cur);
            }
            return groups;
        }

        const choiceGroups = scanChoiceGroups();
        const groupByFirstLine = new Map(choiceGroups.map(g => [g.firstLine, g]));
        let currentGroup = null;
        let currentBranch = null;

        const state = { bg: '', sprites: {}, talker: null, cameraFilter: null, bgm: null };
        const frames = [], entityIds = new Set();
        let dialogueIdx = 0, pendingEffects = [];

        function takeEffects() { const e = pendingEffects; pendingEffects = []; return e; }

        function snapshotSprites() {
            return Object.entries(state.sprites)
                .filter(([, sp]) => sp.visible && sp.entityId)
                .map(([slot, sp]) => ({
                    slot, entityId: sp.entityId, name: sp.name || '',
                    face: sp.face || 1, url: FIG_BASE(sp.entityId),
                    talking: slot === state.talker,
                }));
        }

        let i = 0;
        while (i < lines.length) {
            const lineIdx = i;
            const line = lines[i++].trim();
            if (!line) continue;

            let m;
            if (m = /^\[scene\s+(\d+)\]/.exec(line)) { state.bg = BG_BASE(m[1]); continue; }
            if (m = /^\[bScene\s+(\d+)/.exec(line)) { if (!state.bg) state.bg = BG_BASE(m[1]); continue; }
            if (m = /^\[imageSet\s+\w\s+back(\d+)/.exec(line)) { if (!state.bg) state.bg = BG_BASE(m[1]); continue; }

            if (m = /^\[charaSet\s+(\w)\s+(\d+)\s+(\d+)\s*(.*?)\]/.exec(line)) {
                state.sprites[m[1]] = { entityId: m[2], name: m[4].trim(), face: parseInt(m[3]), visible: false };
                entityIds.add(m[2]); continue;
            }
            if (m = /^\[charaFace\s+(\w)\s+(\d+)\]/.exec(line)) {
                if (state.sprites[m[1]]) state.sprites[m[1]].face = parseInt(m[2]); continue;
            }
            if (m = /^\[charaTalk\s+(\w+)\]/.exec(line)) {
                state.talker = ['off', 'depthOff', 'on'].includes(m[1]) ? null : m[1]; continue;
            }
            if (m = /^\[charaFadein\s+(\w)/.exec(line)) {
                if (state.sprites[m[1]]) state.sprites[m[1]].visible = true; continue;
            }
            if (m = /^\[charaFadeout\s+(\w)/.exec(line)) {
                if (state.sprites[m[1]]) state.sprites[m[1]].visible = false; continue;
            }
            if (m = /^\[charaCrossFade\s+(\w)\s+(\d+)/.exec(line)) {
                if (state.sprites[m[1]]) { state.sprites[m[1]].entityId = m[2]; entityIds.add(m[2]); } continue;
            }

            if (line.startsWith('＠')) {
                // Match Python parsing: distinguish "＠speaker" (no space, content
                // on next line) from "＠ collapsed-content" (empty speaker, body
                // collapsed onto same line by preprocessing).
                const rawAfter = lines[lineIdx].replace(/^.*?＠/, '').replace(/^＠/, '');
                // The above is overkill; lines[lineIdx] starts with ＠ before trim
                // — but we already trimmed `line`, so re-derive from line:
                const rawAfter2 = line.slice(1); // includes leading space if any
                let speaker = '';
                let initialContent = '';
                if (rawAfter2 === '' || rawAfter2[0] === ' ') {
                    speaker = '';
                    initialContent = rawAfter2.trim();
                } else {
                    let speakerRaw = rawAfter2.trim();
                    const sp = speakerRaw.split(/\s+/, 2);
                    if (sp.length === 2) {
                        speakerRaw = sp[0];
                        // recover the rest after first whitespace
                        initialContent = rawAfter2.trim().slice(sp[0].length).trim();
                    }
                    const slotPrefix = /^([A-Z])：(.+)$/.exec(speakerRaw);
                    if (slotPrefix) {
                        const speakerSlot = slotPrefix[1];
                        speaker = slotPrefix[2].trim();
                        if (state.sprites[speakerSlot]) state.talker = speakerSlot;
                    } else {
                        speaker = speakerRaw;
                    }
                }

                const contentParts = [];
                let emittedFromInitial = false;
                if (initialContent) {
                    if (initialContent.includes('[k]')) {
                        const beforeK = initialContent.slice(0, initialContent.indexOf('[k]')).trim();
                        if (beforeK) contentParts.push(beforeK);
                        const rawJoined = contentParts.join('\n');
                        if (rawNonEmpty(rawJoined)) {
                            const content = cleanText(rawJoined);
                            if (content) {
                                frames.push({
                                    type: 'dialogue', bg: state.bg, sprites: snapshotSprites(),
                                    speaker, text: content, dialogueIdx, branchId: currentBranch,
                                    effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm,
                                });
                            }
                            dialogueIdx++;
                        }
                        emittedFromInitial = true;
                    } else {
                        contentParts.push(initialContent);
                    }
                }
                if (emittedFromInitial) continue;

                while (i < lines.length) {
                    const cl = lines[i++].trim();
                    if (cl.includes('[k]')) {
                        const beforeK = cl.slice(0, cl.indexOf('[k]')).trim();
                        if (beforeK) contentParts.push(beforeK);
                        break;
                    }
                    if (cl) contentParts.push(cl);
                }
                const rawJoined = contentParts.join('\n');
                if (rawNonEmpty(rawJoined)) {
                    const content = cleanText(rawJoined);
                    if (content) {
                        frames.push({
                            type: 'dialogue', bg: state.bg, sprites: snapshotSprites(),
                            speaker, text: content, dialogueIdx, branchId: currentBranch,
                            effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm,
                        });
                    }
                    dialogueIdx++;
                }
                continue;
            }

            if (m = /^？(\d+)：(.+)/.exec(line)) {
                const num = parseInt(m[1], 10);
                const txt = m[2].trim().replace(/\[r\]/g, '\n');
                if (currentGroup === null) {
                    let grp = groupByFirstLine.get(lineIdx) || null;
                    if (!grp) {
                        for (const g of choiceGroups) {
                            if (g.firstLine <= lineIdx && lineIdx <= (g.endLine ?? lines.length)) {
                                grp = g; break;
                            }
                        }
                    }
                    if (grp) {
                        currentGroup = grp;
                        frames.push({
                            type: 'choice', bg: state.bg, sprites: snapshotSprites(),
                            choices: grp.choices.map(c => ({ ...c })),
                            dialogueIdx: grp.choices.length ? grp.choices[0].dialogueIdx : dialogueIdx,
                            endDialogueIdx: grp.endDialogueIdx,
                            effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm,
                        });
                    }
                }
                currentBranch = num;
                if (txt) dialogueIdx++; // ？N consumes one slot only when text non-empty
                continue;
            }

            if (line.startsWith('？！')) {
                currentGroup = null;
                currentBranch = null;
                dialogueIdx++; // always consumes one slot ("Choice N Ending")
                continue;
            }

            // Effects
            if (m = /^\[fadeout\s+(\w+)(?:\s+([\d.]+))?\s*\]/.exec(line)) {
                pendingEffects.push({ type: 'fadeOut', color: m[1], dur: parseFloat(m[2] || 1) });
                frames.push({ type: 'transition', bg: state.bg, sprites: [],
                    effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm });
                continue;
            }
            if (m = /^\[fadein\s+(\w+)(?:\s+([\d.]+))?\s*\]/.exec(line)) {
                pendingEffects.push({ type: 'fadeIn', color: m[1], dur: parseFloat(m[2] || 1) }); continue;
            }
            if (m = /^\[cameraFilter\s+(\w+)\s*\]/.exec(line)) {
                state.cameraFilter = m[1]; pendingEffects.push({ type: 'cameraFilter', color: m[1] }); continue;
            }
            if (/^\[cameraFilter(Off|Stop)?\s*\]/.test(line)) {
                state.cameraFilter = null; pendingEffects.push({ type: 'cameraFilter', color: null }); continue;
            }
            if (m = /^\[effect\s+(\w+)\s*\]/.exec(line)) {
                const nm = m[1].toLowerCase();
                pendingEffects.push({ type: nm.includes('shake') ? 'shake' : nm.includes('flash') ? 'flash' : 'effect', name: m[1] }); continue;
            }
            if (m = /^\[bgm\s+(\w+)/.exec(line)) { state.bgm = m[1]; continue; }
            if (/^\[bgmStop\b/.test(line)) { state.bgm = null; continue; }
        }
        return { frames, entityIds: [...entityIds] };
    }

    /**
     * Fetch svtScript metadata for an array of entity IDs (in parallel).
     */
    async function fetchSvtScripts(entityIds, region = 'JP') {
        region = norm(region);
        const results = {};
        await Promise.all(entityIds.map(async eid => {
            try {
                const url = `${BASE}/raw/${region}/svtScript?charaId=${eid}`;
                const data = await get(url);
                if (Array.isArray(data) && data.length) {
                    const m = data[0];
                    results[String(eid)] = {
                        faceX: m.faceX || 0, faceY: m.faceY || 0,
                        offsetX: m.offsetX || 0, offsetY: m.offsetY || 0,
                        scale: m.scale || 1,
                        extendData: m.extendData || {},
                    };
                }
            } catch (e) { /* entity may not have svtScript */ }
        }));
        return results;
    }

    // Export
    return {
        norm,
        searchWar, searchEvent,
        latestWars, latestEvents, latestTasks,
        getWarQuests, getEventQuests,
        getQuestScripts, extractDialogues, parseDialogues,
        checkAtlasTranslations, getAtlasDialogues,
        parseScriptVisual, fetchSvtScripts,
        CDN,
    };
})();

function cleanText(s) {
    s = s.replace(/\[#([^\[\]:]+):[^\[\]]+\]/g, '$1');
    s = s.replace(/\[#([^\[\]]+)\]/g, '$1');
    s = s.replace(/\[([^\[\]:]+):([^\[\]]+)\]/g, '$1');
    s = s.replace(/\[[^\[\]]+\]/g, '');
    return s.trim();
}
