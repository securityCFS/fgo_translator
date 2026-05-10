/**
 * api.js — Atlas Academy API helpers (all CORS-friendly, no proxy needed)
 */
const AA = (() => {
    const BASE = 'https://api.atlasacademy.io';
    const CDN  = 'https://static.atlasacademy.io';

    const REGIONS = ['JP','NA','CN','TW','KR'];
    function norm(r) { return REGIONS.includes((r||'').toUpperCase()) ? r.toUpperCase() : 'JP'; }

    // Simple in-memory cache keyed by URL
    const _cache = new Map();
    async function get(url, opts = {}) {
        if (_cache.has(url)) return _cache.get(url);
        const r = await fetch(url, { signal: opts.signal });
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
        const data = await r.json();
        _cache.set(url, data);
        return data;
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
            const wars = await get(`${BASE}/export/${region}/nice_war.json`);
            if (!Array.isArray(wars)) return [];
            let evStart = {};
            try {
                const evs = await get(`${BASE}/export/${region}/basic_event.json`);
                if (Array.isArray(evs)) evs.forEach(e => { if (e.id && e.startedAt) evStart[e.id] = e.startedAt; });
            } catch {}
            const sorted = [...wars].sort((a, b) => {
                const ta = evStart[a.eventId] || a.startedAt || a.id || 0;
                const tb = evStart[b.eventId] || b.startedAt || b.id || 0;
                return tb - ta;
            });
            return sorted.slice(0, limit).map(w => _normalizeWar(w, region));
        } catch (e) {
            console.warn('latestWars export failed:', e);
            return [];
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
            return [];
        }
    }

    /**
     * Latest quests (used as "Tasks" tab). Fetches recent wars and flattens their quests.
     */
    async function latestTasks(region = 'JP', limit = 50) {
        region = norm(region);
        try {
            // Grab recent war IDs from export, then batch-fetch quest lists
            const wars = await latestWars(region, 20);
            const allQuests = [];
            await Promise.all(wars.slice(0, 10).map(async w => {
                try {
                    const { quests } = await getWarQuests(w.id, region);
                    allQuests.push(...quests);
                } catch { /* skip on error */ }
            }));
            return allQuests
                .sort((a, b) => (b.openedAt || 0) - (a.openedAt || 0))
                .slice(0, limit);
        } catch (e) { console.error('latestTasks', e); return []; }
    }

    // ── Quest detail ─────────────────────────────────────────────────────────

    /**
     * Get quests belonging to a war.
     */
    async function getWarQuests(warId, region = 'JP') {
        region = norm(region);
        // Use nice war which has full quest list and phase info
        const url = `${BASE}/nice/${region}/war/${warId}`;
        const war = await get(url);
        const quests = (war.spots || []).flatMap(sp => (sp.quests || []).map(q => ({
            id: String(q.id),
            name: q.name || '',
            type: q.type || '',
            spotName: sp.name || '',
            openedAt: q.openedAt,
            closedAt: q.closedAt,
            region,
            warId: String(warId),
            warName: war.name || war.longName || '',
        })));
        return { quests, warInfo: { id: String(warId), name: war.name || '', longName: war.longName || '', banner: war.banner || '' } };
    }

    /**
     * Get quests belonging to an event (resolves event → warIds → quests).
     */
    async function getEventQuests(eventId, region = 'JP') {
        region = norm(region);
        const ev = await get(`${BASE}/nice/${region}/event/${eventId}`);
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
        const meta = await get(`${BASE}/nice/${region}/script/${scriptId}`);
        const rawUrl = meta.script;
        if (!rawUrl) throw new Error(`No script URL for ${scriptId}`);
        const rawResp = await fetch(rawUrl);
        const raw = await rawResp.text();
        return { dialogues: parseDialogues(raw, scriptId) };
    }

    /**
     * Parse raw script text into dialogue lines (plain extraction, no visual framing).
     */
    function parseDialogues(raw, scriptId = '') {
        const lines = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
        const dialogues = [];
        const SPEAKER_RE = /^＠(.+)/;
        const CHOICE_RE  = /^？(\d+)：(.+)/;
        const END_RE     = /^？！/;
        let i = 0;
        while (i < lines.length) {
            const line = lines[i].trim();
            i++;
            if (!line) continue;
            const sm = SPEAKER_RE.exec(line);
            if (sm) {
                const speakerRaw = sm[1].trim();
                const slotM = /^([A-Z])：(.+)$/.exec(speakerRaw);
                const speaker = slotM ? slotM[2].trim() : speakerRaw;
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
                const content = cleanText(parts.join('\n'));
                if (content) dialogues.push({ speaker, content, scriptId });
                continue;
            }
            const cm = CHOICE_RE.exec(line);
            if (cm) {
                const txt = cleanText(cm[2].trim());
                if (txt) dialogues.push({ speaker: `Choice ${cm[1]}`, content: txt, scriptId });
                continue;
            }
            if (END_RE.test(line)) {
                dialogues.push({ speaker: 'Choice Ending', content: '（選択終わり）', scriptId });
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
        function scanChoiceGroups() {
            let idx = 0;
            const groups = [];
            let cur = null;
            let j = 0;
            while (j < lines.length) {
                const ln = lines[j].trim();
                if (ln.startsWith('＠')) {
                    let jj = j + 1;
                    while (jj < lines.length) {
                        if (lines[jj].includes('[k]')) { jj++; break; }
                        jj++;
                    }
                    idx++;
                    j = jj;
                    continue;
                }
                const cm = /^？(\d+)：(.+)/.exec(ln);
                if (cm) {
                    if (cur === null) cur = { firstLine: j, choices: [], endDialogueIdx: null, endLine: null };
                    cur.choices.push({
                        num: parseInt(cm[1], 10),
                        text: cleanText(cm[2].trim()),
                        dialogueIdx: idx,
                    });
                    idx++;
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
                        const content = cleanText(contentParts.join('\n'));
                        if (content) {
                            frames.push({
                                type: 'dialogue', bg: state.bg, sprites: snapshotSprites(),
                                speaker, text: content, dialogueIdx, branchId: currentBranch,
                                effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm,
                            });
                        }
                        dialogueIdx++;
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
                const content = cleanText(contentParts.join('\n'));
                if (content) {
                    frames.push({
                        type: 'dialogue', bg: state.bg, sprites: snapshotSprites(),
                        speaker, text: content, dialogueIdx, branchId: currentBranch,
                        effects: takeEffects(), cameraFilter: state.cameraFilter, bgm: state.bgm,
                    });
                }
                dialogueIdx++;
                continue;
            }

            if (m = /^？(\d+)：(.+)/.exec(line)) {
                const num = parseInt(m[1], 10);
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
                dialogueIdx++; // ？N consumes one translation slot
                continue;
            }

            if (line.startsWith('？！')) {
                currentGroup = null;
                currentBranch = null;
                dialogueIdx++; // ？！ consumes one translation slot ("Choice N Ending")
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
