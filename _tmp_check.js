
        console.log("JS loaded");
        function showLoading() {
            document.getElementById('loading').classList.add('active', 'flex');
        }
        function hideLoading() {
            document.getElementById('loading').classList.remove('active', 'flex');
        }
        hideLoading(); // 页面加载后立即隐藏

        console.log("Loading index.html");
        let currentDialogues = [];
        let selectedWar = null;
        let selectedActivity = null;
        let warResults = [];
        let questResults = [];
        let selectedQuest = null;
        let selectedPhaseScripts = [];   // scripts for the currently selected phase
        let _phaseScriptsMap = {};        // phase → scripts[], populated by selectQuest()
        let currentQuestRegion = 'JP';
        let questResultLabel = 'quests';
        const questCache = new Map();    // key: `${kind}_${id}_${region}` → {quests, wars, activity}

        // 初始化 Socket.IO
        const socket = io();
        let currentTranslationSession = null;

        // 监听翻译进度更新
        socket.on('translation_progress', function(data) {
            if (data.session_id === currentTranslationSession) {
                const progressBar = document.getElementById('progressBar');
                const progressText = document.getElementById('progressText');
                progressBar.style.width = `${data.progress}%`;
                progressText.textContent = `Translating: ${data.current}/${data.total} (${data.progress}%) - ${data.speaker || ''}`;
            }
        });

        // 搜索 war
        function escapeHtml(value) {
            return String(value ?? '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function formatUnixTime(value) {
            if (!value) return '';
            const date = new Date(Number(value) * 1000);
            if (Number.isNaN(date.getTime())) return '';
            return date.toLocaleString();
        }

        function resetTranslationState() {
            document.getElementById('translationResults').innerHTML = '';
            document.getElementById('scriptTabs').classList.add('hidden');
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('translateSection').style.display = 'none';
            document.getElementById('translationOptions').style.display = 'none';
            document.getElementById('translationDisplay').style.display = 'none';
            selectedQuest = null;
        }

        async function loadLatestActivities() {
            showLoading();
            resetTranslationState();
            selectedWar = null;
            selectedActivity = null;
            warResults = [];
            currentQuestRegion = document.getElementById('latestActivityRegion').value || 'JP';
            const activityType = document.getElementById('latestActivityType').value || 'war';
            document.getElementById('warResultList').innerHTML = '';
            document.getElementById('warResultInfo').innerText = '';
            document.getElementById('questResultList').innerHTML = '';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('latestActivityInfo').innerText = '';

            const rawLimit = parseInt(document.getElementById('latestActivityLimit').value, 10);
            const limit = Number.isFinite(rawLimit) ? Math.min(Math.max(rawLimit, 1), 200) : 50;

            try {
                const response = await fetch('/latest_activities', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ region: currentQuestRegion, activity_type: activityType, limit })
                });
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                warResults = data.activities || [];
                currentQuestRegion = data.region || currentQuestRegion;
                document.getElementById('latestActivityInfo').innerHTML = `Loaded <b>${warResults.length}</b> latest ${escapeHtml(activityType)} activities from ${escapeHtml(currentQuestRegion)}.`;
                showActivityResults(warResults, activityType === 'event' ? 'events' : 'wars');
            } catch (error) {
                console.error('loadLatestActivities error:', error);
                document.getElementById('latestActivityInfo').innerHTML = `<span style='color:#c00'>${escapeHtml(error.message)}</span>`;
                document.getElementById('warResultInfo').innerHTML = `<span style='color:#c00'>Error occurred.</span>`;
            } finally {
                hideLoading();
            }
        }

        async function loadLatestTasks() {
            showLoading();
            resetTranslationState();
            selectedWar = null;
            selectedActivity = null;
            questResults = [];
            currentQuestRegion = document.getElementById('latestTaskRegion').value || 'JP';
            questResultLabel = 'tasks';
            document.getElementById('questSectionTitle').textContent = 'Latest Tasks';
            document.getElementById('getQuestBtn').style.display = 'none';
            document.getElementById('questSection').style.display = '';
            document.getElementById('questResultList').innerHTML = '';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('latestTaskInfo').innerText = '';
            renderActivityBanner(null, []);

            const rawLimit = parseInt(document.getElementById('latestTaskLimit').value, 10);
            const limit = Number.isFinite(rawLimit) ? Math.min(Math.max(rawLimit, 1), 200) : 50;

            try {
                const response = await fetch('/latest_tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ region: currentQuestRegion, limit })
                });
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                questResults = data.tasks || [];
                currentQuestRegion = data.region || currentQuestRegion;
                document.getElementById('latestTaskInfo').innerHTML = `Loaded <b>${questResults.length}</b> latest tasks from ${escapeHtml(currentQuestRegion)}.`;
                showQuestResults(questResults, 'tasks');
            } catch (error) {
                console.error('loadLatestTasks error:', error);
                document.getElementById('latestTaskInfo').innerHTML = `<span style='color:#c00'>${escapeHtml(error.message)}</span>`;
                document.getElementById('questResultInfo').innerHTML = `<span style='color:#c00'>Error occurred.</span>`;
            } finally {
                hideLoading();
            }
        }

        async function searchActivity() {
            console.log("Starting search");
            showLoading();
            const activityName = document.getElementById('warName').value.trim();
            const activityType = document.getElementById('activitySearchType').value || 'war';
            currentQuestRegion = document.getElementById('activityRegion').value || 'JP';
            questResultLabel = 'quests';
            document.getElementById('questSectionTitle').textContent = 'Related Quests';
            document.getElementById('getQuestBtn').style.display = '';
            document.getElementById('latestActivityInfo').innerText = '';
            document.getElementById('latestTaskInfo').innerText = '';
            selectedWar = null;
            selectedActivity = null;
            // 清空展示区和quest选择内容
            document.getElementById('warResultList').innerHTML = '';
            document.getElementById('questResultList').innerHTML = '';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('translationResults').innerHTML = '';
            document.getElementById('scriptTabs').classList.add('hidden');
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('translateSection').style.display = 'none';
            document.getElementById('translationOptions').style.display = 'none';
            document.getElementById('translationDisplay').style.display = 'none';
            try {
                // 检查是否为纯数字
                const endpoint = activityType === 'event' ? '/search_event' : '/search_war';
                const body = activityType === 'event'
                    ? { event_name: activityName, region: currentQuestRegion, limit: 50 }
                    : { war_name: activityName, region: currentQuestRegion, limit: 50 };
                const activityResponse = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const activityData = await activityResponse.json();
                if (activityData.error) throw new Error(activityData.error);
                warResults = activityType === 'event' ? (activityData.events || []) : (activityData.wars || []);
                showActivityResults(warResults, activityType === 'event' ? 'events' : 'wars');
                return;
                if (/^\d+$/.test(warName)) {
                    const useId = window.confirm('检测到输入为数字，是否以ID检索？\n点击"确定"以ID检索，点击"取消"以名称模糊搜索。');
                    if (useId) {
                        // 以ID检索，直接用id字符串请求
                        const warResponse = await fetch('/search_war', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ war_name: warName })
                        });
                        const warData = await warResponse.json();
                        warResults = warData.wars || [];
                        showWarResults(warResults);
                        return;
                    }
                    // 否则继续走名称模糊搜索
                }
                // Search for war（原有逻辑）
                const warResponse = await fetch('/search_war', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ war_name: warName })
                });
                const warData = await warResponse.json();
                warResults = warData.wars || [];
                showWarResults(warResults);
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('warResultInfo').innerText = 'Error occurred.';
                document.getElementById('warResultList').innerHTML = '';
            } finally {
                hideLoading();
            }
        }

        function showActivityResults(list, label = 'activities') {
            selectedWar = null;
            selectedActivity = null;
            document.getElementById('questSection').style.display = 'none';
            document.getElementById('translateSection').style.display = 'none';
            if (list.length === 0) {
                document.getElementById('warResultInfo').innerHTML = `<span style='color:#c00'>No results found</span>`;
                document.getElementById('warResultList').innerHTML = '';
                return;
            }
            document.getElementById('warResultInfo').innerHTML = `Found <b>${list.length}</b> ${escapeHtml(label)}. Please select one.`;
            document.getElementById('warResultList').innerHTML = list.map((activity, idx) => {
                const kind = activity.itemKind || 'war';
                const meta = kind === 'event'
                    ? [
                        activity.region,
                        activity.type,
                        activity.startedAt ? `Start: ${formatUnixTime(activity.startedAt)}` : '',
                        activity.endedAt ? `End: ${formatUnixTime(activity.endedAt)}` : '',
                        activity.warIds && activity.warIds.length ? `${activity.warIds.length} wars` : 'No war map'
                    ].filter(Boolean).map(escapeHtml).join(' | ')
                    : [
                        activity.region,
                        activity.longName,
                        activity.eventName,
                        activity.age,
                        activity.startedAt ? `Start: ${formatUnixTime(activity.startedAt)}` : '',
                        activity.endedAt ? `End: ${formatUnixTime(activity.endedAt)}` : ''
                    ].filter(Boolean).map(escapeHtml).join(' | ');
                const bannerUrl = activity.banner || activity.noticeBanner || activity.headerImage || activity.icon || '';
                const thumbHtml = bannerUrl
                    ? `<img src="${escapeHtml(bannerUrl)}" alt="" loading="lazy" class="w-full h-auto max-h-28 object-contain rounded-md border border-slate-200 bg-slate-100" onerror="this.style.display='none'"/>`
                    : '';
                return `<div class="result-item selectable group mb-2" data-id="${escapeHtml(activity.id)}" data-idx="${idx}" onclick="selectActivity(this, ${idx})">
                    <div class="cursor-pointer group-hover:bg-blue-50 rounded-lg px-2 py-2 border border-transparent group-hover:border-blue-200 transition-colors">
                        ${thumbHtml ? `<div class="mb-2">${thumbHtml}</div>` : ''}
                        <div class="font-semibold text-base text-blue-900 leading-snug">
                            <span class="text-xs font-mono text-slate-400 mr-1.5">[${escapeHtml(kind)}:${escapeHtml(activity.id)}]</span><span>${escapeHtml(activity.name || activity.longName)}</span>
                        </div>
                        ${meta ? `<div class="text-xs text-gray-500 mt-0.5">${meta}</div>` : ''}
                    </div>
                </div>`;
            }).join('');
        }

        function showWarResults(list) {
            selectedWar = null;
            document.getElementById('questSection').style.display = 'none';
            document.getElementById('translateSection').style.display = 'none';
            if (list.length === 0) {
                document.getElementById('warResultInfo').innerHTML = `<span style='color:#c00'>No results found</span>`;
                document.getElementById('warResultList').innerHTML = '';
                return;
            }
            document.getElementById('warResultInfo').innerHTML = `Found <b>${list.length}</b> wars. Please select one.`;
            document.getElementById('warResultList').innerHTML = list.map((w, idx) =>
                `<div class="result-item selectable group mb-1" data-id="${w.id}" data-idx="${idx}" onclick="selectWar(this, ${w.id})">
                    <div class="flex items-center cursor-pointer font-semibold text-lg text-blue-900 group-hover:bg-blue-50 rounded px-2 py-1">
                        <span class="mr-2">[${w.id}]</span> <span>${w.name}</span>
                    </div>
                </div>`
            ).join('');
        }

        function selectActivity(elem, idx) {
            document.querySelectorAll('.result-item.selectable').forEach(e => e.style.background = '');
            elem.style.background = '#e3f2fd';
            selectedActivity = warResults[idx];
            selectedWar = selectedActivity;
            currentQuestRegion = (selectedActivity && selectedActivity.region) || 'JP';
            questResultLabel = 'quests';
            document.getElementById('questSectionTitle').textContent = 'Related Quests';
            document.getElementById('getQuestBtn').style.display = '';
            document.getElementById('questSection').style.display = '';
            document.getElementById('translateSection').style.display = 'none';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('questResultList').innerHTML = '';
            selectedQuest = null;
            selectedPhaseScripts = [];
            // Restore from cache if available
            const cacheKey = `${selectedActivity.itemKind || 'war'}_${selectedActivity.id}_${currentQuestRegion}`;
            if (questCache.has(cacheKey)) {
                const cached = questCache.get(cacheKey);
                questResults = cached.quests;
                renderActivityBanner(cached.activity, cached.wars);
                showQuestResults(cached.quests, 'quests', cached.wars);
                document.getElementById('questResultInfo').innerHTML =
                    `<b>${cached.quests.length}</b> quests loaded (cached) — click <em>Fetch Quests</em> to refresh.`;
            }
        }

        function selectWar(elem, warId) {
            document.querySelectorAll('.result-item.selectable').forEach(e => e.style.background = '');
            elem.style.background = '#e3f2fd';
            selectedWar = warResults.find(w => w.id === warId);
            selectedActivity = selectedWar;
            currentQuestRegion = selectedWar.region || 'JP';
            questResultLabel = 'quests';
            document.getElementById('questSectionTitle').textContent = 'Related Quests';
            document.getElementById('getQuestBtn').style.display = '';
            document.getElementById('questSection').style.display = '';
            document.getElementById('translateSection').style.display = 'none';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('questResultList').innerHTML = '';
            selectedQuest = null;
        }

        // 获取相关 quest
        async function getRelatedQuests() {
            if (!selectedActivity) return;
            showLoading();
            questResultLabel = 'quests';
            document.getElementById('questSectionTitle').textContent = 'Related Quests';
            document.getElementById('getQuestBtn').style.display = '';
            document.getElementById('questResultList').innerHTML = '';
            document.getElementById('questResultInfo').innerText = '';
            document.getElementById('translationResults').innerHTML = '';
            document.getElementById('scriptTabs').classList.add('hidden');
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('translateSection').style.display = 'none';
            document.getElementById('translationOptions').style.display = 'none';
            document.getElementById('translationDisplay').style.display = 'none';
            renderActivityBanner(null, []);
            try {
                const query = {
                    region: currentQuestRegion
                };
                if (selectedActivity.itemKind === 'event') {
                    query.event_id = selectedActivity.id;
                } else {
                    query.war_id = selectedActivity.id;
                }
                const questResponse = await fetch('/search_quest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(query)
                });
                const questData = await questResponse.json();
                questResults = questData.quests || [];
                renderActivityBanner(questData.activity, questData.wars || []);
                // 如果有错误信息，显示在结果上方
                if (questData.error) {
                    document.getElementById('questResultInfo').innerHTML = `<span style='color:#c00'>${questData.error}</span>`;
                }
                showQuestResults(questResults, 'quests', questData.wars || []);
                // Store in cache for instant restore on re-selection
                const cacheKey = `${selectedActivity.itemKind || 'war'}_${selectedActivity.id}_${currentQuestRegion}`;
                questCache.set(cacheKey, { quests: questResults, wars: questData.wars || [], activity: questData.activity || null });
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('questResultInfo').innerText = 'Error occurred.';
                document.getElementById('questResultList').innerHTML = '';
            } finally {
                hideLoading();
            }
        }

        function showQuestResults(list, label = questResultLabel, wars = []) {
            selectedQuest = null;
            questResultLabel = label;
            document.getElementById('translateSection').style.display = 'none';
            const listEl = document.getElementById('questResultList');
            if (list.length === 0) {
                document.getElementById('questResultInfo').innerHTML = `<span style='color:#c00'>No ${escapeHtml(label)} found</span>`;
                listEl.innerHTML = '';
                return;
            }
            document.getElementById('questResultInfo').innerHTML = `Found <b>${list.length}</b> ${escapeHtml(label)}. Please select one.`;

            // Build a global index map so we can keep the original questResults indexing.
            const indexById = new Map();
            list.forEach((q, i) => indexById.set(String(q.id), i));

            // Group quests by warId. Preserve war order from the wars[] array if provided,
            // otherwise from the order quests appear in the list.
            const groups = new Map();
            list.forEach((q) => {
                const key = String(q.warId || '0');
                if (!groups.has(key)) groups.set(key, { warId: key, warName: q.warName || '', quests: [] });
                groups.get(key).quests.push(q);
            });
            const warOrder = wars && wars.length ? wars.map(w => String(w.id)) : Array.from(groups.keys());
            const warMetaById = new Map((wars || []).map(w => [String(w.id), w]));

            const sectionHtml = warOrder
                .filter(wid => groups.has(wid))
                .map(wid => {
                    const group = groups.get(wid);
                    const meta = warMetaById.get(wid) || {};
                    const showHeader = warOrder.length > 1 || !!(meta.banner || meta.mapImage);
                    const headerHtml = showHeader ? `
                        <div class="flex items-center gap-3 px-3 py-2 bg-gradient-to-r from-indigo-50 to-slate-50 border border-slate-200 rounded-lg">
                            ${meta.banner ? `<img src="${escapeHtml(meta.banner)}" alt="" class="h-10 rounded shadow-sm border border-white/60" loading="lazy" onerror="this.style.display='none'"/>` : ''}
                            <div class="flex-1 min-w-0">
                                <div class="text-sm font-semibold text-slate-800 truncate">${escapeHtml(meta.name || group.warName || `War ${wid}`)}</div>
                                ${meta.longName && meta.longName !== meta.name ? `<div class="text-xs text-slate-500 truncate">${escapeHtml(meta.longName)}</div>` : ''}
                            </div>
                            <span class="text-xs font-medium text-slate-500 bg-white px-2 py-0.5 rounded-full border border-slate-200 whitespace-nowrap">${group.quests.length} quests</span>
                        </div>` : '';

                    const itemsHtml = group.quests.map(q => {
                        const idx = indexById.get(String(q.id));
                        const tags = [];
                        if (q.type) tags.push(`<span class="inline-block text-[10px] uppercase tracking-wide font-semibold text-indigo-700 bg-indigo-50 border border-indigo-100 px-1.5 py-0.5 rounded">${escapeHtml(q.type)}</span>`);
                        if (q.region) tags.push(`<span class="inline-block text-[10px] uppercase tracking-wide font-semibold text-slate-600 bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded">${escapeHtml(q.region)}</span>`);
                        const subParts = [];
                        if (q.spotName) subParts.push(escapeHtml(q.spotName));
                        if (q.openedAt) subParts.push(`Open: ${escapeHtml(formatUnixTime(q.openedAt))}`);
                        return `
                            <div class="result-item selectable group" data-id="${escapeHtml(q.id)}" data-idx="${idx}" onclick="selectQuest(this, '${escapeHtml(q.id)}', ${idx})">
                                <div class="cursor-pointer group-hover:bg-emerald-50/70 rounded-lg px-3 py-2 border border-transparent group-hover:border-emerald-200 transition-colors">
                                    <div class="flex items-start justify-between gap-3">
                                        <div class="min-w-0 flex-1">
                                            <div class="flex items-center gap-2 flex-wrap">
                                                <span class="font-mono text-[11px] text-slate-400">#${escapeHtml(q.id)}</span>
                                                ${tags.join('')}
                                            </div>
                                            <div class="font-medium text-sm text-slate-900 mt-0.5 truncate">${escapeHtml(q.name || '(unnamed)')}</div>
                                            ${subParts.length ? `<div class="text-xs text-slate-500 mt-0.5 truncate">${subParts.join(' · ')}</div>` : ''}
                                        </div>
                                        ${q.warName && warOrder.length === 1 ? `<div class="text-[11px] text-slate-400 whitespace-nowrap pt-1">${escapeHtml(q.warName)}</div>` : ''}
                                    </div>
                                </div>
                            </div>`;
                    }).join('');

                    return `
                        <section class="bg-white border border-slate-200 rounded-xl overflow-hidden">
                            ${headerHtml}
                            <div class="divide-y divide-slate-100">${itemsHtml}</div>
                        </section>`;
                }).join('');

            listEl.innerHTML = sectionHtml;
        }

        function renderActivityBanner(activity, wars) {
            const el = document.getElementById('questActivityBanner');
            if (!el) return;
            if (!activity && (!wars || wars.length === 0)) {
                el.classList.add('hidden');
                el.innerHTML = '';
                return;
            }
            // Prefer event banner; fall back to noticeBanner; fall back to first war banner/map.
            let bannerUrl = '';
            let title = '';
            let subtitle = '';
            if (activity) {
                bannerUrl = activity.banner || activity.noticeBanner || '';
                title = activity.name || '';
                const dateParts = [];
                if (activity.startedAt) dateParts.push(`Start ${formatUnixTime(activity.startedAt)}`);
                if (activity.endedAt) dateParts.push(`End ${formatUnixTime(activity.endedAt)}`);
                subtitle = dateParts.join(' · ');
            }
            if (!bannerUrl && wars && wars.length) {
                const w = wars.find(w => w.banner) || wars[0];
                bannerUrl = w.banner || w.mapImage || '';
                if (!title) title = w.name || w.longName || '';
            }
            if (!bannerUrl && !title) {
                el.classList.add('hidden');
                el.innerHTML = '';
                return;
            }
            el.classList.remove('hidden');
            el.innerHTML = `
                <div class="relative bg-slate-900 overflow-hidden">
                    ${bannerUrl ? `<img src="${escapeHtml(bannerUrl)}" alt="" class="w-full max-h-52 object-cover" loading="lazy" onerror="this.parentElement.classList.add('hidden')"/>` : ''}
                    ${title ? `<div class="absolute bottom-0 left-0 right-0 px-4 py-2.5 bg-gradient-to-t from-black/80 via-black/40 to-transparent text-white">
                        <div class="text-base font-semibold drop-shadow-sm">${escapeHtml(title)}</div>
                        ${subtitle ? `<div class="text-xs text-white/80">${escapeHtml(subtitle)}</div>` : ''}
                    </div>` : ''}
                </div>`;
        }

        async function selectQuest(elem, questId, idx) {
            document.querySelectorAll('#questResultList .result-item.selectable').forEach(e => e.style.background = '');
            elem.style.background = '#e3f2fd';
            // Resolve selected quest object
            if (typeof idx !== 'undefined') {
                selectedQuest = questResults[idx];
            } else {
                selectedQuest = questResults.find(q => q.id == questId);
            }
            currentQuestRegion = (selectedQuest && selectedQuest.region) || currentQuestRegion || 'JP';
            selectedPhaseScripts = [];

            // Show the translate section with a loading spinner in the phase picker
            const phasePickerEl = document.getElementById('phasePicker');
            phasePickerEl.classList.remove('hidden');
            phasePickerEl.innerHTML = `
                <div class="flex items-center gap-3 text-slate-500 text-sm">
                    <svg class="animate-spin w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                    Loading quest phases…
                </div>`;
            document.getElementById('translationOptions').style.display = 'none';
            document.getElementById('translationDisplay').style.display = 'none';
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('translateSection').style.display = '';

            try {
                const resp = await fetch('/get_quest_detail', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ quest_id: selectedQuest.id, region: currentQuestRegion })
                });
                const questDetail = await resp.json();
                if (questDetail.error) throw new Error(questDetail.error);

                const phaseScripts = questDetail.phaseScripts || [];
                const phasesNoBattle = new Set(questDetail.phasesNoBattle || []);
                const phasesWithEnemies = new Set(questDetail.phasesWithEnemies || []);

                if (phaseScripts.length === 0) {
                    phasePickerEl.innerHTML = `<p class="text-sm text-slate-500 italic">No phase scripts found for this quest.</p>`;
                    return;
                }

                const questName = escapeHtml(questDetail.name || selectedQuest.name || `Quest ${selectedQuest.id}`);
                // Store scripts by phase so buttons don't need to embed JSON in onclick attrs
                _phaseScriptsMap = {};
                phaseScripts.forEach(ps => { _phaseScriptsMap[ps.phase] = ps.scripts || []; });

                const phaseButtonsHtml = phaseScripts.map(ps => {
                    const ph = ps.phase;
                    const scriptCount = (ps.scripts || []).length;
                    const hasBattle = phasesWithEnemies.has(ph);
                    const noBattle = phasesNoBattle.has(ph);
                    const tag = hasBattle
                        ? `<span class="ml-1.5 text-[9px] uppercase tracking-wide font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-1 rounded">battle</span>`
                        : noBattle
                            ? `<span class="ml-1.5 text-[9px] uppercase tracking-wide font-semibold text-sky-700 bg-sky-50 border border-sky-200 px-1 rounded">story</span>`
                            : '';
                    return `<button
                        id="phaseBtn-${ph}"
                        onclick="selectPhase(${ph})"
                        class="flex flex-col items-center gap-1 px-4 py-2.5 rounded-lg border border-slate-200 bg-white hover:bg-indigo-50 hover:border-indigo-300 transition-colors text-sm font-medium text-slate-700 min-w-[80px]">
                        <span class="text-base font-bold text-indigo-700">Phase ${ph}</span>
                        <span class="text-[10px] text-slate-400">${scriptCount} script${scriptCount !== 1 ? 's' : ''}</span>
                        ${tag}
                    </button>`;
                }).join('');

                phasePickerEl.innerHTML = `
                    <div class="mb-3">
                        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Quest</p>
                        <p class="text-sm font-medium text-slate-800">${questName}</p>
                    </div>
                    <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Select a Phase to Translate</p>
                    <div class="flex flex-wrap gap-2">${phaseButtonsHtml}</div>`;

            } catch (err) {
                phasePickerEl.innerHTML = `<p class="text-sm text-red-600">Failed to load phases: ${escapeHtml(err.message)}</p>`;
            }
        }

        function selectPhase(phase) {
            selectedPhaseScripts = _phaseScriptsMap[phase] || [];
            // Highlight the active phase button
            document.querySelectorAll('[id^="phaseBtn-"]').forEach(btn => {
                btn.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-600');
                btn.classList.add('bg-white', 'text-slate-700', 'border-slate-200');
            });
            const activeBtn = document.getElementById(`phaseBtn-${phase}`);
            if (activeBtn) {
                activeBtn.classList.remove('bg-white', 'text-slate-700', 'border-slate-200');
                activeBtn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-600');
            }
            // Show translation options
            document.getElementById('translationOptions').style.display = '';
            document.getElementById('translationDisplay').style.display = '';
            // Clear any previous results
            document.getElementById('translationResults').innerHTML = '';
            document.getElementById('scriptTabs').classList.add('hidden');
            document.getElementById('progressSection').classList.add('hidden');
        }

        // Add animation for progress bar
        function animateProgressBar(targetWidth, duration = 1000) {
            const progressBar = document.getElementById('progressBar');
            const startWidth = parseInt(progressBar.style.width) || 0;
            const startTime = performance.now();
            
            function updateProgress(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const currentWidth = startWidth + (targetWidth - startWidth) * progress;
                progressBar.style.width = `${currentWidth}%`;
                
                if (progress < 1) {
                    requestAnimationFrame(updateProgress);
                }
            }
            
            requestAnimationFrame(updateProgress);
        }

        // Create script tab
        function createScriptTab(scriptId, isActive = false) {
            const tab = document.createElement('button');
            tab.className = `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`;
            tab.textContent = `Script ${scriptId}`;
            tab.dataset.scriptId = scriptId;
            tab.onclick = () => switchScriptTab(scriptId);
            return tab;
        }

        // Switch between script tabs
        function switchScriptTab(scriptId) {
            document.querySelectorAll('#scriptTabs button').forEach(tab => {
                tab.classList.toggle('bg-blue-600', tab.dataset.scriptId === scriptId.toString());
                tab.classList.toggle('text-white', tab.dataset.scriptId === scriptId.toString());
                tab.classList.toggle('bg-gray-100', tab.dataset.scriptId !== scriptId.toString());
                tab.classList.toggle('text-gray-700', tab.dataset.scriptId !== scriptId.toString());
            });
            
            document.querySelectorAll('.script-content').forEach(content => {
                content.classList.toggle('hidden', content.dataset.scriptId !== scriptId.toString());
            });
        }

        async function translateSelectedQuest() {
            console.log('translateSelectedQuest called');
            if (!selectedQuest) {
                console.log('No quest selected');
                return;
            }
            if (!selectedPhaseScripts || selectedPhaseScripts.length === 0) {
                alert('Please select a phase first.');
                return;
            }
            // Show progress section
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('scriptTabs').classList.remove('hidden');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            const scriptTabsContainer = document.querySelector('#scriptTabs .flex');
            scriptTabsContainer.innerHTML = '';
            document.getElementById('translationResults').innerHTML = '';
            try {
                // Use the scripts for the selected phase only
                const scriptIds = selectedPhaseScripts
                    .map(s => s.scriptId)
                    .filter(id => id && id !== 0);

                if (scriptIds.length === 0) {
                    throw new Error('No scripts found for this phase');
                }
                console.log('Script IDs for selected phase:', scriptIds);
                progressText.textContent = `Found ${scriptIds.length} script${scriptIds.length !== 1 ? 's' : ''}`;
                animateProgressBar(20);

                // Create tabs for each script
                scriptIds.forEach((scriptId, index) => {
                    const tab = createScriptTab(scriptId, index === 0);
                    scriptTabsContainer.appendChild(tab);
                });

                // Process each script: extract dialogues
                const allDialogues = [];
                const scriptDialogueCounts = [];
                for (let i = 0; i < scriptIds.length; i++) {
                    const scriptId = scriptIds[i];
                    progressText.textContent = `Processing script ${scriptId} (${i + 1}/${scriptIds.length})`;
                    animateProgressBar(20 + (i / scriptIds.length) * 30);
                    console.log('Fetching dialogues for script', scriptId);
                    const extractResponse = await fetch('/extract_dialogues', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ script_id: scriptId, region: currentQuestRegion })
                    });
                    const extractData = await extractResponse.json();
                    if (extractData.error) throw new Error(extractData.error);
                    console.log('Dialogues for script', scriptId, extractData.dialogues);
                    // Create content container for this script
                    const contentContainer = document.createElement('div');
                    contentContainer.className = `script-content ${i === 0 ? '' : 'hidden'}`;
                    contentContainer.dataset.scriptId = scriptId;
                    contentContainer.innerHTML = `
                        <div class="flex items-center cursor-pointer font-semibold text-base text-purple-900 mb-2" onclick="toggleScriptCollapse('${scriptId}')">
                            <span>Script ${scriptId}</span>
                            <svg id="script-arrow-${scriptId}" class="ml-2 w-4 h-4 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                        </div>
                        <div id="script-table-wrap-${scriptId}">
                            <table class="w-full mt-2 border-separate" style="border-spacing:0 8px;">
                                <thead>
                                    <tr>
                                        <th class="w-1/2 text-left text-gray-700">Original</th>
                                        <th class="w-1/2 text-left text-gray-700">Translation</th>
                                    </tr>
                                </thead>
                                <tbody id="dialogueTableBody-${scriptId}"></tbody>
                            </table>
                            <button class="mt-2 bg-gray-700 text-white px-3 py-1 rounded hover:bg-gray-900" onclick="downloadScript('${scriptId}')">Download</button>
                        </div>
                    `;
                    document.getElementById('translationResults').appendChild(contentContainer);
                    scriptDialogueCounts.push(extractData.dialogues.length);
                    allDialogues.push(...extractData.dialogues);
                }
                console.log('All dialogues:', allDialogues);
                // Translate all dialogues
                progressText.textContent = 'Translating dialogues...';
                animateProgressBar(60);
                currentTranslationSession = Date.now().toString();
                const translateResponse = await fetch('/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dialogues: allDialogues,
                        translation_method: document.getElementById('translationMethod').value,
                        target_language: document.getElementById('targetLanguage').value,
                        session_id: currentTranslationSession
                    })
                });
                const translateData = await translateResponse.json();
                if (translateData.error) throw new Error(translateData.error);
                console.log('Translation result:', translateData);
                let dialogueIndex = 0;
                for (let s = 0; s < scriptIds.length; s++) {
                    const scriptId = scriptIds[s];
                    const tableBody = document.getElementById(`dialogueTableBody-${scriptId}`);
                    const count = scriptDialogueCounts[s];
                    const scriptOriginals = allDialogues.slice(dialogueIndex, dialogueIndex + count);
                    const scriptTranslations = translateData.translated_dialogues.slice(dialogueIndex, dialogueIndex + count);
                    for (let j = 0; j < count; j++) {
                        const orig = scriptOriginals[j] || {};
                        const trans = scriptTranslations[j] || {};
                        const formattedOriginal = `<span class='font-bold text-blue-700 mr-2'>${orig.speaker || ''}</span>` + (orig.content || '').replace(/\[(.*?)\]/g, '<span class="text-purple-600">[$1]</span>');
                        const formattedTranslated = `<span class='font-bold text-green-700 mr-2'>${trans.speaker || ''}</span>` + (trans.translated_content || '').replace(/\[(.*?)\]/g, '<span class="text-purple-600">[$1]</span>');
                        tableBody.innerHTML += `
                            <tr>
                                <td class="align-top bg-gray-100 text-gray-800 p-2 rounded">${formattedOriginal}</td>
                                <td class="align-top bg-green-50 text-green-800 p-2 rounded">${formattedTranslated}</td>
                            </tr>
                        `;
                    }
                    dialogueIndex += count;
                }
                progressText.textContent = 'Translation completed!';
                animateProgressBar(100);
            } catch (error) {
                console.error('translateSelectedQuest error:', error);
                progressText.textContent = `Error: ${error.message}`;
                animateProgressBar(0);
                alert('Error during translation: ' + error.message);
            } finally {
                hideLoading();
            }
        }

        function toggleSettings() {
            const settingsMenu = document.getElementById('settingsMenu');
            const willOpen = settingsMenu.classList.contains('hidden');
            settingsMenu.classList.toggle('hidden');
            if (willOpen) {
                // Refresh fields with the latest server-side preferences each time the modal opens.
                loadUserPreferences().catch(err => console.error('loadUserPreferences error:', err));
            }
        }

        const apiTypeDefaults = {
            openai: {
                api_base: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                base_model: 'deepseek-v3',
                auth_type: 'api_key'
            },
            custom: {
                api_base: 'https://api.openai.com',
                base_model: 'gpt-4',
                auth_type: 'bearer'
            },
            gemini: {
                api_base: 'https://generativelanguage.googleapis.com/v1beta',
                base_model: 'gemini-2.5-flash',
                auth_type: 'api_key'
            }
        };

        function getApiDefaults(apiType) {
            return apiTypeDefaults[apiType] || apiTypeDefaults.openai;
        }

        function syncApiPlaceholders(apiType) {
            const defaults = getApiDefaults(apiType);
            document.getElementById('apiBase').placeholder = defaults.api_base;
            document.getElementById('baseModel').placeholder = defaults.base_model;
        }

        function applyApiTypeDefaults() {
            const apiType = document.getElementById('apiType').value;
            const defaults = getApiDefaults(apiType);
            const apiBase = document.getElementById('apiBase');
            const baseModel = document.getElementById('baseModel');
            const authType = document.getElementById('authType');
            const knownApiBases = Object.values(apiTypeDefaults).map(item => item.api_base);
            const knownModels = Object.values(apiTypeDefaults).map(item => item.base_model);

            if (!apiBase.value || knownApiBases.includes(apiBase.value)) {
                apiBase.value = defaults.api_base;
            }
            if (!baseModel.value || knownModels.includes(baseModel.value)) {
                baseModel.value = defaults.base_model;
            }
            authType.value = defaults.auth_type;
            syncApiPlaceholders(apiType);
        }

        function saveSettings() {
            const apiKey = document.getElementById('apiKey').value;
            const apiBase = document.getElementById('apiBase').value;
            const apiType = document.getElementById('apiType').value;
            const baseModel = document.getElementById('baseModel').value;
            const authType = document.getElementById('authType').value;

            // 显示加载状态
            showLoading();

            // 发送设置到后端
            fetch('/save_preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    api_base: apiBase,
                    api_type: apiType,
                    base_model: baseModel,
                    auth_type: authType
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    alert('Error saving settings: ' + data.error);
                } else {
                    // 更新UI显示
                    if (data.preferences) {
                        document.getElementById('apiKey').value = data.preferences.api_key || '';
                        document.getElementById('apiBase').value = data.preferences.api_base || '';
                        document.getElementById('apiType').value = data.preferences.api_type || 'openai';
                        document.getElementById('baseModel').value = data.preferences.base_model || '';
                        document.getElementById('authType').value = data.preferences.auth_type || 'api_key';
                        syncApiPlaceholders(data.preferences.api_type || 'openai');
                    }
                    alert('Settings saved successfully!');
                    toggleSettings();
                }
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                alert('Error saving settings: ' + error.message);
            });
        }

        // Check for user_preferences.db and load settings
        async function loadUserPreferences() {
            try {
                const response = await fetch('/get_preferences');
                if (response.ok) {
                    const preferences = await response.json();
                    // Set default values if preferences are empty
                    const selectedApiType = preferences.api_type || 'openai';
                    const defaultValues = {
                        api_key: '',
                        api_type: selectedApiType,
                        ...getApiDefaults(selectedApiType)
                    };

                    // Update input fields with preferences or default values
                    document.getElementById('apiKey').value = preferences.api_key || defaultValues.api_key;
                    document.getElementById('apiBase').value = preferences.api_base || defaultValues.api_base;
                    document.getElementById('apiType').value = preferences.api_type || defaultValues.api_type;
                    document.getElementById('baseModel').value = preferences.base_model || defaultValues.base_model;
                    document.getElementById('authType').value = preferences.auth_type || defaultValues.auth_type;

                    // Also update the placeholders to show default values
                    syncApiPlaceholders(selectedApiType);
                }
            } catch (error) {
                console.error('Error loading user preferences:', error);
                // Set default values if loading fails
                const defaults = getApiDefaults('openai');
                document.getElementById('apiBase').value = defaults.api_base;
                document.getElementById('apiType').value = 'openai';
                document.getElementById('baseModel').value = defaults.base_model;
                document.getElementById('authType').value = defaults.auth_type;
                syncApiPlaceholders('openai');
            }
        }

        // Call loadUserPreferences when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            loadUserPreferences();
            loadLatestTasks();
            document.getElementById('apiType').addEventListener('change', applyApiTypeDefaults);
        });

        // 翻译展示区可折叠
        function toggleScriptCollapse(scriptId) {
            const content = document.getElementById(`script-table-wrap-${scriptId}`);
            const arrow = document.getElementById(`script-arrow-${scriptId}`);
            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                arrow.style.transform = 'rotate(180deg)';
            } else {
                content.classList.add('hidden');
                arrow.style.transform = '';
            }
        }

        // 下载按钮功能
        function downloadScript(scriptId) {
            const tableBody = document.getElementById(`dialogueTableBody-${scriptId}`);
            if (!tableBody) return;
            let text = '';
            for (const row of tableBody.children) {
                const orig = row.children[0].innerText;
                const trans = row.children[1].innerText;
                text += `${orig}\n${trans}\n\n`;
            }
            const blob = new Blob([text], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `script_${scriptId}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        // war/quest container折叠
        function toggleContainer(containerId, arrowId) {
            const container = document.getElementById(containerId);
            const arrow = document.getElementById(arrowId);
            if (container.classList.contains('hidden')) {
                container.classList.remove('hidden');
                if (arrow) arrow.style.transform = 'rotate(180deg)';
            } else {
                container.classList.add('hidden');
                if (arrow) arrow.style.transform = '';
            }
        }

        // 在翻译选项区增加目标语言设置按钮
        function showLanguageSelector() {
            const langInput = document.getElementById('targetLanguage');
            langInput.focus();
        }
    
        function switchDiscoveryTab(panelId, btnElement) {
            ['searchPanel', 'latestActPanel', 'latestTaskPanel'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
            document.getElementById(panelId).classList.remove('hidden');
            
            document.querySelectorAll('.discovery-tab').forEach(el => {
                el.classList.replace('text-indigo-600', 'text-slate-500');
                el.classList.replace('border-indigo-600', 'border-transparent');
                el.classList.remove('bg-white');
            });
            
            btnElement.classList.replace('text-slate-500', 'text-indigo-600');
            btnElement.classList.replace('border-transparent', 'border-indigo-600');
            btnElement.classList.add('bg-white');
        }

        // Hook to switch tab and hide right panel when specific operations run
        const originalSearchActivity = searchActivity;
        searchActivity = async function() {
            await originalSearchActivity();
            document.getElementById('warResultInfo').innerHTML = document.getElementById('warResultInfo').innerText;
        };
        const originalLoadLatestTasks = loadLatestTasks;
        loadLatestTasks = async function() {
            await originalLoadLatestTasks();
            document.getElementById('latestTaskInfo').innerHTML = document.getElementById('latestTaskInfo').innerText;
        };

    