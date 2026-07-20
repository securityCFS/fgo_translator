from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from dialogue_loader import DialogueLoader
import os
from dotenv import load_dotenv
from flask_cors import CORS
import json
import sqlite3
import asyncio
import re
from pathlib import Path
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor
from translation_cache import (
    TranslationCacheClient,
    TranslationCacheConfig,
    TranslationCacheEntry,
    TranslationCacheKey,
    canonical_source_hash,
    normalize_provider,
    normalize_target_language,
)

# In-memory cache for svtScript metadata (entityId -> dict)
_SVT_SCRIPT_CACHE = {}

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
loader = DialogueLoader()

# Load user preferences from user_preferences.db if it exists
def load_user_preferences():
    preferences = {}
    if os.path.exists('user_preferences.db'):
        with sqlite3.connect('user_preferences.db') as conn:
            cursor = conn.execute("SELECT key, value FROM preferences")
            for key, value in cursor.fetchall():
                preferences[key] = value
    return preferences

# Load user preferences
user_preferences = load_user_preferences()
translation_cache_config = TranslationCacheConfig.from_env()
translation_cache_client = TranslationCacheClient(translation_cache_config)
BUNDLED_TRANSLATION_ROOT = Path(__file__).resolve().parent / 'translations'

@app.route('/')
def index():
    response = make_response(send_from_directory('templates', 'index.html'))
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/settings')
def settings():
    response = make_response(send_from_directory('templates', 'settings.html'))
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/docs/<path:filename>')
def docs(filename):
    response = make_response(send_from_directory('docs', filename))
    if filename.lower().endswith('.md'):
        response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
    return response


@app.route('/translations/<path:filename>')
def bundled_translation_file(filename):
    response = make_response(send_from_directory(BUNDLED_TRANSLATION_ROOT, filename))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.headers['Cache-Control'] = 'public, max-age=300'
    return response

@app.route('/search_war', methods=['POST'])
def search_war():
    print("Entering search_war")
    data = request.get_json() or {}
    war_name = data.get('war_name')
    region = data.get('region')
    limit = int(data.get('limit', 50))
    if war_name is None:
        war_name = ''
    print(f"Searching for war: {war_name}")
    wars = loader.search_war(war_name, region=region, limit=limit)
    print(f"Found {len(wars)} wars")
    return jsonify({'wars': wars})

@app.route('/search_event', methods=['POST'])
def search_event():
    data = request.get_json() or {}
    event_name = data.get('event_name') or ''
    region = data.get('region', 'JP')
    limit = int(data.get('limit', 50))
    events = loader.search_event(event_name, region=region, limit=limit)
    return jsonify({'events': events, 'region': loader.normalize_region(region)})

@app.route('/search_quest', methods=['POST'])
def search_quest():
    data = request.get_json()
    war_id = data.get('war_id')
    event_id = data.get('event_id')
    war_ids = data.get('war_ids')
    region = data.get('region', 'JP')
    print(f"Searching for quest in war/event: {war_id or event_id or war_ids} ({region})")
    try:
        region = loader.normalize_region(region)

        activity_info = None
        if event_id:
            try:
                nice_event_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/event/{event_id}"
                event = loader.db_loader._make_request_with_retry(nice_event_endpoint, max_retries=1)
                war_ids = [str(w) for w in event.get('warIds', [])]
                activity_info = {
                    'kind': 'event',
                    'id': str(event.get('id', event_id)),
                    'name': event.get('name', ''),
                    'banner': event.get('banner') or '',
                    'noticeBanner': event.get('noticeBanner') or '',
                    'startedAt': event.get('startedAt'),
                    'endedAt': event.get('endedAt'),
                    'type': event.get('type', ''),
                }
            except Exception as e:
                print(f"Failed to fetch nice event {event_id}, falling back to basic: {e}")
                event_endpoint = f"{loader.db_loader.BASE_URL}/basic/{region}/event/{event_id}"
                event = loader.db_loader._make_request_with_retry(event_endpoint)
                war_ids = [str(w) for w in event.get('warIds', [])]
                activity_info = {
                    'kind': 'event',
                    'id': str(event.get('id', event_id)),
                    'name': event.get('name', ''),
                    'banner': '',
                    'noticeBanner': '',
                }
        elif war_ids is None:
            if not war_id:
                return jsonify({'error': 'War ID or Event ID is required'}), 400
            war_ids = [war_id]
        elif not isinstance(war_ids, list):
            war_ids = [war_ids]

        quest_list = []
        war_info_list = []
        errors = []
        seen_quest_ids = set()
        for current_war_id in war_ids:
            try:
                requested_war_id = str(current_war_id)
                war_endpoint = f"{loader.db_loader.BASE_URL}/raw/{region}/war/{current_war_id}"
                war = loader.db_loader._make_request_with_retry(war_endpoint)
                if not war:
                    errors.append(f"War {current_war_id} not found")
                    continue

                quests = war.get('mstQuest', [])
                resolved_war_id = str(current_war_id)
                try:
                    numeric_war_id = int(str(current_war_id))
                except ValueError:
                    numeric_war_id = 0
                if not quests and numeric_war_id >= 10000 and str(current_war_id).endswith('01'):
                    parent_war_id = str(numeric_war_id // 100)
                    try:
                        parent_endpoint = f"{loader.db_loader.BASE_URL}/raw/{region}/war/{parent_war_id}"
                        parent_war = loader.db_loader._make_request_with_retry(parent_endpoint, max_retries=1)
                        if parent_war and parent_war.get('mstQuest'):
                            print(f"Resolved area-board shortcut war {current_war_id} -> {parent_war_id}")
                            war = parent_war
                            quests = war.get('mstQuest', [])
                            resolved_war_id = parent_war_id
                    except Exception as e:
                        print(f"Failed to resolve shortcut war {current_war_id}: {e}")

                war_meta = {
                    'id': resolved_war_id,
                    'name': war.get('mstWar', {}).get('name', ''),
                    'longName': war.get('mstWar', {}).get('longName', ''),
                    'banner': '',
                    'mapImage': '',
                }
                if resolved_war_id != requested_war_id:
                    war_meta['shortcutId'] = requested_war_id
                # Fetch nice war for banner/map image (best-effort, cached by retry layer)
                nice_war = None
                map_lookup = {}
                nice_spot_lookup = {}
                try:
                    nice_war_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/war/{resolved_war_id}"
                    nice_war = loader.db_loader._make_request_with_retry(nice_war_endpoint, max_retries=1)
                    if nice_war:
                        war_meta['name'] = nice_war.get('name', war_meta['name'])
                        war_meta['longName'] = nice_war.get('longName', war_meta['longName'])
                        war_meta['banner'] = nice_war.get('banner') or ''
                        maps = nice_war.get('maps') or []
                        if maps:
                            war_meta['mapImage'] = maps[0].get('mapImage') or ''
                            war_meta['mapImageW'] = maps[0].get('mapImageW')
                            war_meta['mapImageH'] = maps[0].get('mapImageH')
                        map_lookup = {str(m.get('id')): m for m in maps or []}
                        nice_spot_lookup = {
                            str(sp.get('id')): sp
                            for sp in nice_war.get('spots', []) or []
                        }
                except Exception as e:
                    print(f"Failed to fetch nice war {resolved_war_id}: {e}")

                war_info_list.append(war_meta)

                print(f"Got {len(quests)} quests from war {resolved_war_id}")
                # Build a lookup of mstSpot for spot names
                spot_lookup = {sp.get('id'): sp.get('name', '') for sp in war.get('mstSpot', [])}
                for quest in quests:
                    qraw = quest['mstQuest']
                    quest_id = str(qraw['id'])
                    if quest_id in seen_quest_ids:
                        continue
                    seen_quest_ids.add(quest_id)
                    try:
                        quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/quest/{quest_id}"
                        quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
                        if quest_data:
                            phase_scripts = []
                            script_ids = []
                            for phase_script in quest_data.get('phaseScripts', []) or []:
                                scripts = []
                                for script in phase_script.get('scripts', []) or []:
                                    script_id = str(script.get('scriptId') or script.get('id') or '')
                                    if not script_id or script_id == '0':
                                        continue
                                    scripts.append({
                                        'scriptId': script_id,
                                        'script': script.get('script', ''),
                                    })
                                    if script_id not in script_ids:
                                        script_ids.append(script_id)
                                if scripts:
                                    phase_scripts.append({
                                        'phase': phase_script.get('phase'),
                                        'scripts': scripts,
                                    })
                            quest_list.append({
                                'id': quest_id,
                                'name': quest_data.get('name', ''),
                                'type': quest_data.get('type', ''),
                                'afterClear': quest_data.get('afterClear', ''),
                                'spotName': quest_data.get('spotName', '') or spot_lookup.get(qraw.get('spotId'), ''),
                                'spotId': quest_data.get('spotId') or qraw.get('spotId'),
                                'spotImage': '',
                                'questOfsX': None,
                                'questOfsY': None,
                                'nameOfsX': None,
                                'nameOfsY': None,
                                'mapId': None,
                                'mapImage': '',
                                'mapImageW': None,
                                'mapImageH': None,
                                'spotX': None,
                                'spotY': None,
                                'phases': quest_data.get('phases', []),
                                'phasesNoBattle': quest_data.get('phasesNoBattle', []),
                                'phasesWithEnemies': quest_data.get('phasesWithEnemies', []),
                                'phaseScripts': phase_scripts,
                                'scriptIds': script_ids,
                                'scriptCount': len(script_ids),
                                'hasDialogueScript': bool(script_ids),
                                'openedAt': quest_data.get('openedAt'),
                                'closedAt': quest_data.get('closedAt'),
                                'region': region,
                                'warId': resolved_war_id,
                                'warName': war_meta['name'],
                            })
                            added = quest_list[-1]
                            nice_spot = nice_spot_lookup.get(str(added.get('spotId'))) or {}
                            map_id = nice_spot.get('mapId')
                            map_meta = map_lookup.get(str(map_id)) or {}
                            added['spotName'] = added.get('spotName') or nice_spot.get('name', '')
                            added['mapId'] = str(map_id) if map_id is not None else ''
                            added['mapImage'] = map_meta.get('mapImage') or ''
                            added['mapImageW'] = map_meta.get('mapImageW')
                            added['mapImageH'] = map_meta.get('mapImageH')
                            added['spotX'] = nice_spot.get('x')
                            added['spotY'] = nice_spot.get('y')
                            added['spotImage'] = nice_spot.get('image') or ''
                            added['questOfsX'] = nice_spot.get('questOfsX')
                            added['questOfsY'] = nice_spot.get('questOfsY')
                            added['nameOfsX'] = nice_spot.get('nameOfsX')
                            added['nameOfsY'] = nice_spot.get('nameOfsY')
                    except Exception as e:
                        error_msg = f"Failed to get quest {quest_id}: {str(e)}"
                        print(error_msg)
                        errors.append(error_msg)
            except Exception as e:
                error_msg = f"Failed to get war {current_war_id}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)

        response = {
            'quests': quest_list,
            'wars': war_info_list,
            'activity': activity_info,
        }
        if errors:
            response['error'] = f"由于错误 {', '.join(errors)}，列表请求不完整"
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/latest_tasks', methods=['POST'])
def latest_tasks():
    data = request.get_json() or {}
    try:
        region = loader.normalize_region(data.get('region', 'JP'))
        limit = int(data.get('limit', 50))
        tasks = loader.list_latest_tasks(region=region, limit=limit)
        hidden_no_script_count = max((int(task.get('hiddenNoScriptCount', 0) or 0) for task in tasks), default=0)
        scanned_count = max((int(task.get('scannedCount', 0) or 0) for task in tasks), default=0)
        return jsonify({
            'tasks': tasks,
            'region': region,
            'hiddenNoScriptCount': hidden_no_script_count,
            'scannedCount': scanned_count,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/latest_activities', methods=['POST'])
def latest_activities():
    data = request.get_json() or {}
    try:
        region = loader.normalize_region(data.get('region', 'JP'))
        limit = int(data.get('limit', 50))
        activity_type = data.get('activity_type', 'event')
        with_wars = data.get('with_wars', True)
        activities = loader.list_latest_activities(
            region=region,
            activity_type=activity_type,
            limit=limit,
            with_wars=with_wars
        )
        return jsonify({'activities': activities, 'region': region, 'activity_type': activity_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_scripts', methods=['POST'])
def get_scripts():
    quest_id = str(request.json.get('quest_id'))
    region = request.json.get('region', 'JP')
    if not quest_id:
        return jsonify({'error': 'Quest ID is required'}), 400
    scripts = loader.get_quest_scripts(quest_id, region=region)
    return jsonify({'scripts': scripts})

@app.route('/extract_dialogues', methods=['POST'])
def extract_dialogues():
    script_id = str(request.json.get('script_id'))
    region = request.json.get('region', 'JP')
    if not script_id:
        return jsonify({'error': 'Script ID is required'}), 400
    print(f"Extracting dialogues for script: {script_id} ({region})")
    dialogues = loader.extract_dialogues(script_id, region=region)
    return jsonify({'dialogues': dialogues})


@app.route('/parse_script_visual', methods=['POST'])
def parse_script_visual():
    """Parse a raw FGO script into visual frames for gaming mode."""
    data = request.get_json() or {}
    script_id = str(data.get('script_id', ''))
    region = loader.normalize_region(data.get('region', 'JP'))
    if not script_id:
        return jsonify({'error': 'script_id required'}), 400
    try:
        raw = loader.load_script_text(script_id, region=region)
        if not raw:
            return jsonify({'error': 'No script text found'}), 404
        frames, entity_ids = _parse_fgo_script(raw, region)
        svt_data = _fetch_svt_scripts_parallel(region, entity_ids)
        return jsonify({
            'frames': frames,
            'svtData': svt_data,
            'region': region,
            'scriptId': script_id,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _fetch_svt_scripts_parallel(region: str, entity_ids):
    """Fetch svtScript metadata for many entities in parallel, with caching."""
    result = {}
    to_fetch = []
    for eid in entity_ids:
        key = f"{region}:{eid}"
        if key in _SVT_SCRIPT_CACHE:
            result[str(eid)] = _SVT_SCRIPT_CACHE[key]
        else:
            to_fetch.append(eid)
    if not to_fetch:
        return result

    def _one(eid):
        try:
            meta = loader.db_loader._make_request_with_retry(
                f"{loader.db_loader.BASE_URL}/raw/{region}/svtScript?charaId={eid}"
            )
            if isinstance(meta, list) and meta:
                m = meta[0]
                return eid, {
                    'faceX': m.get('faceX', 0),
                    'faceY': m.get('faceY', 0),
                    'offsetX': m.get('offsetX', 0),
                    'offsetY': m.get('offsetY', 0),
                    'scale': m.get('scale', 1.0),
                    'extendData': m.get('extendData', {}),
                }
        except Exception as ex:
            print(f"svtScript lookup failed for {eid}: {ex}")
        return eid, None

    with ThreadPoolExecutor(max_workers=12) as pool:
        for eid, meta in pool.map(_one, to_fetch):
            key = f"{region}:{eid}"
            if meta is not None:
                _SVT_SCRIPT_CACHE[key] = meta
                result[str(eid)] = meta
    return result


def _parse_fgo_script(raw_text: str, region: str = 'JP'):
    """Parse a raw FGO script into a list of visual frames."""
    import re

    CDN = 'https://static.atlasacademy.io'
    # The merged figure contains the body plus face pages. The frontend also
    # loads the plain figure to obtain the exact body dimensions.
    FIG_BASE = f'{CDN}/{region}/CharaFigure/{{eid}}/{{eid}}_merged.png'

    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\[\s*\n\s*', '[', text)
    text = re.sub(r'\n\s*(?=[^\[＠？?\n])', ' ', text)
    text = text.replace('[%1]', '藤丸立香').replace('[r]', '\n')
    # Note: formatting tags [line N], [align ...], [f ...], [image ...] are
    # preserved here and rendered by the frontend (gaming.html / index.html).
    lines = text.splitlines()

    def clean_text(s: str) -> str:
        # Preserve ruby/furigana ([#base:reading]) and renderable formatting
        # tags; strip every other bracketed command (e.g. [se ...], [wt 30],
        # [charaFace ...], etc.). Ruby with no reading ([#text]) is unwrapped.
        s = re.sub(r'\[#([^\[\]:]+)\](?!:)', r'\1', s)
        _PRESERVE = re.compile(
            r'\[(?:'
            r'#[^\[\]:]+:[^\[\]]+'              # [#base:reading] ruby
            r'|align(?:\s+\w+)?'
            r'|line\s+\d+'
            r'|f\s+[\w-]+'
            r'|/f'
            r'|image\s+[\w-]+'
            r')\]',
            re.IGNORECASE,
        )
        out = []
        last = 0
        for m in _PRESERVE.finditer(s):
            chunk = s[last:m.start()]
            chunk = re.sub(r'\[[^\[\]]+\]', '', chunk)
            out.append(chunk)
            out.append(m.group(0))
            last = m.end()
        tail = s[last:]
        tail = re.sub(r'\[[^\[\]]+\]', '', tail)
        out.append(tail)
        return ''.join(out).strip()

    state = {
        'bg': '',
        'sprites': {},   # slot -> {entityId, name, face, visible}
        'subLayers': {},
        'subRenders': {},
        'talkers': set(),
        'talkHighlightEnabled': True,
        'cameraFilter': None,  # active color tint
        'bgm': None,           # active BGM name
        'fullScreen': False,
        'activitySeq': 0,
    }
    frames = []
    dialogue_idx = 0
    pending_effects = []
    entity_ids = set()
    visual_dirty = False

    # Pre-scan choice groups so we can emit the choice popup BEFORE any branch
    # response dialogues, and tag branch dialogues with their branchId.
    # The translation list (built in dialogue_loader) enumerates entries in
    # textual order: each ＠..[k] block, each ？N：text, and each ？！ each
    # consume one index. We mirror that count here to compute the dialogueIdx
    # of each ？N choice and the trailing ？！ end marker.
    def _scan_choice_groups():
        idx = 0
        groups = []  # [{ 'first_line': i, 'end_line': i, 'choices': [{num,text,dialogueIdx}], 'end_dialogueIdx': int }]
        cur = None
        j = 0
        while j < len(lines):
            ln = lines[j].strip()
            if ln.startswith('＠'):
                # consume until [k]
                jj = j + 1
                while jj < len(lines):
                    if '[k]' in lines[jj]:
                        jj += 1
                        break
                    jj += 1
                idx += 1
                j = jj
                continue
            cm = re.match(r'[？?](\d+)[：:](.+)', ln)
            if cm:
                if cur is None:
                    cur = {'first_line': j, 'choices': [], 'end_dialogueIdx': None, 'end_line': None}
                cur['choices'].append({
                    'num': int(cm.group(1)),
                    'text': clean_text(cm.group(2).strip()),
                    'dialogueIdx': idx,
                })
                idx += 1
                j += 1
                continue
            if re.match(r'^(?:？！|\?!)', ln):
                if cur is not None:
                    cur['end_dialogueIdx'] = idx
                    cur['end_line'] = j
                    groups.append(cur)
                    cur = None
                idx += 1
                j += 1
                continue
            j += 1
        # Trailing unclosed group (no ？！) — close it anyway
        if cur is not None:
            cur['end_dialogueIdx'] = idx
            cur['end_line'] = len(lines)
            groups.append(cur)
        return groups

    choice_groups = _scan_choice_groups()
    # Map from line index -> group for fast lookup at first ？N
    group_by_first_line = {g['first_line']: g for g in choice_groups}
    current_group = None       # active group while between ？1 and ？！
    current_branch = None      # selected branch number for tagging dialogues

    def take_effects():
        nonlocal pending_effects
        e = pending_effects
        pending_effects = []
        return e

    def mark_visual_dirty():
        nonlocal visual_dirty
        visual_dirty = True

    def is_renderable_sprite(entity_id, name=''):
        label = str(name or '')
        entity_id = str(entity_id or '')
        return (
            entity_id != '98115000'
            and 'エフェクト用' not in label
            and '初期化用ダミー' not in label
        )

    def background_url(background_id):
        suffix = '_1344_626' if state['fullScreen'] else ''
        return f'{CDN}/{region}/Back/back{background_id}{suffix}.png'

    def image_url(image_name):
        return f'{CDN}/{region}/Image/{image_name}/{image_name}.png'

    def is_variant_sprite(name=''):
        label = str(name or '').lower()
        return any(marker in label for marker in ('演出用', 'シルエット', 'silhouette'))

    def touch_sprite(slot):
        if slot not in state['sprites']:
            return
        state['activitySeq'] += 1
        state['sprites'][slot]['activity'] = state['activitySeq']

    def set_sprite_visible(slot, visible):
        if slot in state['sprites']:
            state['sprites'][slot]['visible'] = visible
            if not visible:
                state['talkers'].discard(slot)

    def set_sprite_opacity(slot, opacity):
        if slot not in state['sprites']:
            return
        opacity = max(0.0, min(1.0, float(opacity)))
        state['sprites'][slot]['opacity'] = opacity
        set_sprite_visible(slot, opacity > 0)
        if opacity > 0:
            touch_sprite(slot)

    def parse_filter_color(raw):
        value = str(raw or '000000FF').strip().lstrip('#')
        if not re.fullmatch(r'[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?', value):
            return '#000000', 1.0
        color = f'#{value[:6]}'
        alpha = int(value[6:8], 16) / 255 if len(value) == 8 else 1.0
        return color, alpha

    def parse_position_token(token):
        token = str(token or '').strip()
        if not token:
            return None
        if ',' in token:
            raw_x, raw_y = token.split(',', 1)
            try:
                return float(raw_x), float(raw_y)
            except ValueError:
                return None
        try:
            preset = int(token)
        except ValueError:
            return None
        return ({0: -256.0, 1: 0.0, 2: 256.0}.get(preset, float(preset)), 0.0)

    def set_sprite_position(slot, token):
        position = parse_position_token(token)
        if slot not in state['sprites'] or position is None:
            return
        state['sprites'][slot]['x'], state['sprites'][slot]['y'] = position

    def get_sub_render(layer_id):
        return state['subRenders'].setdefault(layer_id, {
            'visible': False,
            'x': 0.0,
            'y': 0.0,
            'scale': 1.0,
            'depth': None,
            'mask': None,
        })

    def sub_layer_for_slot(slot):
        for layer_id, slots in state['subLayers'].items():
            if slot in slots:
                return layer_id
        return None

    def visible_sprite_entries():
        result = []
        for slot, sp in state['sprites'].items():
            if sp.get('visible') and sp.get('entityId') and sp.get('renderable') is not False:
                layer_id = sub_layer_for_slot(slot)
                sub_render = get_sub_render(layer_id) if layer_id else None
                if sub_render and not sub_render['visible']:
                    continue
                eid = sp['entityId']
                asset_type = sp.get('assetType', 'chara')
                result.append({
                    'slot': slot,
                    'entityId': eid,
                    'assetType': asset_type,
                    'name': sp.get('name', ''),
                    'face': sp.get('face', 1),
                    'url': sp.get('url') or FIG_BASE.format(eid=eid),
                    'talking': (
                        asset_type != 'chara'
                        or not state['talkHighlightEnabled']
                        or slot in state['talkers']
                    ),
                    'x': sp.get('x'),
                    'y': sp.get('y'),
                    'scale': sp.get('scale', 1.0),
                    'depth': sp.get('depth', 0),
                    'opacity': sp.get('opacity', 1.0),
                    'filter': sp.get('filter', 'normal'),
                    'filterColor': sp.get('filterColor', '#000000'),
                    'filterAlpha': sp.get('filterAlpha', 1.0),
                    'subRender': layer_id,
                    '_activity': sp.get('activity', 0),
                    '_variant': asset_type == 'chara' and is_variant_sprite(sp.get('name', '')),
                })
        return result

    def snapshot_sprites():
        candidates = visible_sprite_entries()
        by_entity = {}
        for sprite in candidates:
            by_entity.setdefault(sprite['entityId'], []).append(sprite)

        suppressed_slots = set()
        for same_entity in by_entity.values():
            # FGO frequently preloads a normal, silhouette, and cinematic
            # variant of the same graph in parallel. Only the most recently
            # operated variant is visible; older variants remain available and
            # can become visible again after the active slot fades out.
            if len(same_entity) > 1 and any(sprite['_variant'] for sprite in same_entity):
                active = max(same_entity, key=lambda sprite: sprite['_activity'])
                suppressed_slots.update(sprite['slot'] for sprite in same_entity if sprite is not active)

        result = []
        for sprite in candidates:
            if sprite['slot'] in suppressed_slots:
                continue
            sprite.pop('_activity', None)
            sprite.pop('_variant', None)
            result.append(sprite)
        return result

    def snapshot_sub_renders():
        return {
            layer_id: {
                'visible': bool(render.get('visible')),
                'x': render.get('x', 0.0),
                'y': render.get('y', 0.0),
                'scale': render.get('scale', 1.0),
                'depth': render.get('depth'),
                'mask': render.get('mask'),
            }
            for layer_id, render in state['subRenders'].items()
        }

    def append_frame(frame):
        nonlocal visual_dirty
        frame.setdefault('branchId', current_branch)
        frame.setdefault('subRenders', snapshot_sub_renders())
        frames.append(frame)
        visual_dirty = False

    def emit_visual_wait(duration):
        if not visual_dirty:
            return
        append_frame({
            'type': 'stage',
            'bg': state['bg'],
            'sprites': snapshot_sprites(),
            'duration': max(0.0, float(duration)),
            'effects': take_effects(),
            'cameraFilter': state['cameraFilter'],
            'bgm': state['bgm'],
        })

    def normalize_speaker_label(value):
        label = clean_text(str(value or '')).strip()
        label = re.sub(r'[_＿](?:演出用|シルエット).*$', '', label)
        return re.sub(r'\s+', '', label).casefold()

    def infer_talker_from_speaker(speaker):
        if not state['talkHighlightEnabled']:
            return
        visible = visible_sprite_entries()
        normalized_speaker = normalize_speaker_label(speaker)
        matches = []
        if normalized_speaker:
            matches = [
                sprite for sprite in visible
                if normalize_speaker_label(sprite.get('name', '')) == normalized_speaker
            ]
        if matches:
            matching_slots = {sprite['slot'] for sprite in matches}
            if state['talkers'] & matching_slots:
                return
            active = max(matches, key=lambda sprite: sprite.get('_activity', 0))
            state['talkers'] = {active['slot']}
            return

        visible_slots = {sprite['slot'] for sprite in visible}
        if state['talkers'] & visible_slots:
            return
        if len(visible) == 1 and normalized_speaker:
            state['talkers'] = {visible[0]['slot']}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        if line.startswith('[enableFullScreen'):
            state['fullScreen'] = True
            continue

        # Background commands
        m = re.match(r'\[scene\s+(\d+)(?:\s+[^\]]+)?\]', line)
        if m:
            state['bg'] = background_url(m.group(1))
            mark_visual_dirty()
            continue

        # bScene: multi-layer bg, take first id (it's the base background)
        # Format: [bScene id1,id2,id3] where ids may have garbled separators
        m = re.match(r'\[bScene\s+(\d+)', line)
        if m:
            # Only set if not yet set (don't overwrite a real scene)
            if not state['bg']:
                state['bg'] = background_url(m.group(1))
                mark_visual_dirty()
            continue

        # sceneSet/imageSet declare reusable stage layers; fade commands reveal them.
        m = re.match(r'\[sceneSet\s+(\w+)\s+(\d+)\s*(\d+)?', line)
        if m:
            slot, scene_id = m.group(1), m.group(2)
            state['sprites'][slot] = {
                'entityId': f'scene:{scene_id}',
                'assetType': 'scene',
                'url': background_url(scene_id),
                'name': f'back{scene_id}',
                'face': 0,
                'visible': False,
                'renderable': True,
                'x': None,
                'y': None,
                'scale': 1.0,
                'depth': 0,
                'opacity': 1.0,
                'filter': 'normal',
                'filterColor': '#000000',
                'filterAlpha': 1.0,
                'activity': 0,
            }
            continue

        m = re.match(r'\[(?:imageSet|verticalImageSet|horizontalImageSet)\s+(\w+)\s+([^\s\]]+)', line)
        if m:
            slot, image_name = m.group(1), m.group(2)
            state['sprites'][slot] = {
                'entityId': f'image:{image_name}',
                'assetType': 'image',
                'url': image_url(image_name),
                'name': image_name,
                'face': 0,
                'visible': False,
                'renderable': True,
                'x': None,
                'y': None,
                'scale': 1.0,
                'depth': 0,
                'opacity': 1.0,
                'filter': 'normal',
                'filterColor': '#000000',
                'filterAlpha': 1.0,
                'activity': 0,
            }
            continue

        m = re.match(r'\[charaSet\s+(\w)\s+(\d+)\s+(\d+)\s*(.*?)\]', line)
        if m:
            slot, eid, face, name = m.group(1), m.group(2), int(m.group(3)), m.group(4).strip()
            state['sprites'][slot] = {
                'entityId': eid,
                'assetType': 'chara',
                'url': FIG_BASE.format(eid=eid),
                'name': name,
                'face': face,
                'visible': False,
                'renderable': is_renderable_sprite(eid, name),
                'x': None,
                'y': None,
                'scale': 1.0,
                'depth': 0,
                'opacity': 1.0,
                'filter': 'normal',
                'filterColor': '#000000',
                'filterAlpha': 1.0,
                'activity': 0,
            }
            entity_ids.add(eid)
            continue

        m = re.match(r'\[charaFace(?:Fade)?\s+(\w)\s+(\d+)', line)
        if m:
            slot, face = m.group(1), int(m.group(2))
            if slot in state['sprites']:
                state['sprites'][slot]['face'] = face
                touch_sprite(slot)
                if state['sprites'][slot].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[charaTalk\s+([^\]]+)\]', line)
        if m:
            raw_talkers = m.group(1).strip()
            if raw_talkers == 'off':
                state['talkHighlightEnabled'] = False
                state['talkers'].clear()
            elif raw_talkers == 'on':
                state['talkHighlightEnabled'] = True
                state['talkers'].clear()
            elif raw_talkers in ('depthOff', 'depthOn'):
                pass
            else:
                state['talkHighlightEnabled'] = True
                state['talkers'] = {slot for slot in raw_talkers.split(',') if slot in state['sprites']}
            continue

        m = re.match(r'\[(charaFadein\w*|overlayFadein)\s+(\w+)\s+([^\]]+)\]', line)
        if m:
            args = m.group(3).split()
            slot = m.group(2)
            if len(args) >= 2:
                set_sprite_position(slot, args[1])
            if slot in state['sprites']:
                state['sprites'][slot]['opacity'] = 1.0
            set_sprite_visible(slot, True)
            touch_sprite(slot)
            mark_visual_dirty()
            continue

        m = re.match(r'\[charaFadeout\w*\s+(\w)', line)
        if m:
            set_sprite_visible(m.group(1), False)
            mark_visual_dirty()
            continue

        m = re.match(r'\[charaPut\w*\s+(\w+)\s+([^\s\]]+)', line)
        if m:
            set_sprite_position(m.group(1), m.group(2))
            # charaPut places a preloaded layer but does not change its alpha.
            # A following charaFadeTime/charaFadein is what reveals it.
            if m.group(1) in state['sprites'] and state['sprites'][m.group(1)].get('visible'):
                touch_sprite(m.group(1))
                mark_visual_dirty()
            continue

        m = re.match(r'\[charaFadeTime\w*\s+(\w+)\s+[^\s\]]+\s+([\d.]+)', line)
        if m:
            set_sprite_opacity(m.group(1), m.group(2))
            mark_visual_dirty()
            continue

        m = re.match(r'\[charaFilter\s+(\w+)\s+(\w+)(?:\s+([^\s\]]+))?', line)
        if m:
            slot, mode = m.group(1), m.group(2).lower()
            if slot in state['sprites']:
                state['sprites'][slot]['filter'] = mode
                if mode == 'silhouette':
                    color, alpha = parse_filter_color(m.group(3))
                    state['sprites'][slot]['filterColor'] = color
                    state['sprites'][slot]['filterAlpha'] = alpha
                else:
                    state['sprites'][slot]['filterColor'] = '#000000'
                    state['sprites'][slot]['filterAlpha'] = 1.0
                if state['sprites'][slot].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[charaMoveScale(?:Ease)?\s+(\w+)\s+([\d.]+)', line)
        if m:
            if m.group(1) in state['sprites']:
                state['sprites'][m.group(1)]['scale'] = float(m.group(2))
                touch_sprite(m.group(1))
                if state['sprites'][m.group(1)].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[(charaMove(?!Return|Scale)\w*)\s+(\w+)\s+([^\s\]]+)', line)
        if m:
            set_sprite_position(m.group(2), m.group(3))
            touch_sprite(m.group(2))
            if m.group(2) in state['sprites'] and state['sprites'][m.group(2)].get('visible'):
                mark_visual_dirty()
            continue

        m = re.match(r'\[charaScale\s+(\w+)\s+([\d.]+)', line)
        if m:
            if m.group(1) in state['sprites']:
                state['sprites'][m.group(1)]['scale'] = float(m.group(2))
                touch_sprite(m.group(1))
                if state['sprites'][m.group(1)].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[charaDepth\s+(\w+)\s+(-?\d+)', line)
        if m:
            if m.group(1) in state['sprites']:
                state['sprites'][m.group(1)]['depth'] = int(m.group(2))
                if state['sprites'][m.group(1)].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[charaLayer\s+(\w)\s+sub\s+(#[A-Z])', line)
        if m:
            state['subLayers'].setdefault(m.group(2), set()).add(m.group(1))
            mark_visual_dirty()
            continue

        m = re.match(r'\[charaLayer\s+(\w)\s+(?:main|normal)', line)
        if m:
            slot = m.group(1)
            for slots in state['subLayers'].values():
                slots.discard(slot)
            mark_visual_dirty()
            continue

        m = re.match(r'\[subCameraFilter(?:\s+(#[A-Z]))?\s+maskEdge\s+([^\s\]]+)', line)
        if m:
            get_sub_render(m.group(1) or '#A')['mask'] = m.group(2)
            mark_visual_dirty()
            continue

        m = re.match(r'\[subRenderDepth\s+(#[A-Z])\s+(-?\d+)', line)
        if m:
            get_sub_render(m.group(1))['depth'] = int(m.group(2))
            mark_visual_dirty()
            continue

        m = re.match(r'\[subRenderFadein\w*\s+(#[A-Z])\s+[^\s\]]+\s+([^\s\]]+)', line)
        if m:
            render = get_sub_render(m.group(1))
            position = parse_position_token(m.group(2))
            if position is not None:
                render['x'], render['y'] = position
            render['visible'] = True
            mark_visual_dirty()
            continue

        m = re.match(r'\[subRender(?:MoveScale(?:Ease)?|Scale)\s+(#[A-Z])\s+([\d.]+)', line)
        if m:
            get_sub_render(m.group(1))['scale'] = float(m.group(2))
            mark_visual_dirty()
            continue

        m = re.match(r'\[subRenderMove(?!Scale)\w*\s+(#[A-Z])\s+([^\s\]]+)', line)
        if m:
            render = get_sub_render(m.group(1))
            position = parse_position_token(m.group(2))
            if position is not None:
                render['x'], render['y'] = position
            mark_visual_dirty()
            continue

        m = re.match(r'\[charaCrossFade\s+(\w)\s+(\d+)\s+(\d+)', line)
        if m:
            slot, eid, face = m.group(1), m.group(2), int(m.group(3))
            if slot in state['sprites']:
                state['sprites'][slot]['entityId'] = eid
                state['sprites'][slot]['url'] = FIG_BASE.format(eid=eid)
                state['sprites'][slot]['face'] = face
                state['sprites'][slot]['renderable'] = is_renderable_sprite(
                    eid,
                    state['sprites'][slot].get('name', '')
                )
                touch_sprite(slot)
                entity_ids.add(eid)
                if state['sprites'][slot].get('visible'):
                    mark_visual_dirty()
            continue

        m = re.match(r'\[subRenderFadeout\w*\s+(#[A-Z])', line)
        if m:
            get_sub_render(m.group(1))['visible'] = False
            mark_visual_dirty()
            continue

        m = re.match(r'\[wt\s+([\d.]+)\s*\]', line)
        if m:
            emit_visual_wait(m.group(1))
            continue

        if line.startswith('＠'):
            # In FGO scripts the speaker name immediately follows ＠ with NO space:
            #   ＠ゴルドルフ          → speaker "ゴルドルフ", content on next line
            #   ＠F：マシュ           → slot-prefixed speaker
            #   ＠                    → empty speaker (narration)
            # Preprocessing collapses content lines: ＠\ncontent → ＠ content
            # (space inserted). So ＠ followed by a space means EMPTY speaker and
            # content that was on the next line has been collapsed in.
            raw_after = line[1:]   # everything after ＠, NOT stripped
            if raw_after == '' or raw_after[0] == ' ':
                # Empty speaker — collapsed content (if any) follows after the space
                speaker = ''
                initial_content = raw_after.strip()
            else:
                # Speaker name or slot prefix starts immediately after ＠
                speaker_raw = raw_after.strip()
                initial_content = ''
                sp_split = speaker_raw.split(None, 1)
                if len(sp_split) == 2:
                    speaker_raw = sp_split[0]
                    initial_content = sp_split[1]
                slot_prefix = re.match(r'^([A-Z])：(.+)$', speaker_raw)
                if slot_prefix:
                    speaker_slot = slot_prefix.group(1)
                    speaker = slot_prefix.group(2).strip()
                    if speaker_slot in state['sprites']:
                        if speaker_slot not in state['talkers']:
                            state['talkHighlightEnabled'] = True
                            state['talkers'] = {speaker_slot}
                else:
                    speaker = speaker_raw

            content_parts = []
            if initial_content:
                # initial_content may already contain `[k]`; split there
                if '[k]' in initial_content:
                    before_k = initial_content[:initial_content.index('[k]')].strip()
                    if before_k:
                        content_parts.append(before_k)
                    content = '\n'.join(content_parts).strip()
                    content = clean_text(content)
                    if content:
                        infer_talker_from_speaker(speaker)
                        append_frame({
                            'type': 'dialogue',
                            'bg': state['bg'],
                            'sprites': snapshot_sprites(),
                            'speaker': speaker,
                            'text': content,
                            'dialogueIdx': dialogue_idx,
                            'branchId': current_branch,
                            'effects': take_effects(),
                            'cameraFilter': state['cameraFilter'],
                            'bgm': state['bgm'],
                        })
                    dialogue_idx += 1
                    continue
                content_parts.append(initial_content)
            while i < len(lines):
                cline = lines[i].strip()
                i += 1
                if '[k]' in cline:
                    before_k = cline[:cline.index('[k]')].strip()
                    if before_k:
                        content_parts.append(before_k)
                    break
                if cline:
                    content_parts.append(cline)
            content = '\n'.join(content_parts).strip()
            content = clean_text(content)
            if content:
                infer_talker_from_speaker(speaker)
                append_frame({
                    'type': 'dialogue',
                    'bg': state['bg'],
                    'sprites': snapshot_sprites(),
                    'speaker': speaker,
                    'text': content,
                    'dialogueIdx': dialogue_idx,
                    'branchId': current_branch,
                    'effects': take_effects(),
                    'cameraFilter': state['cameraFilter'],
                    'bgm': state['bgm'],
                })
            # Always advance index whether or not we emitted (mirror the
            # translation list which counts every ＠..[k] block).
            dialogue_idx += 1
            continue

        m = re.match(r'[？?](\d+)[：:](.+)', line)
        if m:
            num = int(m.group(1))
            # Determine which group this choice belongs to (by line index of first ？N).
            # i was already advanced past this line, so the line index is i-1.
            line_idx = i - 1
            if current_group is None:
                # First ？N of a new group: emit the choice popup BEFORE branch dialogues.
                # Find the group whose first_line is at or before this position.
                # In practice the first ？N is the group's first_line.
                grp = group_by_first_line.get(line_idx)
                if grp is None:
                    # Fallback: scan groups for one containing this line
                    for g in choice_groups:
                        if g['first_line'] <= line_idx <= (g['end_line'] or len(lines)):
                            grp = g
                            break
                if grp is not None:
                    current_group = grp
                    append_frame({
                        'type': 'choice',
                        'bg': state['bg'],
                        'sprites': snapshot_sprites(),
                        'choices': list(grp['choices']),
                        'dialogueIdx': grp['choices'][0]['dialogueIdx'] if grp['choices'] else dialogue_idx,
                        'endDialogueIdx': grp['end_dialogueIdx'],
                        'effects': take_effects(),
                        'cameraFilter': state['cameraFilter'],
                        'bgm': state['bgm'],
                    })
            current_branch = num
            dialogue_idx += 1  # ？N consumes one translation slot
            continue

        if re.match(r'^(?:？！|\?!)', line):
            current_group = None
            current_branch = None
            dialogue_idx += 1  # ？！ consumes one translation slot ("Choice N Ending")
            continue

        # ----- Visual effect commands (accumulated until next visible frame) -----
        m = re.match(r'\[criMovie\s+([^\s\]]+)', line)
        if m:
            for slot in list(state['sprites'].keys()):
                set_sprite_visible(slot, False)
            append_frame({
                'type': 'movie',
                'bg': state['bg'],
                'sprites': [],
                'movie': m.group(1),
                'effects': [
                    {'type': 'fadeOut', 'color': 'black', 'dur': 0.35},
                    {'type': 'movie', 'name': m.group(1)},
                ],
                'cameraFilter': state['cameraFilter'],
                'bgm': state['bgm'],
            })
            continue

        m = re.match(r'\[fadeout\s+(\w+)(?:\s+([\d.]+))?\s*\]', line)
        if m:
            color, dur = m.group(1), float(m.group(2) or 1.0)
            pending_effects.append({'type': 'fadeOut', 'color': color, 'dur': dur})
            append_frame({
                'type': 'transition',
                'bg': state['bg'],
                'sprites': snapshot_sprites(),
                'effects': take_effects(),
                'cameraFilter': state['cameraFilter'],
                'bgm': state['bgm'],
            })
            continue

        m = re.match(r'\[fadein\s+(\w+)(?:\s+([\d.]+))?\s*\]', line)
        if m:
            color, dur = m.group(1), float(m.group(2) or 1.0)
            pending_effects.append({'type': 'fadeIn', 'color': color, 'dur': dur})
            mark_visual_dirty()
            continue

        m = re.match(r'\[cameraFilter\s+(\w+)\s*\]', line)
        if m:
            state['cameraFilter'] = m.group(1)
            pending_effects.append({'type': 'cameraFilter', 'color': m.group(1)})
            mark_visual_dirty()
            continue

        if re.match(r'\[cameraFilter(Off|Stop)?\s*\]', line):
            state['cameraFilter'] = None
            pending_effects.append({'type': 'cameraFilter', 'color': None})
            mark_visual_dirty()
            continue

        m = re.match(r'\[effect\s+(\w+)\s*\]', line)
        if m:
            name = m.group(1).lower()
            kind = 'shake' if 'shake' in name else ('flash' if 'flash' in name else 'effect')
            pending_effects.append({'type': kind, 'name': m.group(1)})
            mark_visual_dirty()
            continue

        m = re.match(r'\[bgm\s+(\w+)', line)
        if m:
            state['bgm'] = m.group(1)
            continue

        if re.match(r'\[bgmStop\b', line):
            state['bgm'] = None
            continue

    return frames, list(entity_ids)


@app.route('/gaming')
def gaming_mode():
    """Standalone visual novel gaming-mode popup window."""
    response = make_response(send_from_directory('templates', 'gaming.html'))
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return response


def _split_dialogues_by_script_counts(dialogues, script_ids, script_dialogue_counts):
    if not script_ids or not script_dialogue_counts:
        return {}
    if len(script_ids) != len(script_dialogue_counts):
        return {}

    result = {}
    offset = 0
    try:
        for script_id, raw_count in zip(script_ids, script_dialogue_counts):
            count = int(raw_count)
            if count < 0:
                return {}
            result[str(script_id)] = dialogues[offset:offset + count]
            offset += count
    except (TypeError, ValueError):
        return {}

    if offset != len(dialogues):
        return {}
    return result


def _filter_nonempty_script_dialogues(script_ids, script_dialogues):
    filtered_ids = [
        str(script_id)
        for script_id in script_ids
        if script_dialogues.get(str(script_id))
    ]
    return filtered_ids, {
        script_id: script_dialogues[script_id]
        for script_id in filtered_ids
    }


def _merge_script_translations(script_ids, translations_by_script):
    merged = []
    for script_id in script_ids:
        merged.extend(translations_by_script.get(str(script_id), []))
    return merged


def _has_translation_errors(translations):
    return any("[Translation Error:" in str(item.get("translated_content", "")) for item in translations)


def _cache_key_for_script(script_id, source_region, source_dialogues, target_language, api_type, base_model):
    return TranslationCacheKey(
        script_id=str(script_id),
        source_region=source_region,
        source_hash=canonical_source_hash(source_dialogues),
        target_language=normalize_target_language(target_language),
        provider=normalize_provider(api_type, base_model),
        model=base_model,
        prompt_version=translation_cache_config.prompt_version,
    )


def _entry_translations_with_source(entry, source_dialogues):
    return [
        {
            'speaker': cached.get('speaker') or source.get('speaker', ''),
            'content': source.get('content', ''),
            'translated_content': cached.get('translated_content', ''),
        }
        for source, cached in zip(source_dialogues, entry.translations)
    ]


def _cache_option_key(option):
    return (option.provider, option.model, option.prompt_version)


def _common_cache_options_for_scripts(script_sources, source_region, target_language, cache_client=None):
    cache_client = cache_client or translation_cache_client
    if not script_sources:
        return []

    common_keys = None
    options_by_key = {}
    total_dialogues = 0

    for script_id, source_dialogues in script_sources.items():
        total_dialogues += len(source_dialogues)
        source_hash = canonical_source_hash(source_dialogues)
        options = cache_client.list_options(
            script_id=str(script_id),
            source_region=source_region,
            source_hash=source_hash,
            target_language=target_language,
            expected_dialogue_count=len(source_dialogues),
        )
        keyed = {_cache_option_key(option): option for option in options}
        if common_keys is None:
            common_keys = set(keyed.keys())
        else:
            common_keys &= set(keyed.keys())
        for key, option in keyed.items():
            options_by_key.setdefault(key, option)

    result = []
    for key in sorted(common_keys or set()):
        option = options_by_key[key]
        result.append({
            'id': '||'.join(key),
            'provider': option.provider,
            'model': option.model,
            'prompt_version': option.prompt_version,
            'label': option.label,
            'script_count': len(script_sources),
            'dialogue_count': total_dialogues,
            'generated_at': option.generated_at,
        })
    return result


def _extract_script_sources(script_ids, source_region):
    sources = {}
    for script_id in script_ids:
        sources[str(script_id)] = loader.extract_dialogues(str(script_id), region=source_region)
    return sources


@app.route('/translation_cache_options', methods=['POST'])
def translation_cache_options():
    data = request.get_json() or {}
    script_ids = [str(script_id) for script_id in data.get('script_ids', []) if str(script_id)]
    source_region = loader.normalize_region(data.get('source_region', 'JP'))
    target_language = data.get('target_language', 'Chinese (Simplified)')

    if not script_ids:
        return jsonify({'error': 'script_ids required'}), 400
    if not translation_cache_config.repo:
        return jsonify({
            'options': [],
            'enabled': False,
            'reason': 'Translation cache repository is not configured.',
        })

    try:
        script_sources = _extract_script_sources(script_ids, source_region)
        options = _common_cache_options_for_scripts(script_sources, source_region, target_language)
        return jsonify({
            'options': options,
            'enabled': True,
            'target_language': normalize_target_language(target_language),
            'scripts': [
                {
                    'script_id': script_id,
                    'dialogue_count': len(source),
                    'source_hash': canonical_source_hash(source),
                }
                for script_id, source in script_sources.items()
            ],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/translate_cached', methods=['POST'])
def translate_cached():
    data = request.get_json() or {}
    script_ids = [str(script_id) for script_id in data.get('script_ids', []) if str(script_id)]
    source_region = loader.normalize_region(data.get('source_region', 'JP'))
    target_language = normalize_target_language(data.get('target_language', 'Chinese (Simplified)'))
    provider = str(data.get('provider', '')).strip()
    model = str(data.get('model', '')).strip()
    prompt_version = str(data.get('prompt_version', '')).strip() or translation_cache_config.prompt_version

    if not script_ids:
        return jsonify({'error': 'script_ids required'}), 400
    if not (provider and model):
        return jsonify({'error': 'provider and model required'}), 400
    if not translation_cache_config.enabled:
        return jsonify({'error': 'Translation cache is not configured'}), 503

    try:
        script_sources = _extract_script_sources(script_ids, source_region)
        translations_by_script = {}
        original_dialogues = []

        for script_id in script_ids:
            source_dialogues = script_sources.get(script_id, [])
            original_dialogues.extend(source_dialogues)
            key = TranslationCacheKey(
                script_id=script_id,
                source_region=source_region,
                source_hash=canonical_source_hash(source_dialogues),
                target_language=target_language,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
            )
            entry = translation_cache_client.read(key, expected_dialogue_count=len(source_dialogues))
            if not entry:
                return jsonify({'error': f'Cached translation not found for script {script_id}'}), 404
            translations_by_script[script_id] = _entry_translations_with_source(entry, source_dialogues)

        return jsonify({
            'original_dialogues': original_dialogues,
            'translated_dialogues': _merge_script_translations(script_ids, translations_by_script),
            'cache_hit': True,
            'cache_provider': provider,
            'cache_model': model,
            'cache_prompt_version': prompt_version,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json or {}
    dialogues = data.get('dialogues')
    translation_method = data.get('translation_method', 'gpt')
    target_language = data.get('target_language', 'Chinese')
    script_ids = [str(script_id) for script_id in data.get('script_ids', []) if str(script_id)]
    script_dialogue_counts = data.get('script_dialogue_counts', [])
    source_region = loader.normalize_region(data.get('source_region', 'JP'))
    script_dialogues = _split_dialogues_by_script_counts(dialogues or [], script_ids, script_dialogue_counts)
    script_ids, script_dialogues = _filter_nonempty_script_dialogues(script_ids, script_dialogues)
    script_dialogue_counts = [len(script_dialogues[script_id]) for script_id in script_ids]
    session_id = data.get('session_id')  # 用于标识翻译会话
    
    if not dialogues:
        return jsonify({'error': 'Dialogues are required'}), 400
    
    try:
        if translation_method == 'gpt':
            # 创建一个进度回调函数，支持 speaker
            def progress_callback(current, total, speaker=None):
                progress = int((current / total) * 100)
                socketio.emit('translation_progress', {
                    'session_id': session_id,
                    'progress': progress,
                    'current': current,
                    'total': total,
                    'speaker': speaker
                })
            
            api_type = (user_preferences.get('api_type') or os.getenv('API_TYPE', 'openai')).lower()
            if api_type == 'gemini':
                api_base = user_preferences.get('api_base') or os.getenv('GEMINI_API_BASE', 'https://generativelanguage.googleapis.com/v1beta')
                api_key = user_preferences.get('api_key') or os.getenv('GEMINI_API_KEY') or os.getenv('API_KEY')
                base_model = user_preferences.get('base_model') or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
            else:
                api_base = user_preferences.get('api_base') or os.getenv('API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
                api_key = user_preferences.get('api_key') or os.getenv('API_KEY')
                base_model = user_preferences.get('base_model') or os.getenv('BASE_MODEL', 'deepseek-v3')

            translated = None
            if script_dialogues:
                translations_by_script = {}
                miss_script_ids = []
                miss_dialogues = []
                miss_cache_keys = {}
                miss_sources = {}
                processed_count = 0

                for script_id in script_ids:
                    client_source = script_dialogues.get(script_id, [])
                    cache_source = []
                    try:
                        server_source = loader.extract_dialogues(script_id, region=source_region)
                        if len(server_source) == len(client_source):
                            cache_source = server_source
                    except Exception as cache_source_error:
                        print(f"Failed to re-extract script {script_id} for cache: {cache_source_error}")

                    key = None
                    if translation_cache_config.enabled and cache_source:
                        key = _cache_key_for_script(
                            script_id,
                            source_region,
                            cache_source,
                            target_language,
                            api_type,
                            base_model,
                        )
                        entry = translation_cache_client.read(key, expected_dialogue_count=len(cache_source))
                        if entry:
                            translations_by_script[script_id] = _entry_translations_with_source(entry, client_source)
                            processed_count += len(client_source)
                            progress_callback(min(processed_count, len(dialogues)), len(dialogues), f"Cache {script_id}")
                            continue

                    source_for_translation = cache_source or client_source
                    miss_script_ids.append(script_id)
                    miss_sources[script_id] = source_for_translation
                    if key:
                        miss_cache_keys[script_id] = key
                    miss_dialogues.extend(source_for_translation)

                if miss_dialogues:
                    def miss_progress_callback(current, total, speaker=None):
                        progress_callback(min(processed_count + current, len(dialogues)), len(dialogues), speaker)

                    miss_translated = loader.gpt_dialogue_translate(
                        miss_dialogues,
                        api_base=api_base,
                        api_key=api_key,
                        target_language=target_language,
                        base_model=base_model,
                        api_type=api_type,
                        auth_type=user_preferences.get('auth_type', 'api_key'),
                        progress_callback=miss_progress_callback
                    )

                    offset = 0
                    for script_id in miss_script_ids:
                        source = miss_sources.get(script_id, [])
                        translated_slice = miss_translated[offset:offset + len(source)]
                        offset += len(source)
                        translations_by_script[script_id] = translated_slice
                        key = miss_cache_keys.get(script_id)
                        if key and len(translated_slice) == len(source) and not _has_translation_errors(translated_slice):
                            translation_cache_client.write(TranslationCacheEntry(
                                key=key,
                                dialogue_count=len(source),
                                translations=translated_slice,
                            ))

                translated = _merge_script_translations(script_ids, translations_by_script)

            if translated is None:
                translated = loader.gpt_dialogue_translate(
                    dialogues,
                    api_base=api_base,
                    api_key=api_key,
                    target_language=target_language,
                    base_model=base_model,
                    api_type=api_type,
                    auth_type=user_preferences.get('auth_type', 'api_key'),
                    progress_callback=progress_callback
                )
        else:
            # 为免费翻译也添加进度回调，支持 speaker
            async def progress_callback(current, total, speaker=None):
                progress = int((current / total) * 100)
                socketio.emit('translation_progress', {
                    'session_id': session_id,
                    'progress': progress,
                    'current': current,
                    'total': total,
                    'speaker': speaker
                })
            
            translated = asyncio.run(loader.free_translate(
                dialogues, 
                target_language,
                progress_callback=progress_callback
            ))
        
        if len(translated) != len(dialogues):
            return jsonify({'error': 'Translation count mismatch'}), 500
        return jsonify({
            'original_dialogues': dialogues,
            'translated_dialogues': translated
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_preferences', methods=['GET'])
def get_preferences():
    preferences = load_user_preferences()
    return jsonify(preferences)

@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    try:
        data = request.json
        # 验证必要的字段
        required_fields = ['api_key', 'api_base', 'api_type', 'base_model', 'auth_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # 保存到数据库
        with sqlite3.connect('user_preferences.db') as conn:
            # 创建表（如果不存在）
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # 保存每个设置
            for key, value in data.items():
                conn.execute('INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)',
                           (key, value))
            
            conn.commit()
        
        # 重新加载用户偏好
        global user_preferences
        user_preferences = load_user_preferences()
        
        return jsonify({
            'message': 'Preferences saved successfully',
            'preferences': user_preferences  # 返回更新后的偏好设置
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_quest_detail', methods=['POST'])
def get_quest_detail():
    quest_id = str(request.json.get('quest_id'))
    region = request.json.get('region', 'JP')
    if not quest_id:
        return jsonify({'error': 'Quest ID is required'}), 400
    try:
        region = loader.normalize_region(region)
        quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/quest/{quest_id}"
        quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
        return jsonify(quest_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _bundled_translation_path(script_id, target_language='zh-CN'):
    script_id = str(script_id or '').strip()
    if not re.fullmatch(r'\d+', script_id):
        raise ValueError(f'Invalid script ID: {script_id}')
    language = normalize_target_language(target_language)
    if language != 'zh-CN':
        raise ValueError('Only bundled Simplified Chinese translations are available')
    return BUNDLED_TRANSLATION_ROOT / language / f'{script_id}.json'


def _load_bundled_translation(script_id, target_language='zh-CN'):
    path = _bundled_translation_path(script_id, target_language)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    key = TranslationCacheKey(
        script_id=str(script_id),
        source_region=str(data.get('source_region', 'JP')).upper(),
        source_hash=str(data.get('source_hash', '')),
        target_language=normalize_target_language(data.get('target_language', target_language)),
        provider=str(data.get('provider', 'codex-agent')),
        model=str(data.get('model', 'agent-translation')),
        prompt_version=str(data.get('prompt_version', 'fgo-agent-v1')),
    )
    entry = TranslationCacheEntry.from_json(
        data,
        key,
        expected_dialogue_count=int(data.get('dialogue_count', -1)),
    )
    return (data, entry) if entry else None


@app.route('/check_bundled_translations', methods=['POST'])
def check_bundled_translations():
    data = request.get_json() or {}
    script_ids = [str(value) for value in data.get('script_ids', []) if str(value)]
    target_language = data.get('target_language', 'zh-CN')
    if not script_ids:
        return jsonify({'available': False, 'missing': [], 'reason': 'script_ids required'})
    try:
        loaded = [_load_bundled_translation(script_id, target_language) for script_id in script_ids]
        missing = [script_id for script_id, item in zip(script_ids, loaded) if item is None]
        providers = sorted({item[0].get('provider', '') for item in loaded if item})
        return jsonify({
            'available': not missing,
            'missing': missing,
            'providers': providers,
            'target_language': normalize_target_language(target_language),
        })
    except Exception as exc:
        return jsonify({'available': False, 'error': str(exc)}), 400


@app.route('/get_bundled_dialogues', methods=['POST'])
def get_bundled_dialogues():
    data = request.get_json() or {}
    script_ids = [str(value) for value in data.get('script_ids', []) if str(value)]
    target_language = data.get('target_language', 'zh-CN')
    if not script_ids:
        return jsonify({'error': 'script_ids required'}), 400
    try:
        translations = []
        providers = set()
        for script_id in script_ids:
            loaded = _load_bundled_translation(script_id, target_language)
            if not loaded:
                return jsonify({'error': f'Bundled translation not found for script {script_id}'}), 404
            payload, entry = loaded
            source = loader.extract_dialogues(script_id, region=payload.get('source_region', 'JP'))
            if len(source) != entry.dialogue_count:
                return jsonify({'error': f'Bundled dialogue count mismatch for script {script_id}'}), 409
            if canonical_source_hash(source) != entry.key.source_hash:
                return jsonify({'error': f'Bundled source hash mismatch for script {script_id}'}), 409
            translations.extend(entry.translations)
            providers.add(entry.key.provider)
        return jsonify({
            'translated_dialogues': translations,
            'source_region': 'JP',
            'target_language': normalize_target_language(target_language),
            'providers': sorted(providers),
            'bundled': True,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

@app.route('/check_atlas_translations', methods=['POST'])
def check_atlas_translations():
    """Check which Atlas Academy regions have this quest (NA/CN/TW/KR) and whether Rayshift has it."""
    data = request.get_json() or {}
    quest_id = str(data.get('quest_id', ''))
    if not quest_id:
        return jsonify({'error': 'quest_id required'}), 400

    REGIONS_TO_CHECK = ['NA', 'CN', 'TW', 'KR']

    def check_region(region):
        try:
            import requests as req
            url = f"{loader.db_loader.BASE_URL}/basic/{region}/quest/{quest_id}"
            r = req.get(url, timeout=8)
            if r.status_code == 200:
                j = r.json()
                return region, bool(j and 'id' in j)
            return region, False
        except Exception:
            return region, False

    def check_rayshift():
        """Check Rayshift availability via the JP quest's first phase/script."""
        try:
            import requests as req
            # Get JP quest phase 1 to find the first script ID
            phase_url = f"{loader.db_loader.BASE_URL}/nice/JP/quest/{quest_id}/1"
            r = req.get(phase_url, timeout=8)
            if r.status_code != 200:
                return False
            scripts = r.json().get('scripts', [])
            if not scripts:
                return False
            first_script_id = str(scripts[0].get('scriptId', ''))
            if not first_script_id:
                return False
            rs = req.head(
                f"https://rayshift.io/api/v1/translate/check-ingame/{first_script_id}",
                timeout=8
            )
            return rs.status_code == 200
        except Exception:
            return False

    availability = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        region_futures = [executor.submit(check_region, r) for r in REGIONS_TO_CHECK]
        rayshift_future = executor.submit(check_rayshift)
        for future in region_futures:
            region, available = future.result()
            availability[region] = available
        has_rayshift = rayshift_future.result()

    # If Rayshift has a translation, mark NA as available (Rayshift provides English)
    if has_rayshift:
        availability['NA'] = True
    availability['rayshift'] = has_rayshift

    return jsonify(availability)


@app.route('/get_atlas_dialogues', methods=['POST'])
def get_atlas_dialogues():
    """Return dialogues from a specific Atlas Academy region as pre-translated text."""
    data = request.get_json() or {}
    script_ids = data.get('script_ids', [])
    target_region = str(data.get('target_region', 'NA')).upper()

    if not script_ids:
        return jsonify({'error': 'script_ids required'}), 400

    ALLOWED_REGIONS = {'NA', 'CN', 'TW', 'KR'}
    if target_region not in ALLOWED_REGIONS:
        return jsonify({'error': f'target_region must be one of {sorted(ALLOWED_REGIONS)}'}), 400

    try:
        all_dialogues = []
        for script_id in script_ids:
            dialogues = loader.extract_dialogues(str(script_id), region=target_region)
            all_dialogues.extend(dialogues)

        translated = [
            {
                'speaker': d.get('speaker', ''),
                # If extract_dialogues already paired a Rayshift translation, use it
                # directly; otherwise fall back to the raw 'content' field.
                'translated_content': d.get('translated_content', d.get('content', '')),
                'rayshift': d.get('rayshift', False),
            }
            for d in all_dialogues
        ]

        has_rayshift = any(d.get('rayshift') for d in all_dialogues)
        return jsonify({
            'translated_dialogues': translated,
            'source_region': target_region,
            'rayshift': has_rayshift,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True) 
