from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from dialogue_loader import DialogueLoader
import os
from dotenv import load_dotenv
from flask_cors import CORS
import json
import sqlite3
import asyncio
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor

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
                event = loader.db_loader._make_request_with_retry(nice_event_endpoint)
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
                war_endpoint = f"{loader.db_loader.BASE_URL}/raw/{region}/war/{current_war_id}"
                war = loader.db_loader._make_request_with_retry(war_endpoint)
                if not war:
                    errors.append(f"War {current_war_id} not found")
                    continue

                war_meta = {
                    'id': str(current_war_id),
                    'name': war.get('mstWar', {}).get('name', ''),
                    'longName': war.get('mstWar', {}).get('longName', ''),
                    'banner': '',
                    'mapImage': '',
                }
                # Fetch nice war for banner/map image (best-effort, cached by retry layer)
                try:
                    nice_war_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/war/{current_war_id}"
                    nice_war = loader.db_loader._make_request_with_retry(nice_war_endpoint)
                    if nice_war:
                        war_meta['name'] = nice_war.get('name', war_meta['name'])
                        war_meta['longName'] = nice_war.get('longName', war_meta['longName'])
                        war_meta['banner'] = nice_war.get('banner') or ''
                        maps = nice_war.get('maps') or []
                        if maps:
                            war_meta['mapImage'] = maps[0].get('mapImage') or ''
                except Exception as e:
                    print(f"Failed to fetch nice war {current_war_id}: {e}")

                war_info_list.append(war_meta)

                quests = war.get('mstQuest', [])
                print(f"Got {len(quests)} quests from war {current_war_id}")
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
                            quest_list.append({
                                'id': quest_id,
                                'name': quest_data.get('name', ''),
                                'type': quest_data.get('type', ''),
                                'spotName': quest_data.get('spotName', '') or spot_lookup.get(qraw.get('spotId'), ''),
                                'openedAt': quest_data.get('openedAt'),
                                'closedAt': quest_data.get('closedAt'),
                                'region': region,
                                'warId': str(current_war_id),
                                'warName': war_meta['name'],
                            })
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
        return jsonify({'tasks': tasks, 'region': region})
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
        script_endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/script/{script_id}"
        script_meta = loader.db_loader._make_request_with_retry(script_endpoint)
        script_url = script_meta.get('script', '')
        if not script_url:
            return jsonify({'error': 'No script URL found'}), 404
        import requests as req
        raw = req.get(script_url, timeout=15).text
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
    BG_BASE = f'{CDN}/{region}/Back/back{{id}}.png'
    # Raw atlas: body at top + face strip below. Frontend crops body and overlays
    # the requested face cell using svtScript metadata (faceX/faceY/faceSize).
    FIG_BASE = f'{CDN}/{region}/CharaFigure/{{eid}}/{{eid}}.png'

    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\[\s*\n\s*', '[', text)
    text = re.sub(r'\n\s*(?=[^\[＠？\n])', ' ', text)
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
        'talker': None,
        'cameraFilter': None,  # active color tint
        'bgm': None,           # active BGM name
    }
    frames = []
    dialogue_idx = 0
    pending_effects = []
    entity_ids = set()

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
            cm = re.match(r'？(\d+)：(.+)', ln)
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
            if ln.startswith('？！'):
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

    def snapshot_sprites():
        result = []
        for slot, sp in state['sprites'].items():
            if sp.get('visible') and sp.get('entityId'):
                eid = sp['entityId']
                result.append({
                    'slot': slot,
                    'entityId': eid,
                    'name': sp.get('name', ''),
                    'face': sp.get('face', 1),
                    'url': FIG_BASE.format(eid=eid),
                    'talking': (slot == state['talker']),
                })
        return result

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        # Background commands
        m = re.match(r'\[scene\s+(\d+)\]', line)
        if m:
            state['bg'] = BG_BASE.format(id=m.group(1))
            continue

        # bScene: multi-layer bg, take first id (it's the base background)
        # Format: [bScene id1,id2,id3] where ids may have garbled separators
        m = re.match(r'\[bScene\s+(\d+)', line)
        if m:
            # Only set if not yet set (don't overwrite a real scene)
            if not state['bg']:
                state['bg'] = BG_BASE.format(id=m.group(1))
            continue

        # imageSet I back10000 — slot I gets a "back" image; treat as bg fallback
        m = re.match(r'\[imageSet\s+\w\s+back(\d+)', line)
        if m:
            if not state['bg']:
                state['bg'] = BG_BASE.format(id=m.group(1))
            continue

        m = re.match(r'\[charaSet\s+(\w)\s+(\d+)\s+(\d+)\s*(.*?)\]', line)
        if m:
            slot, eid, face, name = m.group(1), m.group(2), int(m.group(3)), m.group(4).strip()
            state['sprites'][slot] = {'entityId': eid, 'name': name, 'face': face, 'visible': False}
            entity_ids.add(eid)
            continue

        m = re.match(r'\[charaFace\s+(\w)\s+(\d+)\]', line)
        if m:
            slot, face = m.group(1), int(m.group(2))
            if slot in state['sprites']:
                state['sprites'][slot]['face'] = face
            continue

        m = re.match(r'\[charaTalk\s+(\w+)\]', line)
        if m:
            s = m.group(1)
            state['talker'] = None if s in ('off', 'depthOff', 'on') else s
            continue

        m = re.match(r'\[charaFadein\s+(\w)', line)
        if m:
            slot = m.group(1)
            if slot in state['sprites']:
                state['sprites'][slot]['visible'] = True
            continue

        m = re.match(r'\[charaFadeout\s+(\w)', line)
        if m:
            slot = m.group(1)
            if slot in state['sprites']:
                state['sprites'][slot]['visible'] = False
            continue

        m = re.match(r'\[charaCrossFade\s+(\w)\s+(\d+)', line)
        if m:
            slot, eid = m.group(1), m.group(2)
            if slot in state['sprites']:
                state['sprites'][slot]['entityId'] = eid
                entity_ids.add(eid)
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
                        state['talker'] = speaker_slot
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
                        frames.append({
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
                frames.append({
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

        m = re.match(r'？(\d+)：(.+)', line)
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
                    frames.append({
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

        if line.startswith('？！'):
            current_group = None
            current_branch = None
            dialogue_idx += 1  # ？！ consumes one translation slot ("Choice N Ending")
            continue

        # ----- Visual effect commands (accumulated until next visible frame) -----
        m = re.match(r'\[fadeout\s+(\w+)(?:\s+([\d.]+))?\s*\]', line)
        if m:
            color, dur = m.group(1), float(m.group(2) or 1.0)
            pending_effects.append({'type': 'fadeOut', 'color': color, 'dur': dur})
            frames.append({
                'type': 'transition',
                'bg': state['bg'],
                'sprites': [],
                'effects': take_effects(),
                'cameraFilter': state['cameraFilter'],
                'bgm': state['bgm'],
            })
            continue

        m = re.match(r'\[fadein\s+(\w+)(?:\s+([\d.]+))?\s*\]', line)
        if m:
            color, dur = m.group(1), float(m.group(2) or 1.0)
            pending_effects.append({'type': 'fadeIn', 'color': color, 'dur': dur})
            continue

        m = re.match(r'\[cameraFilter\s+(\w+)\s*\]', line)
        if m:
            state['cameraFilter'] = m.group(1)
            pending_effects.append({'type': 'cameraFilter', 'color': m.group(1)})
            continue

        if re.match(r'\[cameraFilter(Off|Stop)?\s*\]', line):
            state['cameraFilter'] = None
            pending_effects.append({'type': 'cameraFilter', 'color': None})
            continue

        m = re.match(r'\[effect\s+(\w+)\s*\]', line)
        if m:
            name = m.group(1).lower()
            kind = 'shake' if 'shake' in name else ('flash' if 'flash' in name else 'effect')
            pending_effects.append({'type': kind, 'name': m.group(1)})
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


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    dialogues = data.get('dialogues')
    translation_method = data.get('translation_method', 'gpt')
    target_language = data.get('target_language', 'Chinese')
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

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True) 
