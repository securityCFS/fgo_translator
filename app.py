from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from dialogue_loader import DialogueLoader
import os
from dotenv import load_dotenv
from flask_cors import CORS
import json
import sqlite3
import asyncio

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
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
    return response

@app.route('/search_war', methods=['POST'])
def search_war():
    print("Entering search_war")
    war_name = request.json.get('war_name')
    if war_name is None:
        war_name = ''
    print(f"Searching for war: {war_name}")
    wars = loader.search_war(war_name)
    print(f"Found {len(wars)} wars")
    return jsonify({'wars': wars})

@app.route('/search_quest', methods=['POST'])
def search_quest():
    data = request.get_json()
    war_id = str(data.get('war_id'))
    print(f"Searching for quest in war: {war_id}")
    if not war_id:
        return jsonify({'error': 'War ID is required'}), 400
    try:
        war_endpoint = f"{loader.db_loader.BASE_URL}/raw/JP/war/{war_id}"
        war = loader.db_loader._make_request_with_retry(war_endpoint)
        # print(f"War data: {war}")
        if not war:
            return jsonify({'error': 'War not found'}), 404
        quests = war.get('mstQuest', [])
        print(f"Got {len(quests)} quests")
        quest_list = []
        for quest in quests:
            quest = quest['mstQuest']
            quest_id = str(quest['id'])
            print(f"Processing quest: {quest_id}")
            quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/JP/quest/{quest_id}"
            quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
            if quest_data:
                quest_list.append({
                    'id': quest_id,
                    'name': quest_data.get('name', '')
                })
        return jsonify({'quests': quest_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_scripts', methods=['POST'])
def get_scripts():
    quest_id = str(request.json.get('quest_id'))
    if not quest_id:
        return jsonify({'error': 'Quest ID is required'}), 400
    scripts = loader.get_quest_scripts(quest_id)
    return jsonify({'scripts': scripts})

@app.route('/extract_dialogues', methods=['POST'])
def extract_dialogues():
    script_id = str(request.json.get('script_id'))
    if not script_id:
        return jsonify({'error': 'Script ID is required'}), 400
    print(f"Extracting dialogues for script: {script_id}")
    dialogues = loader.extract_dialogues(script_id)
    return jsonify({'dialogues': dialogues})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    dialogues = data.get('dialogues')
    translation_method = data.get('translation_method', 'gpt')
    target_language = data.get('target_language', 'Chinese')
    
    if not dialogues:
        return jsonify({'error': 'Dialogues are required'}), 400
    
    try:
        if translation_method == 'gpt':
            translated = loader.gpt_dialogue_translate(
                dialogues,
                api_base=user_preferences.get('api_base', os.getenv('API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')),
                api_key=user_preferences.get('api_key', os.getenv('API_KEY')),
                target_language=target_language,
                base_model=user_preferences.get('base_model', os.getenv('BASE_MODEL', 'deepseek-v3')),
                api_type=user_preferences.get('api_type', 'openai'),
                auth_type=user_preferences.get('auth_type', 'api_key')
            )
        else:
            translated = asyncio.run(loader.free_translate(dialogues, target_language))
        
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

@app.route('/get_quest_detail', methods=['POST'])
def get_quest_detail():
    quest_id = str(request.json.get('quest_id'))
    if not quest_id:
        return jsonify({'error': 'Quest ID is required'}), 400
    try:
        quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/JP/quest/{quest_id}"
        quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
        return jsonify(quest_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 