import os
from pathlib import Path
from dialogue_loader import DialogueLoader
import logging
import sys
import sqlite3
import json
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language-specific messages
MESSAGES = {
    "en": {
        "welcome": "FGO Dialogue Translation Tool",
        "intro": "This tool helps you translate FGO dialogues from Japanese to other languages.",
        "war_hint": "You can find war names (关卡配置) at: https://apps.atlasacademy.io/db/JP/wars",
        "start": "Let's get started!",
        "select_language": "Select target language (1-3):",
        "select_method": "Select translation method (1-2):",
        "api_config": "API Configuration:",
        "api_base": "Enter API base URL (default: https://api.openai.com/v1):",
        "api_key": "Enter your API key:",
        "api_key_required": "API key is required for GPT translation!",
        "enter_war": "Please enter the war name (关卡配置) you want to translate.",
        "searching_war": "Searching for war: {}",
        "no_wars": "No wars found with that name!",
        "found_wars": "Found wars:",
        "select_war": "Select a war (enter number):",
        "invalid_choice": "Invalid choice. Please try again.",
        "enter_valid": "Please enter a valid number.",
        "processing_quest": "Processing quest: {} (ID: {})",
        "getting_scripts": "Getting scripts for quest: {}",
        "no_scripts": "No scripts found for this quest!",
        "processing_scripts": "Processing {} scripts...",
        "processing_script": "Processing script ID: {}",
        "no_dialogues": "No dialogues found in script {}",
        "found_dialogues": "Found {} dialogues",
        "success_script": "Successfully processed script {}",
        "failed_script": "Failed to process script {}: {}",
        "completed": "Translation completed! Check the '{}' directory for results.",
        "cancelled": "Translation cancelled by user.",
        "error": "An error occurred: {}",
        "export_path": "Enter export directory path (press Enter for default 'translations_{}'):",
        "translate_all": "Do you want to translate all quests in this war? (y/n):",
        "enter_quest": "Please enter the quest name to translate:",
        "searching_quest": "Searching for quest: {}",
        "no_quests": "No quests found with that name!",
        "found_quests": "Found quests:",
        "select_quest": "Select a quest (enter number):",
        "main_menu": "\nMain Menu:",
        "menu_options": "1. Translate another war\n2. Change language\n3. Change translation method\n4. Exit",
        "menu_choice": "Enter your choice (1-4):",
        "invalid_menu": "Invalid menu choice. Please try again.",
        "goodbye": "Thank you for using FGO Dialogue Translation Tool. Goodbye!",
        "api_type": "Select API type (1-2):\n1. OpenAI\n2. Custom",
        "base_model": "Enter base model name (default: gpt-4):",
        "auth_type": "Select authentication type (1-2):\n1. Bearer\n2. None",
        "invalid_api_type": "Invalid API type. Please try again.",
        "invalid_auth_type": "Invalid authentication type. Please try again.",
        "use_saved_config": "Use saved API configuration? (y/n):",
        "saved_config": "Saved configuration:\nAPI Base: {}\nAPI Type: {}\nBase Model: {}\nAuth Type: {}",
        "update_config": "Do you want to update the configuration? (y/n):"
    },
    "zh-cn": {
        "welcome": "FGO 对话翻译工具",
        "intro": "这个工具可以帮助你将 FGO 的日语对话翻译成其他语言。",
        "war_hint": "你可以在以下网址找到关卡配置：https://apps.atlasacademy.io/db/JP/wars",
        "start": "让我们开始吧！",
        "select_language": "选择目标语言 (1-3):",
        "select_method": "选择翻译方法 (1-2):",
        "api_config": "API 配置:",
        "api_base": "输入 API 基础 URL (默认: https://api.openai.com/v1):",
        "api_key": "输入你的 API 密钥:",
        "api_key_required": "GPT 翻译需要 API 密钥！",
        "enter_war": "请输入你想要翻译的关卡配置名称。",
        "searching_war": "正在搜索关卡: {}",
        "no_wars": "未找到该名称的关卡！",
        "found_wars": "找到以下关卡:",
        "select_war": "选择一个关卡 (输入编号):",
        "invalid_choice": "无效的选择，请重试。",
        "enter_valid": "请输入有效的数字。",
        "processing_quest": "正在处理任务: {} (ID: {})",
        "getting_scripts": "正在获取任务脚本: {}",
        "no_scripts": "未找到该任务的脚本！",
        "processing_scripts": "正在处理 {} 个脚本...",
        "processing_script": "正在处理脚本 ID: {}",
        "no_dialogues": "在脚本 {} 中未找到对话",
        "found_dialogues": "找到 {} 个对话",
        "success_script": "成功处理脚本 {}",
        "failed_script": "处理脚本 {} 失败: {}",
        "completed": "翻译完成！请查看 '{}' 目录中的结果。",
        "cancelled": "用户取消了翻译。",
        "error": "发生错误: {}",
        "export_path": "输入导出目录路径 (按回车使用默认路径 'translations_{}'):",
        "translate_all": "是否要翻译此关卡中的所有任务？(y/n):",
        "enter_quest": "请输入要翻译的任务名称:",
        "searching_quest": "正在搜索任务: {}",
        "no_quests": "未找到该名称的任务！",
        "found_quests": "找到以下任务:",
        "select_quest": "选择一个任务 (输入编号):",
        "main_menu": "\n主菜单:",
        "menu_options": "1. 翻译另一个关卡\n2. 更改语言\n3. 更改翻译方法\n4. 退出",
        "menu_choice": "请输入你的选择 (1-4):",
        "invalid_menu": "无效的菜单选择，请重试。",
        "goodbye": "感谢使用 FGO 对话翻译工具。再见！",
        "api_type": "选择 API 类型 (1-2):\n1. OpenAI\n2. 自定义",
        "base_model": "输入基础模型名称 (默认: gpt-4):",
        "auth_type": "选择认证类型 (1-2):\n1. Bearer\n2. 无",
        "invalid_api_type": "无效的 API 类型，请重试。",
        "invalid_auth_type": "无效的认证类型，请重试。",
        "use_saved_config": "使用已保存的 API 配置？(y/n):",
        "saved_config": "已保存的配置:\nAPI 基础 URL: {}\nAPI 类型: {}\n基础模型: {}\n认证类型: {}",
        "update_config": "是否要更新配置？(y/n):"
    },
    "zh-tw": {
        "welcome": "FGO 對話翻譯工具",
        "intro": "這個工具可以幫助你將 FGO 的日語對話翻譯成其他語言。",
        "war_hint": "你可以在以下網址找到關卡配置：https://apps.atlasacademy.io/db/JP/wars",
        "start": "讓我們開始吧！",
        "select_language": "選擇目標語言 (1-3):",
        "select_method": "選擇翻譯方法 (1-2):",
        "api_config": "API 配置:",
        "api_base": "輸入 API 基礎 URL (預設: https://api.openai.com/v1):",
        "api_key": "輸入你的 API 金鑰:",
        "api_key_required": "GPT 翻譯需要 API 金鑰！",
        "enter_war": "請輸入你想要翻譯的關卡配置名稱。",
        "searching_war": "正在搜尋關卡: {}",
        "no_wars": "未找到該名稱的關卡！",
        "found_wars": "找到以下關卡:",
        "select_war": "選擇一個關卡 (輸入編號):",
        "invalid_choice": "無效的選擇，請重試。",
        "enter_valid": "請輸入有效的數字。",
        "processing_quest": "正在處理任務: {} (ID: {})",
        "getting_scripts": "正在獲取任務腳本: {}",
        "no_scripts": "未找到該任務的腳本！",
        "processing_scripts": "正在處理 {} 個腳本...",
        "processing_script": "正在處理腳本 ID: {}",
        "no_dialogues": "在腳本 {} 中未找到對話",
        "found_dialogues": "找到 {} 個對話",
        "success_script": "成功處理腳本 {}",
        "failed_script": "處理腳本 {} 失敗: {}",
        "completed": "翻譯完成！請查看 '{}' 目錄中的結果。",
        "cancelled": "用戶取消了翻譯。",
        "error": "發生錯誤: {}",
        "export_path": "輸入導出目錄路徑 (按回車使用預設路徑 'translations_{}'):",
        "translate_all": "是否要翻譯此關卡中的所有任務？(y/n):",
        "enter_quest": "請輸入要翻譯的任務名稱:",
        "searching_quest": "正在搜尋任務: {}",
        "no_quests": "未找到該名稱的任務！",
        "found_quests": "找到以下任務:",
        "select_quest": "選擇一個任務 (輸入編號):",
        "main_menu": "\n主選單:",
        "menu_options": "1. 翻譯另一個關卡\n2. 更改語言\n3. 更改翻譯方法\n4. 退出",
        "menu_choice": "請輸入你的選擇 (1-4):",
        "invalid_menu": "無效的選單選擇，請重試。",
        "goodbye": "感謝使用 FGO 對話翻譯工具。再見！",
        "api_type": "選擇 API 類型 (1-2):\n1. OpenAI\n2. 自定義",
        "base_model": "輸入基礎模型名稱 (預設: gpt-4):",
        "auth_type": "選擇認證類型 (1-2):\n1. Bearer\n2. 無",
        "invalid_api_type": "無效的 API 類型，請重試。",
        "invalid_auth_type": "無效的認證類型，請重試。",
        "use_saved_config": "使用已保存的 API 配置？(y/n):",
        "saved_config": "已保存的配置:\nAPI 基礎 URL: {}\nAPI 類型: {}\n基礎模型: {}\n認證類型: {}",
        "update_config": "是否要更新配置？(y/n):"
    }
}

class UserPreferences:
    """Class to handle user preferences storage and retrieval."""
    
    def __init__(self, db_path: str = "user_preferences.db"):
        """Initialize the preferences database."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
    
    def get_preference(self, key: str) -> Optional[str]:
        """Get a preference value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def set_preference(self, key: str, value: str):
        """Set a preference value."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def get_all_preferences(self) -> Dict[str, str]:
        """Get all preferences."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key, value FROM preferences")
            return dict(cursor.fetchall())

def print_welcome(messages: Dict[str, str]):
    """Print welcome message and instructions."""
    print("\n" + "="*50)
    print(messages["welcome"])
    print("="*50)
    print(f"\n{messages['intro']}")
    print(messages["war_hint"])
    print(f"\n{messages['start']}")

def get_language_choice(messages: Dict[str, str]) -> Tuple[str, str]:
    """Get user's language choice."""
    print("\nAvailable target languages:")
    print("1. English (en)")
    print("2. Simplified Chinese (zh-cn)")
    print("3. Traditional Chinese (zh-tw)")
    
    while True:
        choice = input(f"\n{messages['select_language']} ").strip()
        if choice == "1":
            return "en", "English"
        elif choice == "2":
            return "zh-cn", "Simplified Chinese"
        elif choice == "3":
            return "zh-tw", "Traditional Chinese"
        else:
            print(messages["invalid_choice"])

def get_translation_method(messages: Dict[str, str]) -> str:
    """Get user's preferred translation method."""
    print("\nTranslation methods:")
    print("1. GPT (requires API key)")
    print("2. Google Translate (free)")
    
    while True:
        choice = input(f"\n{messages['select_method']} ").strip()
        if choice == "1":
            return "gpt"
        elif choice == "2":
            return "free"
        else:
            print(messages["invalid_choice"])

def check_saved_config(prefs: UserPreferences) -> Tuple[bool, Optional[Tuple[str, str, str, str, str]]]:
    """Check if there are saved API configurations."""
    api_base = prefs.get_preference("api_base")
    api_key = prefs.get_preference("api_key")
    api_type = prefs.get_preference("api_type")
    base_model = prefs.get_preference("base_model")
    auth_type = prefs.get_preference("auth_type")
    
    if all([api_base, api_key, api_type, base_model, auth_type]):
        return True, (api_base, api_key, api_type, base_model, auth_type)
    return False, None

def get_api_config(messages: Dict[str, str], prefs: UserPreferences) -> Tuple[str, str, str, str, str]:
    """Get API configuration from user."""
    print(f"\n{messages['api_config']}")
    
    # Check for saved configuration
    has_saved_config, saved_config = check_saved_config(prefs)
    if has_saved_config:
        api_base, api_key, api_type, base_model, auth_type = saved_config
        print(messages["saved_config"].format(api_base, api_type, base_model, auth_type))
        
        while True:
            choice = input(f"\n{messages['use_saved_config']} ").strip().lower()
            if choice in ['y', 'n']:
                break
            print(messages["invalid_choice"])
        
        if choice == 'y':
            return api_base, api_key, api_type, base_model, auth_type
    
    # Get new configuration
    # Try to get saved API base
    saved_api_base = prefs.get_preference("api_base")
    api_base = input(f"{messages['api_base']} ").strip() or saved_api_base or "https://api.openai.com/v1"
    
    # Try to get saved API key
    saved_api_key = prefs.get_preference("api_key")
    api_key = input(f"{messages['api_key']} ").strip() or saved_api_key
    
    if not api_key:
        print(messages["api_key_required"])
        sys.exit(1)
    
    # Get API type
    print(f"\n{messages['api_type']}")
    while True:
        try:
            api_type_choice = int(input().strip())
            if api_type_choice in [1, 2]:
                api_type = "openai" if api_type_choice == 1 else "custom"
                break
            print(messages["invalid_api_type"])
        except ValueError:
            print(messages["invalid_api_type"])
    
    # Get base model
    saved_base_model = prefs.get_preference("base_model")
    base_model = input(f"\n{messages['base_model']} ").strip() or saved_base_model or "gpt-4"
    
    # Get auth type
    print(f"\n{messages['auth_type']}")
    while True:
        try:
            auth_type_choice = int(input().strip())
            if auth_type_choice in [1, 2]:
                auth_type = "bearer" if auth_type_choice == 1 else "none"
                break
            print(messages["invalid_auth_type"])
        except ValueError:
            print(messages["invalid_auth_type"])
    
    # Save preferences
    prefs.set_preference("api_base", api_base)
    prefs.set_preference("api_key", api_key)
    prefs.set_preference("api_type", api_type)
    prefs.set_preference("base_model", base_model)
    prefs.set_preference("auth_type", auth_type)
    
    return api_base, api_key, api_type, base_model, auth_type

def get_export_path(messages: Dict[str, str], target_lang: str) -> str:
    """Get export directory path from user."""
    default_path = f"translations_{target_lang}"
    path = input(messages["export_path"].format(target_lang)).strip()
    return path or default_path

def is_id(input_str: str) -> bool:
    """Check if the input string is a numeric ID."""
    return input_str.isdigit()

def handle_war_search(loader: DialogueLoader, messages: Dict[str, str]) -> Tuple[Dict, str]:
    """Handle war search with ID detection."""
    while True:
        war_input = input(f"\n{messages['enter_war']} ").strip()
        
        if is_id(war_input):
            print("\nDetected numeric input. This might be a war ID.")
            print("1. Use as ID directly")
            print("2. Search by this number as name")
            choice = input("Choose an option (1/2): ").strip()
            
            if choice == "1":
                # Use as ID directly
                return {"id": int(war_input), "name": f"War {war_input}"}, war_input
        
        # Search for war
        print(messages["searching_war"].format(war_input))
        wars = loader.search_war(war_input)
        
        if not wars:
            print(messages["no_wars"])
            continue
        
        # Display found wars
        print(f"\n{messages['found_wars']}")
        for i, war in enumerate(wars, 1):
            print(f"{i}. {war['name']} (ID: {war['id']})")
        
        # Get war selection
        while True:
            try:
                choice = int(input(f"\n{messages['select_war']} ").strip())
                if 1 <= choice <= len(wars):
                    return wars[choice-1], war_input
                else:
                    print(messages["invalid_choice"])
            except ValueError:
                print(messages["enter_valid"])

def handle_quest_search(loader: DialogueLoader, war_id: int, messages: Dict[str, str]) -> Dict:
    """Handle quest search with ID detection."""
    while True:
        quest_input = input(f"\n{messages['enter_quest']} ").strip()
        
        if is_id(quest_input):
            print("\nDetected numeric input. This might be a quest ID.")
            print("1. Use as ID directly")
            print("2. Search by this number as name")
            choice = input("Choose an option (1/2): ").strip()
            
            if choice == "1":
                # Use as ID directly
                return {"id": int(quest_input), "name": f"Quest {quest_input}"}
        
        # Search for quest
        print(messages["searching_quest"].format(quest_input))
        quests = loader.search_quest(quest_input, war_id)
        
        if not quests:
            print(messages["no_quests"])
            continue
        
        # Display found quests
        print(f"\n{messages['found_quests']}")
        for i, quest in enumerate(quests, 1):
            print(f"{i}. {quest['name']} (ID: {quest['id']})")
        
        # Get quest selection
        while True:
            try:
                choice = int(input(f"\n{messages['select_quest']} ").strip())
                if 1 <= choice <= len(quests):
                    return quests[choice-1]
                else:
                    print(messages["invalid_choice"])
            except ValueError:
                print(messages["enter_valid"])

def process_quests(
    loader: DialogueLoader,
    selected_war: Dict,
    messages: Dict[str, str],
    save_dir: str,
    translation_method: str,
    api_base: Optional[str],
    api_key: Optional[str],
    api_type: Optional[str],
    base_model: Optional[str],
    auth_type: Optional[str],
    target_lang: str
) -> None:
    """Process quests in a war."""
    # Ask if user wants to translate all quests
    while True:
        choice = input(f"\n{messages['translate_all']} ").strip().lower()
        if choice in ['y', 'n']:
            break
        print(messages["invalid_choice"])
    
    if choice == 'y':
        # Process all quests
        quests = selected_war.get('raw', {}).get('mstQuest', [])
        for quest_data in quests:
            process_single_quest(
                loader, quest_data['mstQuest'], selected_war['name'],
                messages, save_dir, translation_method,
                api_base, api_key, api_type, base_model, auth_type, target_lang
            )
    else:
        # Let user search for specific quest
        selected_quest = handle_quest_search(loader, selected_war['id'], messages)
        process_single_quest(
            loader, selected_quest, selected_war['name'],
            messages, save_dir, translation_method,
            api_base, api_key, api_type, base_model, auth_type, target_lang
        )

def process_single_quest(
    loader: DialogueLoader,
    quest: Dict,
    war_name: str,
    messages: Dict[str, str],
    save_dir: str,
    translation_method: str,
    api_base: Optional[str],
    api_key: Optional[str],
    api_type: Optional[str],
    base_model: Optional[str],
    auth_type: Optional[str],
    target_lang: str
) -> None:
    """Process a single quest and its scripts."""
    quest_id = quest['id']
    quest_name = quest['name']
    
    print(messages["processing_quest"].format(quest_name, quest_id))
    
    try:
        # Get quest data
        quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/JP/quest/{quest_id}"
        quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
        
        # Get all scripts for the quest
        phase_scripts = quest_data.get('phaseScripts', [])
        script_ids = []
        for phase in phase_scripts:
            scripts = phase.get('scripts', [])
            for script in scripts:
                script_id = script.get('scriptId', 0)
                if script_id != 0:
                    script_ids.append(script_id)
        
        if not script_ids:
            print(messages["no_scripts"])
            return
        
        print(messages["processing_scripts"].format(len(script_ids)))
        
        # Process each script
        for script_id in script_ids:
            print(messages["processing_script"].format(script_id))
            
            # Extract dialogues
            dialogues = loader.extract_dialogues(script_id)
            if not dialogues:
                print(messages["no_dialogues"].format(script_id))
                continue
            
            print(messages["found_dialogues"].format(len(dialogues)))
            
            # Save and translate
            try:
                loader.save_dialogues(
                    dialogues,
                    war_name,
                    quest_name,
                    script_id,
                    save_dir=save_dir,
                    translate=True,
                    translation_method=translation_method,
                    api_base=api_base,
                    api_key=api_key,
                    target_language=target_lang,
                    base_model=base_model,
                    api_type=api_type,
                    auth_type=auth_type
                )
                print(messages["success_script"].format(script_id))
            except Exception as e:
                logger.error(messages["failed_script"].format(script_id, str(e)))
                
    except Exception as e:
        logger.error(messages["failed_script"].format(quest_id, str(e)))

def main_menu(
    loader: DialogueLoader,
    prefs: UserPreferences,
    messages: Dict[str, str],
    target_lang: str,
    translation_method: str,
    api_base: Optional[str],
    api_key: Optional[str],
    api_type: Optional[str],
    base_model: Optional[str],
    auth_type: Optional[str]
) -> bool:
    """Display main menu and handle user choice."""
    print(messages["main_menu"])
    print(messages["menu_options"])
    
    while True:
        try:
            choice = int(input(f"\n{messages['menu_choice']} ").strip())
            if 1 <= choice <= 4:
                break
            print(messages["invalid_menu"])
        except ValueError:
            print(messages["invalid_menu"])
    
    if choice == 1:
        # Translate another war
        return True
    elif choice == 2:
        # Change language
        target_lang, _ = get_language_choice(MESSAGES["en"])
        prefs.set_preference("language", target_lang)
        return True
    elif choice == 3:
        # Change translation method
        translation_method = get_translation_method(messages)
        if translation_method == "gpt":
            api_base, api_key, api_type, base_model, auth_type = get_api_config(messages, prefs)
        return True
    else:
        # Exit
        print(messages["goodbye"])
        return False

def main():
    """Main function to run the translation tool."""
    # Initialize preferences
    prefs = UserPreferences()
    
    while True:
        # Get saved language preference or ask user
        saved_lang = prefs.get_preference("language")
        if saved_lang and saved_lang in MESSAGES:
            target_lang = saved_lang
            lang_name = "English" if target_lang == "en" else "简体中文" if target_lang == "zh-cn" else "繁體中文"
        else:
            target_lang, lang_name = get_language_choice(MESSAGES["en"])
            prefs.set_preference("language", target_lang)
        
        messages = MESSAGES[target_lang]
        print_welcome(messages)
        
        # Initialize dialogue loader
        loader = DialogueLoader(cache_dir=Path("cache"))
        
        print(f"\n{messages['start']}")
        
        # Get translation method
        translation_method = get_translation_method(messages)
        print(f"\n{messages['start']}")
        
        # Get API configuration if using GPT
        api_base = None
        api_key = None
        api_type = None
        base_model = None
        auth_type = None
        if translation_method == "gpt":
            api_base, api_key, api_type, base_model, auth_type = get_api_config(messages, prefs)
        
        # Get war information
        print(f"\n{messages['enter_war']}")
        print(messages["war_hint"])
        selected_war, _ = handle_war_search(loader, messages)
        
        # Get export path
        save_dir = get_export_path(messages, target_lang)
        os.makedirs(save_dir, exist_ok=True)
        
        # Process quests
        process_quests(
            loader, selected_war, messages, save_dir,
            translation_method, api_base, api_key, api_type, base_model, auth_type, target_lang
        )
        
        print(messages["completed"].format(save_dir))
        
        # Show main menu
        if not main_menu(loader, prefs, messages, target_lang, translation_method, 
                        api_base, api_key, api_type, base_model, auth_type):
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + MESSAGES["en"]["cancelled"])
    except Exception as e:
        logger.error(MESSAGES["en"]["error"].format(str(e)))
        sys.exit(1) 