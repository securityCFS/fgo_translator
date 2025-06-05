import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import logging
import requests
import os
import openai
from db_loader import AtlasDBLoader
import time
from googletrans import Translator
import asyncio
from tqdm import tqdm
import argparse
import sys
import socket
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def is_port_open(host: str = "127.0.0.1", port: int = 7890, timeout: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except (socket.timeout, socket.error):
            return False

def is_proxy_functional(proxy_url: str = "http://127.0.0.1:7890", test_url: str = "http://httpbin.org/ip", timeout: float = 5.0) -> bool:

    proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }
    try:
        response = requests.get(test_url, proxies=proxies, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Check if clash is running (local proxy enabled)
def is_clash_running():
    return is_port_open() and is_proxy_functional()


if is_clash_running():
    print("Clash proxy detected")
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

class GPTTranslationClient:
    """Client for handling GPT API translations with support for different API types."""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        api_type: str = "openai",  # "openai" or "custom"
        auth_type: str = "bearer",  # "bearer" or "api_key"
        base_model: str = "gpt-4",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the GPT translation client.
        
        Args:
            api_base: Base URL for the API
            api_key: API key or token
            api_type: Type of API ("openai" or "custom")
            auth_type: Type of authentication ("bearer" or "api_key")
            base_model: Model to use for translation
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_base = api_base
        self.api_key = api_key
        self.api_type = api_type.lower()
        self.auth_type = auth_type.lower()
        self.base_model = base_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if self.api_type == "openai":
            self.client = openai.OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=timeout
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API request based on authentication type."""
        if self.auth_type == "bearer":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:  # api_key
            return {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
    
    def _make_openai_request(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Make request using OpenAI client."""
        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def _make_custom_request(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Make request using custom API endpoint."""
        url = f"{self.api_base}/v1/chat/completions"
        data = {
            "model": self.base_model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(
            url,
            headers=self._get_headers(),
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    def translate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        target_language: str = 'Chinese'
    ) -> List[str]:
        """
        Translate text using GPT API.
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            target_language: Target language for translation
        Returns:
            List of translated texts
        """
        for attempt in range(self.max_retries):
            try:
                system_prompt = f"""You are a professional translator for game dialogue.\nTranslate Japanese text from the game \"Fate/Grand Order\" into {target_language}.\nPreserve tone, character speech style, and terminology. Use standard transliterations for names (e.g., キリエライト → Mash Kyrielight/玛修·基列莱特, 藤丸立香 → Ritsuka Fujimaru/藤丸立香).\nOnly return the translated sentence in {target_language}, no extra text or formatting.\n"""
                
                # 构建包含多个对话的提示
                dialogue_prompt = f"""Please translate the following dialogues into {target_language}. You MUST follow these rules:
1. Translate ALL dialogues
2. For each dialogue, write its number followed by a colon (e.g., "1:", "2:", etc.)
3. Write the translation on the next line
4. Keep the translations in the same order as the original dialogues
5. Translate both the speaker's name and their dialogue content
6. For choices, translate both the choice number and content
7. For system messages, keep them as is
8. Use standard transliterations for character names (e.g., ライネス → Lainess/莱尼斯, グレイ → Gray/格雷)

Example format:
1:
[Translated Speaker Name]: [Translation of first dialogue]
2:
[Translated Speaker Name]: [Translation of second dialogue]

Here are the dialogues to translate:

"""
                for i, dialogue in enumerate(messages, 1):
                    dialogue_prompt += f"{i}:\nSpeaker: {dialogue.get('speaker', '')}\nContent: {dialogue.get('content', '')}\n\n"
                
                dialogue_prompt += "\nRemember to translate ALL dialogues and maintain the exact format shown in the example."
                
                formatted_messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": dialogue_prompt
                    }
                ]
                
                if self.api_type == "openai":
                    response = self._make_openai_request(formatted_messages, temperature)
                else:
                    response = self._make_custom_request(formatted_messages, temperature)
                
                # 解析响应，提取每个对话的翻译
                translations = []
                current_translation = []
                current_dialogue_num = None
                
                for line in response.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 检查是否是新的对话标记
                    dialogue_match = re.match(r'^(\d+):', line)
                    if dialogue_match:
                        # 保存之前的翻译
                        if current_translation:
                            translations.append('\n'.join(current_translation))
                            current_translation = []
                        current_dialogue_num = int(dialogue_match.group(1))
                    elif current_dialogue_num is not None:
                        current_translation.append(line)
                
                # 添加最后一个翻译
                if current_translation:
                    translations.append('\n'.join(current_translation))
                
                # 验证翻译数量
                if len(translations) != len(messages):
                    logger.warning(f"Translation count mismatch: got {len(translations)}, expected {len(messages)}")
                    # 如果解析失败，尝试直接按行分割
                    if not translations:
                        lines = [line.strip() for line in response.split('\n') if line.strip()]
                        translations = [line for line in lines if not re.match(r'^\d+:', line)]
                    
                    # 如果仍然没有足够的翻译，补充缺失的翻译
                    while len(translations) < len(messages):
                        translations.append("[Translation Error: Missing translation]")
                    # 如果翻译太多，只取需要的数量
                    translations = translations[:len(messages)]
                
                # 返回所有翻译
                return translations
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise

class DialogueLoader:
    """Class to handle loading and extracting dialogues from FGO quests."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the dialogue loader.
        
        Args:
            cache_dir: Optional directory to cache API responses
        """
        self.db_loader = AtlasDBLoader(cache_dir)
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.translator = Translator()
        
    def _get_text_content(self, url: str) -> str:
        """
        Get text content from a URL.
        
        Args:
            url: URL to fetch text from
            
        Returns:
            Text content as string
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch text content from {url}: {e}")
            return ""
            
    def search_war(self, name: str) -> List[Dict]:
        """
        Search for a war by name.
        First gets basic war info, then fetches both nice and raw data.
        
        Args:
            name: Name of the war to search for
            
        Returns:
            List of matching war data with detailed information
        """
        # If name is a number, search for war by id
        if name.isdigit():
            detailed_matches = []
            try:
                war_id = name
                nice_endpoint = f"{self.db_loader.BASE_URL}/nice/JP/war/{war_id}"
                raw_endpoint = f"{self.db_loader.BASE_URL}/raw/JP/war/{war_id}"
                
                nice_war = self.db_loader._make_request_with_retry(nice_endpoint)
                raw_war = self.db_loader._make_request_with_retry(raw_endpoint)
                
                detailed_matches.append({**nice_war, 'raw': raw_war})
            except Exception as e:
                logger.error(f"Failed to get detailed data for war {war_id}: {e}")
                detailed_matches.append(war)  # Keep basic data if detailed fetch fails
                
            return detailed_matches


        try:
            # First get basic war info
            search_region = self.db_loader._get_search_region(name)
            endpoint = f"{self.db_loader.BASE_URL}/export/{search_region}/basic_war.json"
            wars = self.db_loader._make_request_with_retry(endpoint)
            
            # Search for matches
            name_lower = name.lower()
            if name_lower != '':
                basic_matches = [
                    war for war in wars
                    if name_lower in war['name'].lower() or 
                    name_lower in war.get('longName', '').lower()
                ]
            else:
                basic_matches = wars
            
            if not basic_matches:
                # If no matches found in export data, try API search
                logger.info("No matches found in export data, trying API search...")
                api_endpoint = f"war/search?name={name}"
                basic_matches = self.db_loader._make_request_with_retry(
                    self.db_loader._get_endpoint(api_endpoint)
                )
            
            # Get detailed data for each match
            detailed_matches = []
            for war in basic_matches:
                try:
                    war_id = war['id']
                    # Get both nice and raw war data
                    nice_endpoint = f"{self.db_loader.BASE_URL}/nice/{search_region}/war/{war_id}"
                    raw_endpoint = f"{self.db_loader.BASE_URL}/raw/{search_region}/war/{war_id}"
                    
                    nice_war = self.db_loader._make_request_with_retry(nice_endpoint)
                    raw_war = self.db_loader._make_request_with_retry(raw_endpoint)
                    
                    # Combine nice and raw data
                    detailed_war = {**nice_war, 'raw': raw_war}
                    detailed_matches.append(detailed_war)
                except Exception as e:
                    logger.error(f"Failed to get detailed data for war {war['id']}: {e}")
                    detailed_matches.append(war)  # Keep basic data if detailed fetch fails
            
            return detailed_matches
            
        except Exception as e:
            logger.error(f"Failed to search for war: {e}")
            return []
            
    def search_quest(self, name: str, war_id: Optional[str] = None) -> List[Dict]:
        """
        Search for a quest by name within a war.
        
        Args:
            name: Name of the quest to search for
            war_id: ID of the war to search within
            
        Returns:
            List of matching quest data
        """
        try:
            if war_id is None:
                logger.error("War ID is required for quest search")
                return []
                
            search_region = self.db_loader._get_search_region(name)
            name_lower = name.lower()
            
            # Get both nice and raw war data
            nice_endpoint = f"{self.db_loader.BASE_URL}/nice/{search_region}/war/{war_id}"
            raw_endpoint = f"{self.db_loader.BASE_URL}/raw/{search_region}/war/{war_id}"
            
            nice_war = self.db_loader._make_request_with_retry(nice_endpoint)
            raw_war = self.db_loader._make_request_with_retry(raw_endpoint)
            
            # Extract quest IDs from raw war data
            quests = raw_war.get('mstQuest', [])
            
            # Create mapping of quest names to IDs
            quest_name_to_id = {}
            for quest in quests:
                try:
                    quest_id = quest['mstQuest']['id']
                    quest_endpoint = f"{self.db_loader.BASE_URL}/nice/{search_region}/quest/{quest_id}"
                    quest_data = self.db_loader._make_request_with_retry(quest_endpoint)
                    quest_name = quest_data.get('name', '')
                    if quest_name:
                        quest_name_to_id[quest_name.lower()] = quest_id
                except Exception as e:
                    logger.error(f"Failed to get quest data for ID {quest_id}: {e}")
            
            # Search for quest name in mapping
            matches = []
            for quest_name, quest_id in quest_name_to_id.items():
                if name_lower in quest_name:
                    try:
                        quest_endpoint = f"{self.db_loader.BASE_URL}/nice/{search_region}/quest/{quest_id}"
                        quest_data = self.db_loader._make_request_with_retry(quest_endpoint)
                        matches.append(quest_data)
                    except Exception as e:
                        logger.error(f"Failed to get quest data for ID {quest_id}: {e}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search for quest: {e}")
            return []
            
    def get_quest_scripts(self, quest_id: str) -> List[Dict]:
        """
        Get all phase scripts for a quest.
        
        Args:
            quest_id: ID of the quest
            
        Returns:
            List of phase scripts
        """
        try:
            search_region = 'JP'  # Always use JP region for scripts
            endpoint = f"{self.db_loader.BASE_URL}/nice/{search_region}/quest/{quest_id}"
            quest_data = self.db_loader._make_request_with_retry(endpoint)
            return quest_data.get('phaseScripts', [])
        except Exception as e:
            logger.error(f"Failed to get quest scripts: {e}")
            return []
            
    def extract_dialogues(self, script_id: str) -> List[Dict]:
        """
        Extract dialogues from a script.
        
        Args:
            script_id: ID of the script to extract dialogues from
            
        Returns:
            List of dialogue dictionaries
        """
        try:
            # Get script data
            script_endpoint = f"script/{script_id}"
            script_data = self.db_loader._make_request_with_retry(
                self.db_loader._get_endpoint(script_endpoint)
            )
            
            if not script_data:
                logger.warning(f"No script data found for script ID {script_id}")
                return []
            
            # Get the raw script text from the script URL
            script_url = script_data.get('script', '')
            if not script_url:
                logger.warning(f"No script URL found for script ID {script_id}")
                return []
            
            # Fetch the raw script text
            text_content = self._get_text_content(script_url)
            if not text_content:
                logger.warning(f"Failed to fetch script text from {script_url}")
                return []
            
            text_content = text_content.replace('[%1]', '藤丸立香').replace('[line 3]', "——").replace(
                '[line 6]', "——").replace('[line 18]', "——")
                        
            dialogues = []
            
            # Pattern 1: Regular dialogue with speaker
            # Format: ＠speaker\ncontent\n[k]
            pattern1 = r'＠([^\n]*)\n(.*?)\n\[k\]'
            
            # Pattern 2: Protagonist choice
            # Format: ？num：choice\n
            pattern2 = r'？(\d+)：(.*?)\n'
            
            # Pattern 3: Choice ending marker
            # Format: ？！
            pattern3 = r'？！'
            
            # Find all matches for all patterns
            matches1 = list(re.finditer(pattern1, text_content, re.DOTALL))
            matches2 = list(re.finditer(pattern2, text_content, re.DOTALL))
            matches3 = list(re.finditer(pattern3, text_content, re.DOTALL))
            
            # Combine and sort all matches by their position in the text
            all_matches = []
            
            # Process regular dialogues
            for match in matches1:
                try:
                    speaker = match.group(1).strip() or 'Narrator'
                    content = match.group(2).strip()
                    if content:  # Only add non-empty dialogues
                        all_matches.append({
                            'pos': match.start(),
                            'type': 'dialogue',
                            'speaker': speaker,
                            'content': content.replace('[r]', '\n')
                        })
                except Exception as e:
                    logger.warning(f"Failed to process dialogue match: {e}")
                    continue
            
            # Process protagonist choices
            last_choice_num = None
            for match in matches2:
                try:
                    choice_num = match.group(1)
                    choice_text = match.group(2).strip()
                    if choice_text:  # Only add non-empty choices
                        # Pre-process the choice text
                        processed_text = choice_text.replace('[r]', '\n') #.replace('[%1]', '藤丸立香').replace('[line 3]', "——")
                        all_matches.append({
                            'pos': match.start(),
                            'type': 'choice',
                            'speaker': '藤丸立香',
                            'content': f"Choice {choice_num}: {processed_text}"
                        })
                        last_choice_num = choice_num
                except Exception as e:
                    logger.warning(f"Failed to process choice match: {e}")
                    continue
            
            # Process choice ending markers
            for match in matches3:
                try:
                    all_matches.append({
                        'pos': match.start(),
                        'type': 'choice_ending',
                        'speaker': 'System',
                        'content': f'Choice {last_choice_num} Ending' if last_choice_num else 'Choice Ending'
                    })
                except Exception as e:
                    logger.warning(f"Failed to process choice ending match: {e}")
                    continue
            
            # Sort matches by their position in the text
            all_matches.sort(key=lambda x: x['pos'])
            
            # Convert matches to dialogues
            for match in all_matches:
                dialogues.append({
                    'speaker': match['speaker'],
                    'content': match['content']
                })
            
            return dialogues
            
        except Exception as e:
            logger.error(f"Failed to extract dialogues from script {script_id}: {e}")
            return []
            
    async def _translate_single_dialogue(
        self,
        dialogue: Dict,
        target_language: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Dict:
        """
        Translate a single dialogue using Google Translate.
        Returns a dict with 'translated_content' even for System or error cases.
        """
        # Always return translated_content
        if dialogue['speaker'] == 'System':
            return {
                'speaker': dialogue['speaker'],
                'content': dialogue['content'],
                'translated_content': dialogue['content']
            }

        for attempt in range(max_retries):
            try:
                async with Translator() as translator:
                    result = await translator.translate(
                        dialogue['content'],
                        src='ja',
                        dest=target_language.lower()
                    )
                    return {
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': result.text
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All translation attempts failed for dialogue: {dialogue['content']}")
                    return {
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': dialogue['content']
                    }
        # Fallback
        return {
            'speaker': dialogue['speaker'],
            'content': dialogue['content'],
            'translated_content': dialogue['content']
        }

    async def free_translate(
        self,
        dialogues: List[Dict],
        target_language: str = 'Chinese',
        progress_callback: Callable = None
    ) -> List[Dict]:
        """Translate dialogues using free translation service."""
        total = len(dialogues)
        translated = []
        pbar = tqdm(total=total, desc="Translating (free)", unit="dialogue")
        
        for i, dialogue in enumerate(dialogues, 1):
            try:
                speaker = dialogue.get('speaker', '')
                # 始终显示 tqdm，同时有回调也调用
                if progress_callback:
                    await progress_callback(i, total, speaker)
                pbar.update(1)
                pbar.set_postfix({'speaker': speaker})
                # 原有的翻译逻辑
                translated_dialogue = await self._translate_single_dialogue(
                    dialogue,
                    target_language
                )
                translated.append(translated_dialogue)
            except Exception as e:
                logger.error(f"Failed to translate dialogue {i}: {e}")
                translated.append({
                    'speaker': dialogue.get('speaker', ''),
                    'translated_content': f"[Translation Error: {str(e)}]"
                })
        pbar.close()
        return translated

    def save_dialogues(
        self,
        dialogues: List[Dict],
        war_name: str,
        quest_name: str,
        script_id: str,
        save_dir: Optional[str] = None,
        translate: bool = False,
        translation_method: str = "gpt",  # "gpt" or "free"
        **translate_kwargs
    ):
        """
        Save extracted dialogues to a text file.
        
        Args:
            dialogues: List of dialogue data
            war_name: Name of the war
            quest_name: Name of the quest
            script_id: ID of the script
            save_dir: Optional directory to save dialogues
            translate: Whether to translate dialogues
            translation_method: Method to use for translation ("gpt" or "free")
            translate_kwargs: Arguments for translation
        """
        try:
            # Create a safe filename
            safe_war_name = "".join(c for c in war_name if c.isalnum() or c in (' ', '-', '_'))
            safe_quest_name = "".join(c for c in quest_name if c.isalnum() or c in (' ', '-', '_'))
            filename = f"{safe_war_name}_{safe_quest_name}_script{script_id}.txt"
            
            # Determine save directory
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)
            else:
                filepath = self.cache_dir / filename
            
            # Check if translated file already exists
            if translate and os.path.exists(filepath):
                logger.info(f"Translation already exists for {filename}, skipping...")
                return
            
            # Translate if requested
            if translate:
                if translation_method.lower() == "gpt":
                    dialogues = self.gpt_dialogue_translate(dialogues, **translate_kwargs)
                elif translation_method.lower() == "free":
                    # Run async translation
                    dialogues = asyncio.run(self.free_translate(
                        dialogues,
                        translate_kwargs.get('target_language', 'en'),
                        progress_callback=translate_kwargs.get('progress_callback')
                    ))
                else:
                    logger.error(f"Unknown translation method: {translation_method}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"War: {war_name}\n")
                f.write(f"Quest: {quest_name}\n")
                f.write(f"Script ID: {script_id}\n")
                f.write(f"Translation Method: {translation_method if translate else 'None'}\n")
                f.write("-" * 50 + "\n\n")
                
                for dialogue in dialogues:
                    f.write(f"{dialogue['speaker']}:\n")
                    f.write(f"{dialogue['content']}\n")
                    if 'translated_content' in dialogue:
                        f.write(f"\n{dialogue['translated_content']}\n")
                    f.write("\n")
                    
            logger.info(f"Saved dialogues to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save dialogues: {e}")

    def gpt_dialogue_translate(
        self,
        dialogues: List[Dict],
        api_base: str = None,
        api_key: str = None,
        target_language: str = 'Chinese',
        base_model: str = 'deepseek-v3',
        api_type: str = 'openai',
        auth_type: str = 'api_key',
        progress_callback: Callable = None
    ) -> List[Dict]:
        """Translate dialogues using GPT."""
        if not api_key:
            raise ValueError("API key is required for GPT translation")
        
        total = len(dialogues)
        translated = []
        pbar = tqdm(total=total, desc="Translating (gpt)", unit="dialogue")
        
        # 创建翻译客户端
        client = GPTTranslationClient(
            api_base=api_base,
            api_key=api_key,
            api_type=api_type,
            auth_type=auth_type,
            base_model=base_model,
            timeout=30,
            max_retries=8,
            retry_delay=10
        )
        
        # 批量处理对话，每批最多10个对话
        batch_size = 10
        for i in range(0, total, batch_size):
            batch = dialogues[i:i + batch_size]
            try:
                # 更新进度条
                if progress_callback:
                    progress_callback(i + 1, total, f"Batch {i//batch_size + 1}")
                pbar.update(len(batch))
                pbar.set_postfix({'batch': f"{i//batch_size + 1}/{(total + batch_size - 1)//batch_size}"})
                
                # 批量翻译
                translations = client.translate(batch, 0.7, target_language=target_language)
                
                # 添加翻译结果
                for dialogue, translation in zip(batch, translations):
                    translated.append({
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': translation,
                        'dialogue_number': i + 1  # 添加对话编号以便追踪
                    })
                    
            except Exception as e:
                logger.error(f"Failed to translate batch starting at dialogue {i}: {e}")
                # 为失败的批次添加错误信息
                for dialogue in batch:
                    translated.append({
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': f"[Translation Error: {str(e)}]"
                    })
        
        pbar.close()
        return translated

# Test cases
if __name__ == "__main__":
    def debug_script_translation():
        """Debug function to translate a specific script ID."""
        print("\nFGO Script Translation Debug Tool")
        print("--------------------------------")
        
        # Get script ID
        while True:
            script_id = input("\nEnter script ID to translate (or 0 to exit. default: '0400041440'): ") or '0400041440'
            if script_id == '0':
                return
            break
            
        # Get API key
        api_key = input("\nEnter your API key: ").strip()
        if not api_key:
            print("API key is required!")
            return
            
        # Optional settings
        save_dir = input("\nEnter save directory (default: translated_dialogues_debug): ").strip() or "translated_dialogues_debug"
        translation_method = input("\nEnter translation method (gpt/free, default: gpt): ").strip().lower() or "gpt"
        target_language = input("\nEnter target language (default: Chinese): ").strip() or "Chinese"
        api_base = input("\nEnter API base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1): ").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        base_model = input("\nEnter base model (default: deepseek-v3): ").strip() or "deepseek-v3"
        
        loader = DialogueLoader()
        
        # Extract dialogues
        print(f"\nExtracting dialogues from script ID: {script_id}")
        dialogues = loader.extract_dialogues(script_id)
        
        if not dialogues:
            print("No dialogues found!")
            return
            
        print(f"Found {len(dialogues)} dialogues")
        
        # Save and translate
        loader.save_dialogues(
            dialogues,
            f"Debug Script {script_id}",
            f"Script {script_id}",
            script_id,
            save_dir=save_dir,
            translate=True,
            translation_method=translation_method,
            api_base=api_base,
            api_key=api_key,
            target_language=target_language,
            base_model=base_model,
            api_type="openai",
            auth_type="api_key"
        )
        
        print(f"\nTranslation completed. Check {save_dir} for results.")
    
    # Check if running in debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug_script_translation()
    else:
        # Original test code
        loader = DialogueLoader()
        
        # Test war search
        war_name = "人類裁決法廷 トリニティ・メタトロニオス"
        print(f"\nSearching for war: {war_name}")
        wars = loader.search_war(war_name)
        for war in wars:
            print(f"Found war: {war['name']} (ID: {war['id']})")
            print("Related quests:")
            quests = war.get('raw', {}).get('mstQuest', [])
            for quest in quests:
                quest = quest['mstQuest']
                print(f"- {quest['name']} (ID: {quest['id']})")
                quest_id = quest['id']
                try:
                    quest_endpoint = f"{loader.db_loader.BASE_URL}/nice/JP/quest/{quest_id}"
                    quest_data = loader.db_loader._make_request_with_retry(quest_endpoint)
                    phase_scripts = quest_data.get('phaseScripts', [])
                    script_ids = []
                    for phase in phase_scripts:
                        scripts = phase.get('scripts', [])
                        for script in scripts:
                            script_ids.append(script.get('scriptId', 0))
                        
                    for script_id in script_ids:
                        if script_id == 0:
                            continue
                        dialogues = loader.extract_dialogues(script_id)
                        if dialogues:
                            # Example of saving with GPT translation
                            loader.save_dialogues(
                                dialogues,
                                war['name'],
                                quest_data['name'],
                                script_id,
                                save_dir="translated_dialogues_deepseek",
                                translate=True,
                                translation_method="gpt",
                                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                api_key="sk-xxx",
                                target_language="Chinese",
                                base_model="deepseek-v3",
                                api_type="openai",
                                auth_type="api_key"
                            )

                            # Example of saving with custom API translation
                            # loader.save_dialogues(
                            #     dialogues,
                            #     war['name'],
                            #     quest_data['name'],
                            #     script_id,
                            #     save_dir="custom_translated_dialogues",
                            #     translate=True,
                            #     translation_method="gpt",
                            #     api_base="https://openrouter.ai/api/v1",
                            #     api_key="sk-or-v1-xxx",
                            #     target_language="Chinese",
                            #     base_model="deepseek-v3",
                            #     api_type="custom",
                            #     auth_type="api_key"
                            # )
                            
                            # # Example of saving with free translation
                            # loader.save_dialogues(
                            #     dialogues,
                            #     war['name'],
                            #     quest_data['name'],
                            #     script_id,
                            #     save_dir="free_translated_dialogues",
                            #     translate=True,
                            #     translation_method="free",
                            #     target_language="zh-cn"
                            # )
                except Exception as e:
                    logger.error(f"Failed to process quest {quest_id}: {e}")
