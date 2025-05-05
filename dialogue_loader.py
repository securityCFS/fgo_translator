import json
from pathlib import Path
from typing import Dict, List, Optional, Union
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

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
        temperature: float = 0.7
    ) -> str:
        """
        Translate text using GPT API.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            
        Returns:
            Translated text
        """
        for attempt in range(self.max_retries):
            try:
                if self.api_type == "openai":
                    return self._make_openai_request(messages, temperature)
                else:
                    return self._make_custom_request(messages, temperature)
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
        try:
            # First get basic war info
            search_region = self.db_loader._get_search_region(name)
            endpoint = f"{self.db_loader.BASE_URL}/export/{search_region}/basic_war.json"
            wars = self.db_loader._make_request_with_retry(endpoint)
            
            # Search for matches
            name_lower = name.lower()
            basic_matches = [
                war for war in wars
                if name_lower in war['name'].lower() or 
                   name_lower in war.get('longName', '').lower()
            ]
            
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
            
    def search_quest(self, name: str, war_id: Optional[int] = None) -> List[Dict]:
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
            
    def get_quest_scripts(self, quest_id: int) -> List[Dict]:
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
            
    def extract_dialogues(self, script_id: int) -> List[Dict]:
        """
        Extract dialogues from a script by fetching and parsing the text file.
        
        Args:
            script_id: ID of the script
            
        Returns:
            List of extracted dialogues
        """
        try:
            # Get script data to get the text file URL
            endpoint = f"{self.db_loader.BASE_URL}/nice/JP/script/{script_id}"
            script_data = self.db_loader._make_request_with_retry(endpoint)
            
            # Get the text file URL
            text_url = script_data.get('script', '')
            if not text_url:
                logger.error(f"No text URL found for script {script_id}")
                return []
                
            # Fetch the text file content
            text_content = self._get_text_content(text_url)
            if not text_content:
                logger.error(f"Failed to fetch text content from {text_url}")
                return []
            
            dialogues = []
            current_speaker = None
            current_text = []
            current_choices = []
            is_protagonist_choice = False
            
            # Process the text content line by line
            for line in text_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Skip lines that start with [ and don't contain @
                if line.startswith('[') and '@' not in line:
                    continue
                    
                # Check for speaker line
                if line.startswith('＠'):
                    # Save previous dialogue if exists
                    if current_speaker and current_text:
                        dialogue_content = '\n'.join(current_text)
                        if current_choices:
                            dialogue_content += '\n\nChoices:\n' + '\n'.join(current_choices)
                        dialogues.append({
                            'speaker': current_speaker,
                            'content': dialogue_content
                        })
                        current_text = []
                        current_choices = []
                    
                    # Extract new speaker
                    # if line is only @, it means the speaker is narrator
                    if line.strip() == '＠':
                        current_speaker = 'Narrator'
                    else:
                        current_speaker = line[1:].strip()
                    is_protagonist_choice = False
                    continue
                    
                # Check for end of dialogue
                if line == '[k]':
                    if current_speaker and current_text:
                        dialogue_content = '\n'.join(current_text)
                        if current_choices:
                            dialogue_content += '\n\nChoices:\n' + '\n'.join(current_choices)
                        dialogues.append({
                            'speaker': current_speaker,
                            'content': dialogue_content
                        })
                        current_speaker = None
                        current_text = []
                        current_choices = []
                    continue
                    
                # Process dialogue text
                if current_speaker:
                    # Replace [r] with newline
                    line = line.replace('[r]', '\n')
                    
                    # Check for protagonist choice pattern (？num：text)
                    if '？' in line and '：' in line:
                        choice_num = line.split('：')[0].strip()
                        choice_text = line.split('：')[1].strip()
                        current_choices.append(f"{choice_num}: {choice_text}")
                        is_protagonist_choice = True
                        continue
                    
                    # Check for protagonist dialogue (？！)
                    if line.startswith('？！'):
                        current_speaker = 'Fujimaru Ritsuka'
                        line = line[2:].strip()  # Remove the ？！ prefix
                    
                    # Remove [] blocks if they are on their own line
                    if line.startswith('[') and line.endswith(']'):
                        continue
                        
                    # Remove [] blocks that are separated by newlines
                    if '[' in line and ']' in line:
                        # Split the line by newlines
                        parts = line.split('\n')
                        filtered_parts = []
                        for part in parts:
                            part = part.strip()
                            # Keep the part if it's not a [] block or if it's part of a larger text
                            if not (part.startswith('[') and part.endswith(']')):
                                filtered_parts.append(part)
                        line = '\n'.join(filtered_parts)
                    
                    current_text.append(line.strip())
            
            # Save any remaining dialogue
            if current_speaker and current_text:
                dialogue_content = '\n'.join(current_text)
                if current_choices:
                    dialogue_content += '\n\nChoices:\n' + '\n'.join(current_choices)
                dialogues.append({
                    'speaker': current_speaker,
                    'content': dialogue_content
                })
            
            return dialogues
            
        except Exception as e:
            logger.error(f"Failed to extract dialogues from script {script_id}: {e}")
            return []
            
    def gpt_dialogue_translate(
        self,
        dialogues: List[Dict],
        api_base: str,
        api_key: str,
        target_language: str,
        base_model: str = "gpt-4",
        max_context_size: int = 8,
        temperature: float = 0.7,
        max_retries: int = 8,
        timeout: int = 30,
        retry_delay: int = 10,
        api_type: str = "openai",
        auth_type: str = "bearer"
    ) -> List[Dict]:
        """
        Translate dialogues using GPT API, one dialogue at a time with context awareness.
        
        Args:
            dialogues: List of dialogue data
            api_base: OpenAI API base URL
            api_key: OpenAI API key
            target_language: Target language for translation
            base_model: GPT model to use
            max_context_size: Maximum number of previous dialogues to keep in context
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            api_type: Type of API ("openai" or "custom")
            auth_type: Type of authentication ("bearer" or "api_key")
            
        Returns:
            List of translated dialogues
        """
        try:
            # Initialize translation client
            client = GPTTranslationClient(
                api_base=api_base,
                api_key=api_key,
                api_type=api_type,
                auth_type=auth_type,
                base_model=base_model,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            translated_dialogues = []
            
            # Initialize context with system prompt
            context = [
                {
                    "role": "system",
                    "content": f"""You are a professional translator specializing in game dialogue translation.
The text you are translating is from game "Fate/Grand Order".
Your task is to translate Japanese game dialogue to {target_language}.

Follow these rules:
1. Translate one dialogue at a time, maintaining natural conversation flow
2. Keep the speaker's personality and speech style consistent
3. Make the translation sound natural and conversational, not like machine translation
4. Preserve any special formatting or emphasis
5. Keep proper names and technical terms consistent
6. Handle cultural references appropriately
7. Consider the context of previous dialogues when translating
8. If the dialogue is part of a conversation, maintain the flow and tone
9. Keep character names consistent throughout the translation
10. If a character name appears in the dialogue, maintain its original form

Example input:
カドック:
ああ。
依頼は[#煉獄:れんごく]から地獄へと向かう二人の護衛。

Example output (Chinese):
卡多克：
啊。
任务是要护送两个人从炼狱前往地狱。

Remember to:
- Keep translations natural and flowing
- Maintain character voice consistency
- Consider conversation context
- Make it sound like real human dialogue
- Keep character names consistent"""
                }
            ]
            
            # First, collect all unique speaker names
            speaker_names = set()
            for dialogue in dialogues:
                if dialogue['speaker'] != 'Narrator':
                    speaker_names.add(dialogue['speaker'])
            
            # Add speaker names to context if any
            if speaker_names:
                context.append({
                    "role": "system",
                    "content": f"Important character names to keep consistent: {', '.join(speaker_names)}"
                })
            
            # Create progress bar
            pbar = tqdm(
                total=len(dialogues),
                desc="Translating dialogues",
                unit="dialogue"
            )
            
            for dialogue in dialogues:
                # Prepare the current dialogue
                current_dialogue = f"{dialogue['speaker']}:\n{dialogue['content']}"
                
                # Add current dialogue to context
                context.append({
                    "role": "user",
                    "content": f"Please translate this dialogue, with translated speaker name '{dialogue['speaker']}' at beginning:\n{current_dialogue}"
                })
                
                try:
                    # Get translation
                    translated_text = client.translate(context, temperature)
                    
                    # Keep the entire translated text including the speaker line
                    translated_content = translated_text.strip()
                    
                    # Add to translated dialogues
                    translated_dialogues.append({
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': translated_content
                    })
                    
                    # Add response to context
                    context.append({
                        "role": "assistant",
                        "content": translated_text
                    })
                    
                    # Maintain context size
                    while len(context) > max_context_size + 2:  # +2 for system prompts
                        context.pop(2)  # Remove oldest user message
                        context.pop(2)  # Remove corresponding assistant response
                        
                except Exception as e:
                    logger.error(f"Failed to translate dialogue: {e}")
                    translated_dialogues.append({
                        'speaker': dialogue['speaker'],
                        'content': dialogue['content'],
                        'translated_content': dialogue['content']
                    })
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'speaker': dialogue['speaker'],
                    'status': 'success' if 'translated_content' in translated_dialogues[-1] else 'failed'
                })
            
            # Close progress bar
            pbar.close()
            
            return translated_dialogues
            
        except Exception as e:
            logger.error(f"Failed to translate dialogues: {e}")
            return dialogues
        
    async def _translate_single_dialogue(
        self,
        dialogue: Dict,
        target_language: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Dict:
        """
        Translate a single dialogue using Google Translate.
        
        Args:
            dialogue: Dialogue data to translate
            target_language: Target language for translation
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Translated dialogue data
        """
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
        
        return {
            'speaker': dialogue['speaker'],
            'content': dialogue['content'],
            'translated_content': dialogue['content']
        }

    async def free_translate(
        self,
        dialogues: List[Dict],
        target_language: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> List[Dict]:
        """
        Translate dialogues using free translation service (Google Translate).
        
        Args:
            dialogues: List of dialogue data
            target_language: Target language for translation
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of translated dialogues
        """
        try:
            # Create tasks for all dialogues
            tasks = [
                self._translate_single_dialogue(
                    dialogue,
                    target_language,
                    max_retries,
                    retry_delay
                )
                for dialogue in dialogues
            ]
            
            # Create progress bar
            pbar = tqdm(
                total=len(tasks),
                desc="Translating dialogues",
                unit="dialogue"
            )
            
            # Wait for all translations to complete
            translated_dialogues = []
            for task in asyncio.as_completed(tasks):
                result = await task
                translated_dialogues.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    'speaker': result['speaker'],
                    'status': 'success' if result['translated_content'] != result['content'] else 'failed'
                })
            
            # Close progress bar
            pbar.close()
            
            return translated_dialogues
            
        except Exception as e:
            logger.error(f"Failed to translate dialogues: {e}")
            return dialogues

    def save_dialogues(
        self,
        dialogues: List[Dict],
        war_name: str,
        quest_name: str,
        script_id: int,
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
                        max_retries=translate_kwargs.get('max_retries', 3),
                        retry_delay=translate_kwargs.get('retry_delay', 2)
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
