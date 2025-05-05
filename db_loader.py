import requests
import json
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import time
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AtlasDBLoader:
    """Class to handle loading and caching data from Atlas Academy DB."""
    
    BASE_URL = "https://api.atlasacademy.io"
    REGION = "JP"  # Default to JP region, can be changed to NA
    DATA_TYPE = "nice"  # Use nice data by default (human cleaned data)
    MAX_RETRIES = 3  # Maximum number of retries for API calls
    RETRY_DELAY = 1  # Delay between retries in seconds
    
    # Language mapping for search
    LANGUAGE_MAP = {
        'en': 'NA',  # English -> NA server
        'ja': 'JP',  # Japanese -> JP server
        'zh': 'CN',  # Chinese -> CN server
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the DB loader.
        
        Args:
            cache_dir: Optional directory to cache API responses
        """
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._basic_servants = None
        self._exported_data = {}
        self._name_to_id_maps = {}  # Cache for name to ID mappings
        
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: The text to detect language for
            
        Returns:
            Language code (en/ja/zh)
        """
        try:
            lang = detect(text)
            # Map detected language to our supported languages
            if lang in ['en', 'ja', 'zh']:
                return lang
            
            if lang == 'zh-cn':
                return 'zh'
            # Default to English if language not supported
            return 'en'
        except LangDetectException:
            # Default to English if detection fails
            return 'en'
    
    def _get_search_region(self, text: str) -> str:
        """
        Get the appropriate region for searching based on text language.
        
        Args:
            text: The text to search for
            
        Returns:
            Region code (NA/JP/CN)
        """
        lang = self._detect_language(text)
        return self.LANGUAGE_MAP.get(lang, 'JP')  # Default to JP if language not mapped
    
    def _load_name_to_id_map(self, region: str, use_cache: bool = True) -> Dict[str, int]:
        """
        Load or create name to ID mapping for a specific region.
        
        Args:
            region: The region to load mapping for
            use_cache: Whether to use cached data
            
        Returns:
            Dict mapping names to IDs
        """
        if region in self._name_to_id_maps:
            return self._name_to_id_maps[region]
            
        cache_file = self.cache_dir / f"name_to_id_{region}.json"
        
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data:
                self._name_to_id_maps[region] = cached_data
                return cached_data
        
        try:
            # Load basic servant data for the region
            endpoint = f"{self.BASE_URL}/export/{region}/basic_servant.json"
            data = self._make_request_with_retry(endpoint)
            
            # Create name to ID mapping
            name_map = {}
            for servant in data:
                # Add all possible name variations
                name_map[servant['name'].lower()] = servant['id']
                if 'originalName' in servant:
                    name_map[servant['originalName'].lower()] = servant['id']
                if 'overwriteName' in servant:
                    name_map[servant['overwriteName'].lower()] = servant['id']
                if 'nameWithSuffix' in servant:
                    name_map[servant['nameWithSuffix'].lower()] = servant['id']
            
            if use_cache:
                self._save_to_cache(name_map, cache_file)
            
            self._name_to_id_maps[region] = name_map
            return name_map
            
        except Exception as e:
            logger.error(f"Failed to load name to ID map for region {region}: {e}")
            return {}

    def _get_endpoint(self, endpoint: str) -> str:
        """Construct the full API endpoint URL."""
        return f"{self.BASE_URL}/{self.DATA_TYPE}/{self.REGION}/{endpoint}"
    
    def _load_from_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load data from cache if available."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load cache from {cache_file}")
        return None
    
    def _save_to_cache(self, data: Dict, cache_file: Path):
        """Save data to cache."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")

    def _make_request_with_retry(self, url: str, max_retries: int = None) -> Dict:
        """
        Make an API request with retry logic.
        
        Args:
            url: The URL to request
            max_retries: Maximum number of retries (defaults to MAX_RETRIES)
            
        Returns:
            The JSON response data
            
        Raises:
            requests.RequestException: If all retries fail
        """
        max_retries = max_retries or self.MAX_RETRIES
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(self.RETRY_DELAY)
                continue
        
        raise last_error or requests.RequestException("All retry attempts failed")

    def load_exported_data(self, use_cache: bool = True) -> Dict:
        """
        Load all exported data from the export endpoint.
        This contains minimal information about all game entities.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict containing all exported data
        """
        cache_file = self.cache_dir / "exported_data.json"
        
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data:
                self._exported_data = cached_data
                return cached_data
        
        try:
            # Load basic servant data
            endpoint = f"{self.BASE_URL}/export/{self.REGION}/basic_servant.json"
            servant_data = self._make_request_with_retry(endpoint)
            
            # Load other exported data as needed
            # TODO: Add more exported data loading here
            
            data = {
                "servants": servant_data,
                # Add other data types here
            }
            
            if use_cache:
                self._save_to_cache(data, cache_file)
            
            self._exported_data = data
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to load exported data: {e}")
            if use_cache:
                cached_data = self._load_from_cache(cache_file)
                if cached_data:
                    logger.info("Using cached exported data")
                    self._exported_data = cached_data
                    return cached_data
            raise

    def search_servants_by_name(self, name: str, use_cache: bool = True) -> List[Dict]:
        """
        Search for servants by name in any supported language.
        First searches in the appropriate region's data, then gets detailed data from JP.
        
        Args:
            name: Name to search for (in any supported language)
            use_cache: Whether to use cached data
            
        Returns:
            List of detailed servant data dictionaries
        """
        try:
            # Determine search region based on input language
            search_region = self._get_search_region(name)
            logger.info(f"Detected language for '{name}', using {search_region} region for search")
            
            # Load name to ID mapping for the search region
            name_map = self._load_name_to_id_map(search_region, use_cache)
            
            # Search for matches
            name_lower = name.lower()
            matching_ids = [
                servant_id for servant_name, servant_id in name_map.items()
                if name_lower in servant_name
            ]
            
            if not matching_ids:
                logger.info(f"No servants found matching name: {name}")
                return []
            
            # Get detailed data from JP region
            detailed_matches = []
            for servant_id in matching_ids:
                try:
                    detailed_data = self.get_servant(servant_id, use_cache)
                    detailed_matches.append(detailed_data)
                except Exception as e:
                    logger.error(f"Failed to load detailed data for servant ID {servant_id}: {e}")
            
            return detailed_matches
            
        except Exception as e:
            logger.error(f"Failed to search servants: {e}")
            return []

    def get_servant(self, servant_id: int, use_cache: bool = True) -> Dict:
        """
        Get servant data by ID.
        First tries to get from exported data, then falls back to API.
        
        Args:
            servant_id: The ID of the servant
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict containing servant data
        """
        cache_file = self.cache_dir / f"servant_{servant_id}.json"
        
        if use_cache:
            cached_data = self._load_from_cache(cache_file)
            if cached_data:
                return cached_data
        
        try:
            # First try to get from exported data
            if not self._exported_data:
                self.load_exported_data(use_cache)
            
            # Search in exported data
            for servant in self._exported_data.get("servants", []):
                if servant['id'] == servant_id:
                    # Found in exported data, now get detailed data
                    endpoint = f"servant/{servant_id}"
                    response = self._make_request_with_retry(self._get_endpoint(endpoint))
                    if use_cache:
                        self._save_to_cache(response, cache_file)
                    return response
            
            # If not found in exported data, try API directly
            endpoint = f"servant/{servant_id}"
            response = self._make_request_with_retry(self._get_endpoint(endpoint))
            if use_cache:
                self._save_to_cache(response, cache_file)
            return response
            
        except requests.RequestException as e:
            logger.error(f"Failed to get servant data: {e}")
            if use_cache:
                cached_data = self._load_from_cache(cache_file)
                if cached_data:
                    logger.info("Using cached servant data")
                    return cached_data
            raise

# Example usage
if __name__ == "__main__":
    loader = AtlasDBLoader()
    
    # Test different language inputs
    test_names = ["altria caster", "キャスター"]
    
    for name in test_names:
        print(f"\nSearching for: {name}")
        try:
            servants = loader.search_servants_by_name(name)
            for servant in servants:
                print(f"Found servant: {servant['name']} (ID: {servant['id']})")
                print(f"Class: {servant['className']}")
                print(f"Gender: {servant['gender']}")
                print(f"Rarity: {servant['rarity']}")
                for noble_phantasm in servant['noblePhantasms']:
                    print(f"Noble Phantasm: {noble_phantasm['name']}")
                for skill in servant['skills']:
                    print(f"Skill: {skill['name']}")
                
        except Exception as e:
            logger.error(f"Failed to search for {name}: {e}") 
