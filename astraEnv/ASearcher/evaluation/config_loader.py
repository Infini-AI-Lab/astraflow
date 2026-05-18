import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    
    def __init__(self, config_path: str = "eval_config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"config {self.config_path} do not exists")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            if self.config is None:
                self.config = {}
                
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error when parsing {self.config_path}: {e}")
    
    def set_env_vars(self) -> None:
        if self.config is None:
            self.load_config()
            
        api_keys = self.config.get('api_keys', {})
        
        env_mapping = {
            'serper_api_key': 'SERPER_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY', 
            'openai_api_base': 'OPENAI_API_BASE',
            'jina_api_key': 'JINA_API_KEY'
        }
        
        for config_key, env_key in env_mapping.items():
            value = api_keys.get(config_key)
            if value and value != f"your_{config_key}_here":
                os.environ[env_key] = str(value)
                print(f"Environment variables have been set.: {env_key}")
            else:
                print(f"Warning: {config_key} Not set in the configuration file or using default values")
    
    def get_api_key(self, key_name: str) -> str:

        env_mapping = {
            'serper_api_key': 'SERPER_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY',
            'jina_api_key': 'JINA_API_KEY'
        }
        
        env_key = env_mapping.get(key_name)
        if env_key:
            env_value = os.environ.get(env_key)
            if env_value:
                return env_value
        
        if self.config is None:
            self.load_config()
            
        api_keys = self.config.get('api_keys', {})
        return api_keys.get(key_name, '')
    
    def get_setting(self, setting_name: str, default_value: Any = None) -> Any:

        if self.config is None:
            self.load_config()
            
        settings = self.config.get('settings', {})
        return settings.get(setting_name, default_value)
    
    def get_local_server_config(self) -> Dict[str, str]:
        """Get local server configuration (address and port)"""
        if self.config is None:
            self.load_config()
            
        settings = self.config.get('settings')
        local_server = settings.get('local_server')
        
        return {
            'address': local_server.get('address'),
            'port': local_server.get('port')
        }


_config_loader = None


def get_config_loader(config_path: str = "eval_config.yaml") -> ConfigLoader:

    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def load_config_and_set_env(config_path: str = "eval_config.yaml") -> None:
    try:
        if not os.path.sep in config_path and not os.path.isabs(config_path):
            possible_paths = [
                config_path,
                os.path.join(os.path.dirname(__file__), config_path), 
                os.path.join(os.path.dirname(__file__), "..", config_path), 
            ]
            
            actual_config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_config_path = path
                    break
            
            if actual_config_path is None:
                print(f"Warning: No configuration file was found at any of the following path '{config_path}':")
                for path in possible_paths:
                    print(f"  - {os.path.abspath(path)}")
                config_path = possible_paths[0]
            else:
                config_path = actual_config_path
                print(f"Find the configuration file: {os.path.abspath(config_path)}")
        
        config_loader = get_config_loader(config_path)
        config_loader.load_config()
        config_loader.set_env_vars()
        print("Configuration loading completed.")
    except Exception as e:
        print(f"Error when loading configuration: {e}")
        print("The default configuration or existing values from environment variables will be used.")


def get_api_key(key_name: str) -> str:

    config_loader = get_config_loader()
    return config_loader.get_api_key(key_name)


def get_local_server_config() -> Dict[str, str]:
    """Get local server configuration from config file"""
    config_loader = get_config_loader()
    return config_loader.get_local_server_config()


if __name__ == "__main__":
    load_config_and_set_env()