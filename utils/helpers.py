import yaml

def load_prompt_from_yaml(yaml_path: str, agent_name: str, prompt_key: str) -> str:
    with open(yaml_path, 'r', encoding='utf-8') as file:
        templates = yaml.safe_load(file)
    return templates[agent_name][prompt_key]