"""
A simple fake LLM adapter used for unit tests to avoid external network calls.
"""
from typing import List, Dict


class FakeLLM:
    def __init__(self):
        pass

    def get_response(self, prompt: str) -> str:
        # Minimal deterministic placeholder
        return "{\"ok\": true}"

    def deduplicate_commands(self, commands: List[List[str]], layer: int) -> Dict:
        """Return a deduplicated command list using simple deterministic rules:
        - Remove any command that exactly matches a command in prior layers
        - If both `dirb` and `ffuf` appear targeting the same host, prefer `dirb` and remove `ffuf`
        """
        if not commands or not isinstance(commands, list):
            return {"deduplicated_commands": []}

        current_layer = commands[layer] if layer < len(commands) else []
        prior = [cmd for i, layer_cmds in enumerate(commands) if i != layer for cmd in layer_cmds]

        result = []
        for cmd in current_layer:
            if cmd in prior:
                continue
            # prefer dirb over ffuf when both present
            if "ffuf" in cmd:
                # find if there's a dirb for same host/url
                host = None
                parts = cmd.split()
                for p in parts:
                    if p.startswith("http://") or p.startswith("https://"):
                        host = p
                        break
                found_dirb = False
                for c in current_layer:
                    if "dirb" in c and host and host in c:
                        found_dirb = True
                if found_dirb:
                    continue
            result.append(cmd)

        return {"deduplicated_commands": result}
