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
                # normalize host to scheme://netloc for comparison
                norm_host = None
                if host:
                    try:
                        from urllib.parse import urlparse

                        up = urlparse(host)
                        if up.scheme and up.netloc:
                            norm_host = f"{up.scheme}://{up.netloc}"
                        else:
                            norm_host = host
                    except Exception:
                        norm_host = host

                found_dirb = False
                for c in current_layer:
                    if "dirb" in c and norm_host:
                        # check for base host or netloc in the dirb command
                        if norm_host in c or ("//" in norm_host and norm_host.split("//",1)[1] in c):
                            found_dirb = True
                if found_dirb:
                    continue
            result.append(cmd)

        return {"deduplicated_commands": result}

    def post_step(self, command, command_output_file, previous_commands=None, command_output_override=None, similar_context=None):
        """Return a minimal summary structure expected by `execute_command`.

        This keeps behavior deterministic and avoids external calls during tests.
        """
        summary = ""
        services = []
        recommended = []
        # Very small heuristic: if output contains 'open' and a port number, record an http service
        content = ""
        if command_output_override is not None:
            content = command_output_override
        else:
            try:
                with open(command_output_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                content = ""

        if "open" in content.lower():
            # naive port/service extraction
            import re

            ports = re.findall(r"(\d{1,5})/tcp\s+open\s+([a-zA-Z0-9_\-]+)", content)
            for p, svc in ports:
                services.append(f"{svc} {p}")

        return {"summary": summary, "recommended_steps": recommended, "services_found": services}

    def executive_summary(self, base_dir):
        return "Fake executive summary"
