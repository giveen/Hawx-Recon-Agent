#!/usr/bin/env python3
"""
Run layer0 commands from configs/layer0.yaml against a given target using FakeLLM.
Usage: python3 scripts/run_initial_test.py 192.168.0.4
"""
import sys
import os
import yaml
from agent.llm.fake_llm import FakeLLM
from agent.workflow.runner import run_layer
from agent.workflow.output import execute_command


def load_layer0_for_host():
    p = os.path.join(os.getcwd(), "configs", "layer0.yaml")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("host_mode", {}).get("commands", [])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_initial_test.py <target_ip>")
        sys.exit(1)
    target = sys.argv[1]
    commands_cfg = load_layer0_for_host()
    commands = []
    for cmd in commands_cfg:
        c = cmd.get("command", "").replace("{target}", target)
        if c:
            commands.append(c)
    if not commands:
        print("No layer0 commands found for host_mode in configs/layer0.yaml")
        sys.exit(1)

    llm = FakeLLM()
    base_dir = os.path.join(os.getcwd(), "triage", target)
    os.makedirs(base_dir, exist_ok=True)
    # run_layer expects llm_client and will call execute_command internally
    recs = run_layer(commands, 0, llm, base_dir, type("R", (), {"services": [], "commands": {}})(), interactive=False)
    print("\nRecommended next steps:\n", recs)
