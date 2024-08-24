"""
    .___                        .___           __
  __| _/____   ____ _____     __| _/____     _/  |___  _  __
 / __ |/ __ \_/ ___\\__  \   / __ |/ __ \    \   __\ \/ \/ /
/ /_/ \  ___/\  \___ / __ \_/ /_/ \  ___/     |  |  \     /
\____ |\___  >\___  >____  /\____ |\___  > /\ |__|   \/\_/
     \/    \/     \/     \/      \/    \/  \/
   _____          __           .____    .____       _____
  /  _  \  __ ___/  |_  ____   |    |   |    |     /     \
 /  /_\  \|  |  \   __\/  _ \  |    |   |    |    /  \ /  \
/    |    \  |  /|  | (  <_> ) |    |___|    |___/    Y    \
\____|__  /____/ |__|  \____/  |_______ \_______ \____|__  /
        \/                             \/       \/       \/
             · -—+ auto-prompt-llm-text-vision Extension for ComfyUI +—- ·
             trigger more detail using AI render AI
             https://decade.tw
"""

from .auto_prompt_llm import *
# import launch
#
# if not launch.is_installed("OpenAI"):
#     launch.run_pip(f"install OpenAI", "OpenAI")

NODE_CLASS_MAPPINGS = {
    "Auto-LLM-Text-Vision": LLM_ALL,
    "Auto-LLM-Text": LLM_TEXT,
    "Auto-LLM-Vision": LLM_VISION,
}
# WEB_DIRECTORY = "./js"
# __all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
print("\033[34mComfyUI Tutorial Nodes: \033[92mLoaded\033[0m")