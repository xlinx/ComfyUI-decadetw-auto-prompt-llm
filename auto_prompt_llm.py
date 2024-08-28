import base64
import enum
import io
import logging
import random
import subprocess
import pprint
import numpy as np
import requests
from PIL import Image

random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5'  # ⇅
restore_progress_symbol = '\U0001F300'  # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐
log = logging.getLogger("[auto-llm]")
# log.setLevel(logging.INFO)
# Logging
default_user_prompt = (
    "1girl,")
default_llm_sys_prompt = (
    "You are an AI prompt word engineer. Use the provided keywords to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")
default_llm_sys_prompt_vision = (
    "You are an AI prompt word engineer. Use the provided image to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")
default_llm_user_prompt_vision = (
    "What's in this image?")
default_llm_user_prompt = (
    "A superstar on stage.")
default_settings_llm_url = (
    "http://localhost:1234/v1/chat/completions")
default_settings_llm_api_key = (
    "lm-studio")
default_settings_llm_model_name = (
    "llama3.1")
default_openai_echo = """{"id": "", "choices": [{"finish_reason": "stop", "index": 0, "logprobs": null, "message": {"content": "LLM SERVER not found", "refusal": null, "role": "assistant", "function_call": null, "tool_calls": null}}], "created": 0, "model": "x", "object": "x", "service_tier": null, "system_fingerprint": null, "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}}"""
llm_history_array = []
llm_history_array_eye = []
last_time_result_text_vision = ''


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def image_to_base64(image):
    pli_image = tensor_to_pil(image)
    image_data = io.BytesIO()
    pli_image.save(image_data, format='PNG', pnginfo=None)
    image_data_bytes = image_data.getvalue()
    encoded_image = "data:image/png;base64," + base64.b64encode(image_data_bytes).decode('utf-8')
    # log.warning("[][image_to_base64][]"+encoded_image)
    # if not str(base64_image).startswith("data:image"):
    #     base64_image = f"data:image/jpeg;base64,{base64_image}"
    return encoded_image


def encodeX(clip, text):
    print("[][Auto-LLM][clip-encode-text] text=", text)
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]


def print_obj_x(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):
            print(attr + "==>", getattr(obj, attr))


def call_llm_mix(headers_x, json_str_x, llm_apiurl):
    result_mix = ''

    try:
        completion = requests.post(llm_apiurl, headers=headers_x, json=json_str_x).json()
        # print(f'[][][]{completion}')
        print('call_llm_mix')
        pprint.pprint(completion)
        result_mix = completion['choices'][0]['message']['content']
    except Exception as e:
        e = str(e)
        llm_history_array.append([e, e, e, e])
        result_mix = "[Auto-LLM][Result][Missing LLM-Text]" + e
        log.warning("[Auto-LLM][OpenAILib][OpenAIError]Missing LLM Server?")
    result_mix = result_mix.replace('\n', ' ')
    return result_mix


def do_subprocess_action(llm_post_action_cmd):
    if len(llm_post_action_cmd) <= 1:
        return ""
    p = subprocess.Popen(llm_post_action_cmd.split(" "), text=True, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (out, err) = p.communicate()
    ret = p.wait()
    ret = True if ret == 0 else False
    if ret:
        log.warning("Command succeeded. " + llm_post_action_cmd + " output=" + out)
        llm_history_array.append(["[O]PostAction-Command succeeded.", err, llm_post_action_cmd, out])
    else:
        log.warning("Command failed. " + llm_post_action_cmd + " err=" + err)
        llm_history_array.append(["[X]PostAction-Command failed.", err, llm_post_action_cmd, out])
    return out


def call_llm_all(clip,
                 text_prompt_postive, text_prompt_negative,
                 llm_apiurl, llm_apikey, llm_api_model_name,
                 llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
                 llm_text_ur_prompt,
                 llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                 llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                 llm_recursive_use, llm_keep_your_prompt_ahead,
                 llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                 llm_post_action_cmd_feedback_type, llm_post_action_cmd):
    encode_neg = encodeX(clip, text_prompt_negative)

    result_text_vision = ""
    result_before = do_subprocess_action(llm_before_action_cmd)
    if llm_before_action_cmd_feedback_type == EnumCmdReturnType.LLM_USER_PROMPT.value:
        result_text_vision += result_before

    headers_x = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {llm_apikey}',
    }
    result_text = ''
    json_x1 = {
        'model': f'{llm_api_model_name}',
        'messages': [
            {'role': 'system', 'content': f'{llm_text_system_prompt}'},
            {'role': 'user', 'content': f'{llm_text_ur_prompt}'}
        ],
        'max_tokens': f'{llm_text_max_token}',
        'temperature': f'{llm_text_tempture}',
        'stream': f'{False}',
    }

    result_text = call_llm_mix(headers_x, json_x1, llm_apiurl)

    result_vision = ''
    if llm_vision_result_append_enabled:
        base64_image = image_to_base64(image_to_llm_vision)
        json_x2 = {
            'model': f'{llm_api_model_name}',
            'messages': [
                {'role': 'system', 'content': f'{llm_vision_system_prompt}'},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': f'{llm_vision_ur_prompt}'},
                    {'type': 'image_url', 'image_url': {'url': f'{base64_image}'}}
                ]}
            ],
            'max_tokens': f'{llm_vision_max_token}',
            'temperature': f'{llm_vision_tempture}',
            'stream': f'{False}',
        }
        result_vision = call_llm_mix(headers_x, json_x2, llm_apiurl)

    result_after = do_subprocess_action(llm_post_action_cmd)
    if llm_post_action_cmd_feedback_type == EnumCmdReturnType.LLM_USER_PROMPT.value:
        result_text_vision += result_after

    if not llm_text_result_append_enabled:
        result_text = ''
    result_text_vision = ",".join([result_text, result_vision])
    if llm_keep_your_prompt_ahead:
        result_text_vision = text_prompt_postive + ',' + result_text_vision

    result_text_vision = result_text_vision.replace(',,', ',')

    global last_time_result_text_vision
    if llm_recursive_use:
        final_result = last_time_result_text_vision + result_text_vision
    else:
        final_result = result_text_vision

    last_time_result_text_vision = result_text_vision

    encode_pos = encodeX(clip, final_result)

    return (encode_pos, encode_neg, text_prompt_postive, text_prompt_negative,
            result_text, result_vision, result_text_vision,)


class dict_2_class:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class dict_2_class_pass:
    pass


class default_openai_completion_class:
    choices = [{"message": {"content": "Missing LLM Server"}}]


class EnumCmdReturnType(enum.Enum):
    PASS = 'Pass'
    JUST_CALL = 'just-call'
    LLM_USER_PROMPT = 'LLM-USER-PROMPT'
    LLM_VISION_IMG_PATH = 'LLM-VISION-IMG_PATH'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class AnyTypeX(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


class LLM_TEXT:
    @classmethod
    def IS_CHANGED(s, is_trigger_every_generated):
        if is_trigger_every_generated:
            return random.random()
        else:
            return 0
    @classmethod
    def INPUT_TYPES(cls):
        return {  #https://docs.comfy.org/essentials/custom_node_more_on_inputs#hidden-inputs
            "hidden": {
                "image_to_llm_vision": ("IMAGE",),
                "llm_vision_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_vision_tempture": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.01}),
                "llm_vision_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt_vision}),
                "llm_vision_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt_vision}),
                "llm_vision_result_append_enabled": ([True, False],),
            },
            "optional": {

            },
            "required": {
                "clip": ("CLIP",),
                # "image_to_llm_vision": ("STRING", {"multiline": True,}),
                "llm_text_result_append_enabled": ([True, False],),

                "text_prompt_postive": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_user_prompt}),
                "text_prompt_negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "llm_keep_your_prompt_ahead": ([True, False],),
                "llm_recursive_use": ([False, True],),

                "llm_apiurl": ("STRING", {"multiline": False, "default": default_settings_llm_url}),
                "llm_apikey": ("STRING", {"multiline": False, "default": default_settings_llm_api_key}),
                "llm_api_model_name": ("STRING", {"multiline": False, "default": "llama3.1"}),
                "llm_text_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_text_tempture": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 2.0, "step": 0.01}),

                # "llm_text_system_prompt": ("STRING", {"multiline": False, "default": dafault_llm_sys_prompt}),
                # "llm_text_ur_prompt": ("STRING", {"multiline": False, "default": dafault_llm_user_prompt}),

                "llm_text_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt}),
                "llm_text_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt}),

                "llm_before_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_before_action_cmd": ("STRING", {"multiline": False, "default": ""}),
                "llm_post_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_post_action_cmd": (
                    "STRING", {"multiline": False,  #curl http://localhost:11434/api/generate -d '{"keep_alive": 0}'
                               "default": ""}),

            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("postive", "negative", "orignal-postive", "orignal-negative",
                    "🌀LLM-Text",
                    "🌀LLM-Vision",
                    "🌀postive+LLM-Text+LLM-Vision")
    FUNCTION = "call_all"
    CATEGORY = "🧩 Auto-Prompt-LLM"

    def call_all(self, clip,
                 text_prompt_postive, text_prompt_negative,
                 llm_apiurl, llm_apikey, llm_api_model_name,
                 llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
                 llm_text_ur_prompt,
                 # llm_vision_max_token, llm_vision_tempture, llm_vision_system_prompt,
                 # llm_vision_ur_prompt, image_to_llm_vision, llm_vision_result_append_enabled,
                 llm_recursive_use, llm_keep_your_prompt_ahead,
                 llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                 llm_post_action_cmd_feedback_type, llm_post_action_cmd):
        image_to_llm_vision = None
        llm_vision_max_token = None
        llm_vision_tempture = None
        llm_vision_system_prompt = None
        llm_vision_ur_prompt = None
        llm_vision_result_append_enabled = False
        return call_llm_all(clip,
                            text_prompt_postive, text_prompt_negative,
                            llm_apiurl, llm_apikey, llm_api_model_name,
                            llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled,
                            llm_text_system_prompt, llm_text_ur_prompt,
                            llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                            llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                            llm_recursive_use, llm_keep_your_prompt_ahead,
                            llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                            llm_post_action_cmd_feedback_type, llm_post_action_cmd)


class LLM_VISION:
    @classmethod
    def IS_CHANGED(s, is_trigger_every_generated):
        if is_trigger_every_generated:
            return random.random()
        else:
            return 0
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # https://docs.comfy.org/essentials/custom_node_more_on_inputs#hidden-inputs
            "hidden": {
                "llm_text_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt}),
                "llm_text_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt}),
                "llm_text_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_text_tempture": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 2.0, "step": 0.01}),
                "llm_text_result_append_enabled": ([True, False],),
            },
            "optional": {

            },
            "required": {

                "clip": ("CLIP",),
                "image_to_llm_vision": ("IMAGE",),

                "llm_vision_result_append_enabled": ([True, False],),

                "text_prompt_postive": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_user_prompt}),
                "text_prompt_negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "llm_keep_your_prompt_ahead": ([True, False],),
                "llm_recursive_use": ([False, True],),

                "llm_apiurl": ("STRING", {"multiline": False, "default": default_settings_llm_url}),
                "llm_apikey": ("STRING", {"multiline": False, "default": default_settings_llm_api_key}),
                "llm_api_model_name": ("STRING", {"multiline": False, "default": "llama3.1"}),

                "llm_vision_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_vision_tempture": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.01}),
                "llm_vision_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt_vision}),
                "llm_vision_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt_vision}),

                "llm_before_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_before_action_cmd": ("STRING", {"multiline": False, "default": ""}),
                "llm_post_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_post_action_cmd": (
                    "STRING", {"multiline": False,  # curl http://localhost:11434/api/generate -d '{"keep_alive": 0}'
                               "default": ""}),

            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("postive", "negative", "orignal-postive", "orignal-negative",
                    "🌀LLM-Text",
                    "🌀LLM-Vision",
                    "🌀postive+LLM-Text+LLM-Vision")
    FUNCTION = "call_all"
    CATEGORY = "🧩 Auto-Prompt-LLM"

    def call_all(self, clip,
                 text_prompt_postive, text_prompt_negative,
                 llm_apiurl, llm_apikey, llm_api_model_name,
                 # llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,llm_text_ur_prompt,
                 llm_vision_max_token, llm_vision_tempture, llm_vision_system_prompt,
                 llm_vision_ur_prompt, image_to_llm_vision,
                 llm_vision_result_append_enabled,
                 llm_recursive_use, llm_keep_your_prompt_ahead,
                 llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                 llm_post_action_cmd_feedback_type, llm_post_action_cmd):
        llm_text_system_prompt = None
        llm_text_ur_prompt = None
        llm_text_max_token = None
        llm_text_tempture = None
        llm_text_result_append_enabled = False
        return call_llm_all(clip,
                            text_prompt_postive, text_prompt_negative,
                            llm_apiurl, llm_apikey, llm_api_model_name,
                            llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled,
                            llm_text_system_prompt, llm_text_ur_prompt,
                            llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                            llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                            llm_recursive_use, llm_keep_your_prompt_ahead,
                            llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                            llm_post_action_cmd_feedback_type, llm_post_action_cmd)


class LLM_ALL:
    @classmethod
    def IS_CHANGED(s, is_trigger_every_generated):
        if is_trigger_every_generated:
            return random.random()
        else:
            return 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            #https://docs.comfy.org/essentials/custom_node_more_on_inputs#hidden-inputs
            "hidden": {
            },
            "optional": {
                # "trigger_any_type": ("IMAGE",),
            },
            "required": {
                "clip": ("CLIP",),
                "image_to_llm_vision": ("IMAGE",),
                "is_trigger_every_generated": ([True, False],),

                # "image_to_llm_vision": ("STRING", {"multiline": True,}),
                "llm_text_result_append_enabled": ([True, False],),
                "llm_vision_result_append_enabled": ([True, False],),

                "text_prompt_postive": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_user_prompt}),
                "text_prompt_negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "llm_keep_your_prompt_ahead": ([True, False],),
                "llm_recursive_use": ([False, True],),
                # "text_prompt_postive": ("CONDITIONING",),
                # "text_prompt_negative": ("CONDITIONING",),
                # "text_llm_prompt_postive": ("text_llm_prompt",),

                "llm_apiurl": ("STRING", {"multiline": False, "default": default_settings_llm_url}),
                "llm_apikey": ("STRING", {"multiline": False, "default": default_settings_llm_api_key}),
                "llm_api_model_name": ("STRING", {"multiline": False, "default": "llama3.1"}),
                "llm_text_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_text_tempture": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 2.0, "step": 0.01}),

                # "llm_text_system_prompt": ("STRING", {"multiline": False, "default": dafault_llm_sys_prompt}),
                # "llm_text_ur_prompt": ("STRING", {"multiline": False, "default": dafault_llm_user_prompt}),

                "llm_text_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt}),
                "llm_text_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt}),

                "llm_vision_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_vision_tempture": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.01}),
                "llm_vision_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_sys_prompt_vision}),
                "llm_vision_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": default_llm_user_prompt_vision}),

                "llm_before_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_before_action_cmd": ("STRING", {"multiline": False, "default": ""}),
                "llm_post_action_cmd_feedback_type": (EnumCmdReturnType.values(),),
                "llm_post_action_cmd": (
                    "STRING", {"multiline": False,  #curl http://localhost:11434/api/generate -d '{"keep_alive": 0}'
                               "default": ""}),

            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("postive", "negative", "orignal-postive", "orignal-negative",
                    "🌀LLM-Text",
                    "🌀LLM-Vision",
                    "🌀postive+LLM-Text+LLM-Vision")
    FUNCTION = "call_all"
    CATEGORY = "🧩 Auto-Prompt-LLM"

    # @classmethod
    # def IS_CHANGED(s):
    #     return True

    def call_all(self, is_trigger_every_generated=None, clip=None,
                 text_prompt_postive=None, text_prompt_negative=None,
                 llm_apiurl=None, llm_apikey=None, llm_api_model_name=None,
                 llm_text_max_token=None, llm_text_tempture=None, llm_text_result_append_enabled=None,
                 llm_text_system_prompt=None,
                 llm_text_ur_prompt=None,
                 llm_vision_max_token=None, llm_vision_tempture=None, llm_vision_result_append_enabled=None,
                 llm_vision_system_prompt=None,
                 llm_vision_ur_prompt=None, image_to_llm_vision=None,
                 llm_recursive_use=None, llm_keep_your_prompt_ahead=None,
                 llm_before_action_cmd_feedback_type=None, llm_before_action_cmd=None,
                 llm_post_action_cmd_feedback_type=None, llm_post_action_cmd=None):
        return call_llm_all(clip,
                            text_prompt_postive, text_prompt_negative,
                            llm_apiurl, llm_apikey, llm_api_model_name,
                            llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled,
                            llm_text_system_prompt, llm_text_ur_prompt,
                            llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                            llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                            llm_recursive_use, llm_keep_your_prompt_ahead,
                            llm_before_action_cmd_feedback_type, llm_before_action_cmd,

                            llm_post_action_cmd_feedback_type, llm_post_action_cmd)

# Unit test
# if __name__ == "__main__":
#     node = LLM_ALL()
#     example_prompt = "Generate a random number using the input as seed"
#     example_any = 5
#     result = call_llm_mix(call_llm_mix(headers_x, json_x, llm_apiurl)
#     print("Result:", result)


# def call_llm_text(clip,
#                   text_prompt_postive, text_prompt_negative,
#                   llm_apiurl, llm_apikey, llm_api_model_name,
#                   llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
#                   llm_text_ur_prompt,
#                   llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
#                   llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
#                   llm_recursive_use, llm_keep_your_prompt_ahead,
#                   llm_before_action_cmd_feedback_type, llm_before_action_cmd,
#                   llm_post_action_cmd_feedback_type, llm_post_action_cmd):
#     if llm_recursive_use and (len(llm_history_array) > 1):
#         llm_text_ur_prompt = (llm_text_ur_prompt if llm_keep_your_prompt_ahead else "") + " " + \
#                              llm_history_array[len(llm_history_array) - 1][0]
#     result_t = ''
#     try:
#         completion = requests.post(llm_apiurl,
#                                    headers={
#                                        'Content-Type': 'application/json',
#                                        'Authorization': f'Bearer {llm_apikey}',
#                                    },
#                                    json={
#                                        'model': f'{llm_api_model_name}',
#                                        'messages': [
#                                            {'role': 'system', 'content': f'{llm_text_system_prompt}'},
#                                            {'role': 'user', 'content': f'{llm_text_ur_prompt}'}
#                                        ],
#                                        'max_tokens': f'{llm_text_max_token}',
#                                        'temperature': f'{llm_text_tempture}',
#                                        'stream': f'{False}',
#                                    }
#                                    ).json()
#         # print(f'[][][]{completion}')
#         print('call_llm_text')
#         pprint.pprint(completion)
#         result_t = completion['choices'][0]['message']['content']
#     except Exception as e:
#         e = str(e)
#         llm_history_array.append([e, e, e, e])
#         result_t = "[Auto-LLM][Result][Missing LLM-Text]" + e
#         log.warning("[Auto-LLM][OpenAILib][OpenAIError]Missing LLM Server?")
#     result_t = result_t.replace('\n', ' ')
#     return result_t
#
#
# def call_llm_eye_open(clip,
#                       text_prompt_postive, text_prompt_negative,
#                       llm_apiurl, llm_apikey, llm_api_model_name,
#                       llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
#                       llm_text_ur_prompt,
#                       llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
#                       llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
#                       llm_recursive_use, llm_keep_your_prompt_ahead,
#                       llm_before_action_cmd_feedback_type, llm_before_action_cmd,
#                       llm_post_action_cmd_feedback_type, llm_post_action_cmd):
#     # base64_image = """data:image/jpeg;base64,"""
#     base64_image = image_to_base64(image_to_llm_vision)
#
#     if not str(base64_image).startswith("data:image"):
#         base64_image = f"data:image/jpeg;base64,{base64_image}"
#     result_v = ''
#     try:
#         completion = requests.post(llm_apiurl,
#                                    headers={
#                                        'Content-Type': 'application/json',
#                                        'Authorization': f'Bearer {llm_apikey}',
#                                    },
#                                    json={
#                                        'model': f'{llm_api_model_name}',
#                                        'messages': [
#                                            {'role': 'system', 'content': f'{llm_vision_system_prompt}'},
#                                            {'role': 'user', 'content': [
#                                                {'type': 'text', 'text': f'{llm_vision_ur_prompt}'},
#                                                {'type': 'image_url', 'image_url': {'url': f'{base64_image}'}}
#                                            ]}
#                                        ],
#                                        'max_tokens': f'{llm_vision_max_token}',
#                                        'temperature': f'{llm_vision_tempture}',
#                                        'stream': f'{False}',
#                                    }
#                                    ).json()
#         # print(f'[][][]{completion}')
#         print('call_llm_eye_open')
#         pprint.pprint(completion)
#         result_v = completion['choices'][0]['message']['content']
#     except Exception as e:
#         e = str(e)
#         llm_history_array.append([e, e, e, e])
#         pprint.pprint(e)
#         result_v = "[Auto-LLM][Result][Missing LLM-Vision Module?]" + e
#
#     # log.warning("[Auto-LLM][OpenAILib][completion]" + json.dumps(completion, default=vars))
#     result_v = result_v.replace('\n', ' ')
#
#     return result_v
