import base64
import enum
import io
import json
import os
import subprocess
import logging
import zlib
from json import JSONEncoder
from json.decoder import JSONArray

import PIL.Image
import numpy as np
from PIL.PngImagePlugin import iTXt
from openai import OpenAI, OpenAIError

from PIL import Image, ImageDraw, ImageFont

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
dafault_user_prompt = (
    "1girl,")
dafault_llm_sys_prompt = (
    "You are an AI prompt word engineer. Use the provided keywords to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")
dafault_llm_sys_prompt_vision = (
    "You are an AI prompt word engineer. Use the provided image to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")
dafault_llm_user_prompt_vision = (
    "What's in this image?")
dafault_llm_user_prompt = (
    "A superstar on stage.")
dafault_settings_llm_url = (
    "http://localhost:1234/v1")
dafault_settings_llm_api_key = (
    "lm-studio")
dafault_settings_llm_model_name = (
    "llama3.1")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_history_array = []
llm_history_array_eye = []


def encodeX(clip, text):
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]


def check_api_uri(llm_apiurl, llm_apikey, clientx):
    try:
        if clientx.base_url != llm_apiurl or clientx.api_key != llm_apikey:
            clientx = OpenAI(base_url=llm_apiurl, api_key=llm_apikey)
    except OpenAIError as e:
        log.warning(e)


def print_obj_x(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):
            print(attr + "==>", getattr(obj, attr))


def call_llm_text(clip,
                  text_prompt_postive, text_prompt_negative,
                  llm_apiurl, llm_apikey, llm_api_model_name,
                  llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
                  llm_text_ur_prompt,
                  llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                  llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                  llm_recursive_use, llm_keep_your_prompt_ahead,
                  llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                  llm_post_action_cmd_feedback_type, llm_post_action_cmd):
    llm_before_action_cmd_return_value = do_subprocess_action(llm_before_action_cmd)
    if EnumCmdReturnType.LLM_USER_PROMPT.value in llm_before_action_cmd_feedback_type:
        llm_text_ur_prompt += llm_before_action_cmd_return_value

    if llm_recursive_use and (len(llm_history_array) > 1):
        llm_text_ur_prompt = (llm_text_ur_prompt if llm_keep_your_prompt_ahead else "") + " " + \
                             llm_history_array[len(llm_history_array) - 1][0]
    try:
        check_api_uri(llm_apiurl, llm_apikey, client)

        completion = client.chat.completions.create(
            model=f"{llm_api_model_name}",
            messages=[
                {"role": "system", "content": llm_text_system_prompt},
                {"role": "user", "content": llm_text_ur_prompt}
            ],
            max_tokens=llm_text_max_token,
            temperature=llm_text_tempture,

        )
    except OpenAIError as e:
        llm_history_array.append([e.message, e.message, e.message, e.message])
        return e.message, llm_history_array

    result_text = completion.choices[0].message.content

    result_text = result_text.replace('\n', ' ')

    do_subprocess_action(llm_post_action_cmd)
    return result_text


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def image_to_base64(pli_image, pnginfo=None):
    image_data = io.BytesIO()
    pli_image.save(image_data, format='PNG', pnginfo=pnginfo)
    image_data_bytes = image_data.getvalue()
    encoded_image = "data:image/png;base64," + base64.b64encode(image_data_bytes).decode('utf-8')
    return encoded_image


def convert(images, imageType=None, prompt=None, extra_pnginfo=None):
    if imageType is None:
        imageType = imageType

    result = list()
    for i in images:
        img = tensor_to_pil(i)
        metadata = None

        encoded_image = image_to_base64(img, pnginfo=metadata)
        result.append(encoded_image)
    base64Images = JSONEncoder().encode(result)
    # print(images)
    return {"ui": {"base64Images": result, "imageType": [imageType]}, "result": (base64Images,)}


def call_llm_eye_open(clip,
                      text_prompt_postive, text_prompt_negative,
                      llm_apiurl, llm_apikey, llm_api_model_name,
                      llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
                      llm_text_ur_prompt,
                      llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                      llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                      llm_recursive_use, llm_keep_your_prompt_ahead,
                      llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                      llm_post_action_cmd_feedback_type, llm_post_action_cmd):
    llm_before_action_cmd_return_value = do_subprocess_action(llm_before_action_cmd)

    base64_image = """data:image/jpeg;base64,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"""
    # try:
    # image = open(image_to_llm_vision, "rb").read()
    # image = Image.fromarray(np.clip(255. * image_to_llm_vision.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    # image = tensor_to_pil(image_to_llm_vision)
    # base64_image = base64.b64encode(image).decode("utf-8")
    base64_image = str(json.loads(image_to_llm_vision)[0])
    # log.warning(f"[2][call_llm_eye_open][ base64_image]: "+ base64_image)
    # base64_image.replace("[\"","")
    # base64_image.replace("]\"", "")
    if not str(base64_image).startswith("data:image"):
        base64_image = f"data:image/jpeg;base64,{base64_image}"
    # log.warning(f"[3][call_llm_eye_open][ base64_image]: ",base64_image)

    # print("[][call_llm_eye_open][]base64_image", base64_image)

    # except Exception as e:
    # log.error(f"[][][call_llm_eye_open]IO Error: {e}")
    # llm_history_array.append(["missing input image ?", e, e, e])
    # return "[][call_llm_eye_open]missing input image ?" + e, self.llm_history_array
    # log.warning("[][Auto-llm-vision][Exception]",e)

    try:
        check_api_uri(llm_apiurl, llm_apikey,client)

        completion = client.chat.completions.create(
            model=f"{llm_api_model_name}",
            messages=[
                {
                    "role": "system",
                    "content": f"{llm_vision_system_prompt}",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{llm_vision_ur_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=llm_vision_max_token,
            temperature=llm_vision_tempture,
        )

    except OpenAIError as e:
        log.error(f"[][][call_llm_eye_open]Model Error: {e.message}")
        llm_history_array.append([e.message, e.message, e.message, e.message])
        return e.message

    # for chunk in completion:
    #     if chunk.choices[0].delta.content:
    #         result = chunk.choices[0].delta.content
    # print(chunk.choices[0].delta.content, end="", flush=True)
    result_vision = completion.choices[0].message.content
    result_vision = result_vision.replace('\n', ' ')
    result_translate = "wawa"

    llm_history_array.append([result_vision, llm_vision_system_prompt, llm_vision_ur_prompt, result_translate])
    if len(llm_history_array) > 3:
        llm_history_array.remove(llm_history_array[0])
    print("[][auto-llm][call_llm_eye_open] result_vision=", result_vision)

    # do_subprocess_action(llm_post_action_cmd)

    return result_vision


def do_subprocess_action(llm_post_action_cmd):
    # if llm_post_action_cmd.__len__() <= 0:
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
    # if (llm_text_result_append_enabled or llm_vision_result_append_enabled) is False:
    #     encode_pos = encodeX(clip, text_prompt_postive)
    #     return encode_pos, encode_neg, text_prompt_postive, text_prompt_negative, 'LLM-disabled',
    result_text = ""
    if llm_text_result_append_enabled:
        result_text = call_llm_text(clip,
                                    text_prompt_postive, text_prompt_negative,
                                    llm_apiurl, llm_apikey, llm_api_model_name,
                                    llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled,
                                    llm_text_system_prompt, llm_text_ur_prompt,
                                    llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                                    llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                                    llm_recursive_use, llm_keep_your_prompt_ahead,
                                    llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                                    llm_post_action_cmd_feedback_type, llm_post_action_cmd)

    result_vision = ""
    if llm_vision_result_append_enabled:
        result_vision = call_llm_eye_open(clip,
                                          text_prompt_postive, text_prompt_negative,
                                          llm_apiurl, llm_apikey, llm_api_model_name,
                                          llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled,
                                          llm_text_system_prompt, llm_text_ur_prompt,
                                          llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled,
                                          llm_vision_system_prompt, llm_vision_ur_prompt, image_to_llm_vision,
                                          llm_recursive_use, llm_keep_your_prompt_ahead,
                                          llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                                          llm_post_action_cmd_feedback_type, llm_post_action_cmd)

    result_text_vision = ",".join([text_prompt_postive, result_text, result_vision])
    encode_pos = encodeX(clip, result_text_vision)
    log.warning("[][Auto-LLM][ LLM-Text-Answer] " + result_text)

    log.warning("[][Auto-LLM][ LLM-Vision-Answer] " + result_vision)
    log.warning("[][Auto-LLM][SD-PostivePrompt + LLM-Text-Vision-Answer] " + result_text_vision)

    return (encode_pos, encode_neg, text_prompt_postive, text_prompt_negative,
            result_text, result_vision, result_text_vision,)


class EnumCmdReturnType(enum.Enum):
    PASS = 'Pass'
    JUST_CALL = 'just-call'
    LLM_USER_PROMPT = 'LLM-USER-PROMPT'
    LLM_VISION_IMG_PATH = 'LLM-VISION-IMG_PATH'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


class LLM_ALL:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            #https://docs.comfy.org/essentials/custom_node_more_on_inputs#hidden-inputs
            "hidden": {
            },
            "optional": {
                "image_to_llm_vision": ("*",),
            },
            "required": {
                "clip": ("CLIP",),
                # "image_to_llm_vision": ("STRING", {"multiline": True,}),
                "llm_text_result_append_enabled": ([True, False],),
                "llm_vision_result_append_enabled": ([True, False],),

                "text_prompt_postive": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": dafault_user_prompt}),
                "text_prompt_negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "llm_keep_your_prompt_ahead": ([True, False],),
                "llm_recursive_use": ([False, True],),
                # "text_prompt_postive": ("CONDITIONING",),
                # "text_prompt_negative": ("CONDITIONING",),
                # "text_llm_prompt_postive": ("text_llm_prompt",),

                "llm_apiurl": ("STRING", {"multiline": False, "default": dafault_settings_llm_url}),
                "llm_apikey": ("STRING", {"multiline": False, "default": dafault_settings_llm_api_key}),
                "llm_api_model_name": ("STRING", {"multiline": False, "default": "llama3.1"}),
                "llm_text_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_text_tempture": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.01}),

                # "llm_text_system_prompt": ("STRING", {"multiline": False, "default": dafault_llm_sys_prompt}),
                # "llm_text_ur_prompt": ("STRING", {"multiline": False, "default": dafault_llm_user_prompt}),


                "llm_text_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": dafault_llm_sys_prompt}),
                "llm_text_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": dafault_llm_user_prompt}),

                "llm_vision_max_token": ("INT", {"default": 50, "min": 10, "max": 1024, "step": 1}),
                "llm_vision_tempture": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.01}),
                "llm_vision_system_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": dafault_llm_sys_prompt_vision}),
                "llm_vision_ur_prompt": (
                    "STRING", {"multiline": True, "dynamicPrompts": True, "default": dafault_llm_user_prompt_vision}),

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

    def call_all(self, clip,
                 text_prompt_postive, text_prompt_negative,
                 llm_apiurl, llm_apikey, llm_api_model_name,
                 llm_text_max_token, llm_text_tempture, llm_text_result_append_enabled, llm_text_system_prompt,
                 llm_text_ur_prompt,
                 llm_vision_max_token, llm_vision_tempture, llm_vision_result_append_enabled, llm_vision_system_prompt,
                 llm_vision_ur_prompt, image_to_llm_vision,
                 llm_recursive_use, llm_keep_your_prompt_ahead,
                 llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                 llm_post_action_cmd_feedback_type, llm_post_action_cmd):
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


class PngInfo:
    """
    PNG chunk container (for use with save(pnginfo=))

    """

    def __init__(self) -> None:
        self.chunks: list[tuple[bytes, bytes, bool]] = []

    def add(self, cid: bytes, data: bytes, after_idat: bool = False) -> None:
        """Appends an arbitrary chunk. Use with caution.

        :param cid: a byte string, 4 bytes long.
        :param data: a byte string of the encoded data
        :param after_idat: for use with private chunks. Whether the chunk
                           should be written after IDAT

        """

        self.chunks.append((cid, data, after_idat))

    def add_itxt(
            self,
            key: str | bytes,
            value: str | bytes,
            lang: str | bytes = "",
            tkey: str | bytes = "",
            zip: bool = False,
    ) -> None:
        """Appends an iTXt chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        :param zip: compression flag

        """

        if not isinstance(key, bytes):
            key = key.encode("latin-1", "strict")
        if not isinstance(value, bytes):
            value = value.encode("utf-8", "strict")
        if not isinstance(lang, bytes):
            lang = lang.encode("utf-8", "strict")
        if not isinstance(tkey, bytes):
            tkey = tkey.encode("utf-8", "strict")

        if zip:
            self.add(
                b"iTXt",
                key + b"\0\x01\0" + lang + b"\0" + tkey + b"\0" + zlib.compress(value),
            )
        else:
            self.add(b"iTXt", key + b"\0\0\0" + lang + b"\0" + tkey + b"\0" + value)

    def add_text(
            self, key: str | bytes, value: str | bytes | iTXt, zip: bool = False
    ) -> None:
        """Appends a text chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key, text or an
           :py:class:`PIL.PngImagePlugin.iTXt` instance
        :param zip: compression flag

        """
        if isinstance(value, iTXt):
            return self.add_itxt(
                key,
                value,
                value.lang if value.lang is not None else b"",
                value.tkey if value.tkey is not None else b"",
                zip=zip,
            )

        # The tEXt chunk stores latin-1 text
        if not isinstance(value, bytes):
            try:
                value = value.encode("latin-1", "strict")
            except UnicodeError:
                return self.add_itxt(key, value, zip=zip)

        if not isinstance(key, bytes):
            key = key.encode("latin-1", "strict")

        if zip:
            self.add(b"zTXt", key + b"\0\0" + zlib.compress(value))
        else:
            self.add(b"tEXt", key + b"\0" + value)
