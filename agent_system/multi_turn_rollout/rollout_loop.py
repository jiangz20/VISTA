# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
import random
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, to_object_array, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from vista.prompts import (
    IMAGE_ZOOM_IN_TOOLS,
    THINK_WITH_IMAGES_SYSTEM_PROMPT,
    append_teacher_context,
    deserialize_messages,
    interleave_text_images,
    message_images,
    sanitize_message_text_image_markers,
    serialize_messages,
)
from vista.hindsight import annotate_trajectory_events


QWEN_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
FORBIDDEN_GENERATED_VISION_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def sanitize_generated_vision_tokens(
    responses: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> np.ndarray:
    """Replace visual placeholder tokens emitted inside generated responses.

    Visual features exist only for prompt images. If the model generates a
    hidden ``image_pad`` token, Qwen sees one more image token than feature on
    the subsequent log-probability pass and aborts. Decoding with
    ``skip_special_tokens=True`` hides the offending token, so sanitize the
    tensors before decoding and mark the action invalid in the environment.
    """

    forbidden_ids: set[int] = set()
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for token in FORBIDDEN_GENERATED_VISION_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unknown_id:
            continue
        forbidden_ids.add(int(token_id))

    invalid_rows = torch.zeros(
        responses.size(0), dtype=torch.bool, device=responses.device
    )
    if not forbidden_ids:
        return invalid_rows.cpu().numpy()

    forbidden_mask = torch.zeros_like(responses, dtype=torch.bool)
    for token_id in forbidden_ids:
        forbidden_mask |= responses.eq(token_id)
    invalid_rows = forbidden_mask.any(dim=-1)
    if not bool(invalid_rows.any()):
        return invalid_rows.cpu().numpy()

    replacement_id = unknown_id
    if replacement_id is None:
        replacement_id = getattr(tokenizer, "eos_token_id", None)
    if replacement_id is None:
        replacement_id = getattr(tokenizer, "pad_token_id", None)
    if replacement_id is None:
        raise ValueError("Tokenizer has no safe token for generated vision-token sanitization")

    sanitized = responses.masked_fill(forbidden_mask, int(replacement_id))
    responses.copy_(sanitized)
    if input_ids.size(0) != responses.size(0) or input_ids.size(1) < responses.size(1):
        raise ValueError("Rollout input_ids do not contain the generated response suffix")
    input_ids[:, -responses.size(1) :].copy_(sanitized)
    return invalid_rows.cpu().numpy()

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def _prompt_preprocess_workers(self) -> int:
        env_config = getattr(self.config, "env", None)
        if env_config is None:
            return 1
        active_visual_config = env_config.get("active_visual", {})
        return max(1, int(active_visual_config.get("preprocess_num_workers", 1)))

    def build_text_prompt_sample(
        self,
        obs_content: str,
        data_source: Optional[str] = None,
    ) -> Dict:
        """
        Build a text-only prompt sample using the same chat-template path as rollout.
        This is used by OPID teacher scoring to reconstruct prompt-enhanced inputs.
        """
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}."
                )

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "obs_text": obs_content,
            "data_source": data_source,
        }

    def build_text_prompt_batch(
        self,
        obs_contents: List[str],
        data_sources: Optional[List[Optional[str]]] = None,
        meta_info: Optional[Dict] = None,
    ) -> DataProto:
        """
        Build a batch of text-only prompts. Used for OPID teacher scoring.
        """
        processed_samples = []
        for sample_idx, obs_content in enumerate(obs_contents):
            data_source = None if data_sources is None else data_sources[sample_idx]
            processed_samples.append(
                self.build_text_prompt_sample(obs_content=obs_content, data_source=data_source)
            )
        batch = collate_fn(processed_samples)
        return DataProto.from_single_dict(data=batch, meta_info=meta_info)

    def build_multimodal_prompt_sample(
        self,
        obs_content: str,
        images: List,
        data_source: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Dict:
        """Build a native think-with-images prompt from legacy marker text."""
        messages = [
            {"role": "system", "content": THINK_WITH_IMAGES_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": interleave_text_images(obs_content, list(images)),
            },
        ]
        return self.build_multimodal_chat_prompt_sample(
            messages=messages,
            data_source=data_source,
            max_length=max_length,
        )

    def build_multimodal_chat_prompt_sample(
        self,
        messages: List[Dict],
        data_source: Optional[str] = None,
        max_length: Optional[int] = None,
        image_preprocess_cache: Optional[Dict] = None,
        image_preprocess_cache_lock: Optional[Lock] = None,
    ) -> Dict:
        """Render native Qwen system/tools/assistant/tool multimodal history."""
        if self.processor is None:
            raise ValueError("active visual exploration requires a multimodal processor")
        prompt_max_length = int(
            self.config.data.max_prompt_length if max_length is None else max_length
        )
        if prompt_max_length <= 0:
            raise ValueError("multimodal prompt max_length must be positive")
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        sanitized_messages = sanitize_message_text_image_markers(messages)
        prompt = self.tokenizer.apply_chat_template(
            sanitized_messages,
            tools=IMAGE_ZOOM_IN_TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        source_images = message_images(sanitized_messages)
        raw_prompt = prompt.replace("<image>", QWEN_IMAGE_MARKER)
        if raw_prompt.count(QWEN_IMAGE_MARKER) != len(source_images):
            raise ValueError(
                f"active-visual prompt has {raw_prompt.count(QWEN_IMAGE_MARKER)} image slots "
                f"for {len(source_images)} images"
            )
        # A long trajectory can contain the root image plus several crops.  Each
        # image is individually bounded by ``process_image``, but their combined
        # visual-token count can still exceed the fixed student prompt budget.
        # Preserve every image and the full textual history; only lower the
        # shared pixel cap until the rendered multimodal prompt fits.
        max_pixels = int(self.config.data.get("max_pixels", 12845056))
        min_pixels = 256 * 256
        while True:
            if image_preprocess_cache is None:
                processed_images = [
                    process_image(
                        image,
                        max_pixels=max_pixels,
                        min_pixels=min(min_pixels, max_pixels),
                    )
                    for image in source_images
                ]
                image_inputs = self.processor.image_processor(
                    processed_images, return_tensors="pt"
                )
            else:
                cached_rows = []
                for image in source_images:
                    cache_key = (
                        id(image),
                        int(max_pixels),
                        int(min(min_pixels, max_pixels)),
                    )
                    if image_preprocess_cache_lock is None:
                        cached = image_preprocess_cache.get(cache_key)
                    else:
                        with image_preprocess_cache_lock:
                            cached = image_preprocess_cache.get(cache_key)
                    if cached is None:
                        # Hold the short-lived cache lock over a miss so that
                        # parallel group rows do not preprocess the same shared
                        # artifact concurrently.  The expensive processor call
                        # is still outside the per-row Python loop on hits.
                        lock = image_preprocess_cache_lock
                        if lock is None:
                            processed_image = process_image(
                                image,
                                max_pixels=max_pixels,
                                min_pixels=min(min_pixels, max_pixels),
                            )
                            single_inputs = self.processor.image_processor(
                                [processed_image], return_tensors="pt"
                            )
                            cached = (processed_image, dict(single_inputs))
                            image_preprocess_cache[cache_key] = cached
                        else:
                            with lock:
                                cached = image_preprocess_cache.get(cache_key)
                                if cached is None:
                                    processed_image = process_image(
                                        image,
                                        max_pixels=max_pixels,
                                        min_pixels=min(min_pixels, max_pixels),
                                    )
                                    single_inputs = self.processor.image_processor(
                                        [processed_image], return_tensors="pt"
                                    )
                                    cached = (processed_image, dict(single_inputs))
                                image_preprocess_cache.setdefault(cache_key, cached)
                                cached = image_preprocess_cache[cache_key]
                    cached_rows.append(cached)
                processed_images = [row[0] for row in cached_rows]
                image_inputs = (
                    {
                        key: torch.cat([row[1][key] for row in cached_rows], dim=0)
                        for key in cached_rows[0][1]
                    }
                    if cached_rows
                    else self.processor.image_processor([], return_tensors="pt")
                )
            image_grid_thw = image_inputs.get("image_grid_thw")
            if image_grid_thw is None or len(image_grid_thw) != len(processed_images):
                raise ValueError(
                    "teacher image grid count does not match its image slots: "
                    f"{0 if image_grid_thw is None else len(image_grid_thw)} grids for "
                    f"{len(processed_images)} images"
                )
            merge_length = self.processor.image_processor.merge_size**2
            expanded_prompt = raw_prompt
            visual_tokens = 0
            for grid in image_grid_thw:
                grid_tokens = int(grid.prod().item() // merge_length)
                visual_tokens += grid_tokens
                expanded_prompt = expanded_prompt.replace(
                    QWEN_IMAGE_MARKER,
                    "<|vision_start|>"
                    + "<|placeholder|>" * grid_tokens
                    + "<|vision_end|>",
                    1,
                )
            expanded_prompt = expanded_prompt.replace(
                "<|placeholder|>", self.processor.image_token
            )
            tokenized_prompt = self.tokenizer(
                expanded_prompt, return_tensors="pt", add_special_tokens=False
            )
            actual_prompt_length = int(tokenized_prompt["input_ids"].shape[-1])
            if actual_prompt_length <= prompt_max_length:
                break
            if max_pixels <= min_pixels or visual_tokens <= 0:
                raise ValueError(
                    "multimodal prompt exceeds its configured max length "
                    f"even at minimum image resolution: {actual_prompt_length} > "
                    f"{prompt_max_length}. Image tokens are never truncated."
                )
            non_visual_tokens = max(0, actual_prompt_length - visual_tokens)
            visual_budget = max(1, prompt_max_length - non_visual_tokens - 8)
            scale = min(0.9, max(0.25, visual_budget / visual_tokens * 0.95))
            next_max_pixels = max(min_pixels, int(max_pixels * scale))
            if next_max_pixels >= max_pixels:
                next_max_pixels = max(min_pixels, int(max_pixels * 0.75))
            max_pixels = next_max_pixels
        # The vision tower consumes these values in bfloat16 (see the Qwen3-VL
        # forward patch), but image processors produce float32 tensors.  A
        # The actor update transports both student and collective-experience teacher
        # images through Ray, so retaining float32 here doubles the driver,
        # object-store, and worker copies without preserving any values that
        # reach the model.  Compact only floating-point transport tensors;
        # grid metadata must remain integral.
        image_inputs = {
            key: value.to(dtype=torch.bfloat16)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in image_inputs.items()
        }
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],
            max_length=prompt_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index
        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids[0],
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask[0],
        )
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids,
            "raw_prompt_ids": self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            "multi_modal_data": {"image": processed_images},
            "multi_modal_inputs": {key: value for key, value in image_inputs.items()},
            "data_source": data_source,
        }

    def build_multimodal_prompt_batch(
        self,
        obs_contents: List[str],
        images: List[List],
        data_sources: Optional[List[Optional[str]]] = None,
        meta_info: Optional[Dict] = None,
        max_length: Optional[int] = None,
    ) -> DataProto:
        samples = []
        for index, content in enumerate(obs_contents):
            samples.append(
                self.build_multimodal_prompt_sample(
                    content,
                    images[index],
                    None if data_sources is None else data_sources[index],
                    max_length=max_length,
                )
            )
        return DataProto.from_single_dict(data=collate_fn(samples), meta_info=meta_info)

    def build_collective_experience_teacher_prompt_batch(
        self,
        message_payloads: List[str],
        target_images: List[List],
        experience_texts: List[str],
        experience_images: List[List],
        alignment_texts: List[str],
        repair_instructions: Optional[List[str]] = None,
        data_sources: Optional[List[Optional[str]]] = None,
        meta_info: Optional[Dict] = None,
        max_length: Optional[int] = None,
        reuse_image_preprocessing: bool = False,
    ) -> DataProto:
        """Build collective-experience prompts around unchanged student histories."""
        image_preprocess_cache = {} if reuse_image_preprocessing else None
        image_preprocess_cache_lock = Lock() if reuse_image_preprocessing else None

        def build_sample(index: int):
            payload = message_payloads[index]
            messages = deserialize_messages(payload, list(target_images[index]))
            messages = append_teacher_context(
                messages,
                experience_texts[index],
                list(experience_images[index]),
                alignment_text=alignment_texts[index] + (
                    "\n" + repair_instructions[index] if repair_instructions else ""
                ),
            )
            return self.build_multimodal_chat_prompt_sample(
                messages,
                None if data_sources is None else data_sources[index],
                max_length=max_length,
                image_preprocess_cache=image_preprocess_cache,
                image_preprocess_cache_lock=image_preprocess_cache_lock,
            )

        workers = min(self._prompt_preprocess_workers(), max(1, len(message_payloads)))
        if workers == 1:
            samples = [build_sample(index) for index in range(len(message_payloads))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                samples = list(executor.map(build_sample, range(len(message_payloads))))
        return DataProto.from_single_dict(data=collate_fn(samples), meta_info=meta_info)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        native_prompt_cache: Optional[Dict] = None,
        native_group_id: Optional[int] = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_base_texts = obs.get('text_base', None)
        obs_images = obs.get('image', None)
        obs_messages = obs.get('messages', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_text_base = obs_base_texts[item] if obs_base_texts is not None else obs_text
        obs_image = obs_images[item] if obs_images is not None else None
        if obs_image is None:
            obs_image_list = []
        elif isinstance(obs_image, (list, tuple)):
            obs_image_list = list(obs_image)
        elif isinstance(obs_image, np.ndarray) and obs_image.dtype == object:
            obs_image_list = list(obs_image)
        else:
            obs_image_list = [obs_image]
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = len(obs_image_list) > 0

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        if obs_messages is not None:
            native_messages = list(obs_messages[item])
            native_images = message_images(native_messages)
            if len(native_images) != len(obs_image_list):
                raise ValueError(
                    f"Native message/image mismatch for sample {item}: "
                    f"{len(native_images)} message images for {len(obs_image_list)} artifacts"
                )
            message_payload = serialize_messages(native_messages)
            cache_key = (native_group_id, message_payload)
            cached_prompt = (
                None if native_prompt_cache is None else native_prompt_cache.get(cache_key)
            )
            if cached_prompt is None:
                row_dict = self.build_multimodal_chat_prompt_sample(
                    messages=native_messages,
                    data_source=data_source,
                )
                if native_prompt_cache is not None:
                    native_prompt_cache[cache_key] = row_dict
            else:
                # Rollouts in one source group share the same image when their
                # serialized native histories are identical. Prompt tensors are
                # immutable here and can safely be stacked from shared references.
                row_dict = dict(cached_prompt)
            processed_images = list(row_dict["multi_modal_data"]["image"])
            row_dict.update(
                {
                    "anchor_obs": _obs_anchor,
                    "obs_text": "" if obs_text is None else str(obs_text),
                    "obs_text_base": "" if obs_text_base is None else str(obs_text_base),
                    "obs_images": to_object_array(processed_images),
                    "active_visual_messages_json": message_payload,
                    "index": item,
                    "data_source": data_source,
                }
            )
            if self.config.data.get('return_raw_chat', False):
                row_dict['raw_prompt'] = native_messages
            return row_dict

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            processed_images = [process_image(image) for image in obs_image_list]
            expected_placeholders = prompt_with_chat_template.count('<image>')
            if expected_placeholders != len(processed_images):
                raise ValueError(
                    f"Prompt/image mismatch for sample {item}: {expected_placeholders} placeholders "
                    f"for {len(processed_images)} images"
                )
            row_dict['multi_modal_data'] = {'image': processed_images}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'obs_text': obs_content,
            'obs_text_base': "" if obs_text_base is None else str(obs_text_base),
            'obs_images': to_object_array(processed_images if is_multi_modal else []),
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        rollout_group_size = max(1, int(self.config.env.rollout.n))

        def process_group(group_id: int):
            start = group_id * rollout_group_size
            stop = min(start + rollout_group_size, batch_size)
            native_prompt_cache = {} if obs.get("messages") is not None else None
            return [
                self.preprocess_single_sample(
                    item=item,
                    gen_batch=gen_batch,
                    obs=obs,
                    native_prompt_cache=native_prompt_cache,
                    native_group_id=group_id,
                )
                for item in range(start, stop)
            ]

        num_groups = (batch_size + rollout_group_size - 1) // rollout_group_size
        workers = min(self._prompt_preprocess_workers(), max(1, num_groups))
        if workers == 1:
            grouped_samples = [process_group(group_id) for group_id in range(num_groups)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                grouped_samples = list(executor.map(process_group, range(num_groups)))
        processed_samples = [sample for group in grouped_samples for sample in group]
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        rollout_group_size = int(self.config.env.rollout.n) if self.config.env.rollout.n > 0 else 1
        experience_cfg = getattr(self.config.algorithm, "collective_experience", {})
        collect_outcomes = bool(experience_cfg.get("enabled", False))
        for bs in range(batch_size):
            sample_id = bs // rollout_group_size
            rollout_id = bs % rollout_group_size
            episode_trajectory_valid = all(
                bool(data.get('is_action_valid', True))
                for data in total_batch_list[bs]
                if data.get('active_masks', False)
            )
            # Backfill the completed outcome onto every crop/final decision.
            # This turns the already-collected rollout into hindsight metadata;
            # it does not require a second policy generation.
            if collect_outcomes:
                annotate_trajectory_events(
                    total_batch_list[bs],
                    episode_return=float(episode_rewards[bs]),
                    trajectory_valid=episode_trajectory_valid,
                )
            # sum the rewards for each data in total_batch_list[bs]
            for step_position, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    step_num = int(data.get('step_num', step_position))
                    data['sample_id'] = sample_id
                    data['rollout_id'] = rollout_id
                    data['step_num'] = step_num
                    data['step_id'] = f"{sample_id}_{rollout_id}_{step_num}"
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    data['episode_trajectory_valid'] = episode_trajectory_valid
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Keep environment-side schedules aligned with the checkpointed trainer
        # update. Other environment managers need not implement this hook.
        if hasattr(envs, "set_global_step"):
            envs.set_global_step(int(gen_batch.meta_info.get("global_steps", 0)))

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.UUID(int=random.getrandbits(128)))
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.UUID(int=random.getrandbits(128)))
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        rollout_group_size = int(self.config.env.rollout.n) if self.config.env.rollout.n > 0 else 1
        sample_ids = np.asarray(
            [i // rollout_group_size for i in range(batch_size)],
            dtype=np.int64,
        )
        rollout_ids = np.asarray(
            [i % rollout_group_size for i in range(batch_size)],
            dtype=np.int64,
        )
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.UUID(int=random.getrandbits(128))) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            invalid_vision_token_rows = sanitize_generated_vision_tokens(
                batch_output.batch["responses"],
                batch_output.batch["input_ids"],
                self.tokenizer,
            )

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            step_nums = np.full(batch_size, _step, dtype=np.int64)
            batch.non_tensor_batch['sample_id'] = sample_ids
            batch.non_tensor_batch['rollout_id'] = rollout_ids
            batch.non_tensor_batch['step_num'] = step_nums
            batch.non_tensor_batch['step_id'] = np.asarray(
                [
                    f"{int(sample_ids[i])}_{int(rollout_ids[i])}_{int(step_nums[i])}"
                    for i in range(batch_size)
                ],
                dtype=object,
            )

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            for item_idx in np.flatnonzero(invalid_vision_token_rows):
                text_actions[item_idx] = (
                    "Invalid generated multimodal special token; use only text, "
                    "one valid tool call, or a boxed answer."
                )
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            if 'experience_event' in infos[0]:
                experience_events = []
                policy_version = int(gen_batch.meta_info.get("global_steps", 0))
                for item_idx, info in enumerate(infos):
                    event = info.get('experience_event')
                    if event is not None:
                        event = dict(event)
                        event['step'] = int(_step)
                        event['trajectory_id'] = str(traj_uid[item_idx])
                        event['policy_version'] = policy_version
                    experience_events.append(event)
                batch.non_tensor_batch['experience_event'] = np.asarray(experience_events, dtype=object)
                batch.non_tensor_batch['experience_task_id'] = np.asarray(
                    [str(info.get('task_id', '')) for info in infos], dtype=object
                )
                batch.non_tensor_batch['experience_ground_truth'] = np.asarray(
                    [str(info.get('ground_truth', '')) for info in infos], dtype=object
                )
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def _start_rollout_generation_session(self, actor_rollout_wg) -> bool:
        start_session = getattr(actor_rollout_wg, "start_rollout_generation_session", None)
        if start_session is None:
            return False

        try:
            start_session()
        except Exception:
            stop_session = getattr(actor_rollout_wg, "stop_rollout_generation_session", None)
            if stop_session is not None:
                try:
                    stop_session()
                except Exception:
                    pass
            raise
        return True

    def _stop_rollout_generation_session(self, actor_rollout_wg, session_started: bool) -> None:
        if not session_started:
            return
        stop_session = getattr(actor_rollout_wg, "stop_rollout_generation_session", None)
        if stop_session is not None:
            stop_session()

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        session_started = False
        try:
            session_started = self._start_rollout_generation_session(actor_rollout_wg)

            # Initial observations from the environment
            if self.config.algorithm.filter_groups.enable and is_train:
                # Dynamic Sampling (for DAPO and Dynamic GiGPO)
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            else:
                # Vanilla Sampling
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            assert len(total_batch_list) == len(total_episode_rewards)
            assert len(total_batch_list) == len(total_episode_lengths)
            assert len(total_batch_list) == len(total_traj_uid)
            assert len(total_batch_list) == len(totoal_tool_callings)

            # Create trajectory data
            gen_batch_output: DataProto = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
            )

            return gen_batch_output
        finally:
            self._stop_rollout_generation_session(actor_rollout_wg, session_started)
