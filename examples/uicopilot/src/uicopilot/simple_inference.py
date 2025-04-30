import logging
from typing import Any

import torch
from PIL import Image
from transformers.models.pix2struct import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers.models.t5 import T5Tokenizer, T5TokenizerFast

from .agents import AgentI2C, AgentOptimize, T_LLMModels
from .utils import BboxTree2Html, BboxTree2StyleList, Html2BboxTree, add_special_tokens, move_to_device

logger = logging.getLogger(__name__)


class SimpleInference:
    def __init__(self, llm_model: T_LLMModels, API_KEY: str, ENDPOINT: str | None = None):
        pretrained_model_name: str = "xcodemind/webcoder"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        processor_or_tuple = Pix2StructProcessor.from_pretrained(pretrained_model_name)
        self.processor = processor_or_tuple[0] if isinstance(processor_or_tuple, tuple) else processor_or_tuple
        self.tokenizer: T5Tokenizer | T5TokenizerFast = self.processor.tokenizer  # type: ignore

        self.model_bbox = Pix2StructForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            is_encoder_decoder=True,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        add_special_tokens(self.model_bbox, self.tokenizer)

        self.agent_i2c = AgentI2C(llm_model=llm_model, api_key=API_KEY, endpoint=ENDPOINT)
        self.agent_optimize = AgentOptimize(llm_model=llm_model, api_key=API_KEY, endpoint=ENDPOINT)

    def infer_bbox(self, image: Any) -> str:
        self.model_bbox.eval()
        with torch.no_grad():
            _input = "<body bbox=["
            decoder_input_ids = torch.Tensor(self.tokenizer.encode(_input, return_tensors="pt", add_special_tokens=True))
            decoder_input_ids = decoder_input_ids[..., :-1]
            encoding = self.processor(images=[image], text=[""], return_tensors="pt", images_kwargs={"max_patches": 1024})
            item = {
                "decoder_input_ids": decoder_input_ids,
                "flattened_patches": encoding["flattened_patches"].half() if self.device == "cuda" else encoding["flattened_patches"],  # type: ignore
                "attention_mask": encoding["attention_mask"],
            }
            item = move_to_device(item, self.device)

            outputs = self.model_bbox.generate(**item, max_new_tokens=2560, eos_token_id=self.tokenizer.eos_token_id, do_sample=True)

            prediction_html = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return prediction_html

    def locateByIndex(self, bboxTree: dict[str, Any], index: str) -> dict:
        target = bboxTree
        for i in list(filter(lambda x: x, index.split("-"))):
            target = target["children"][int(i)]
        return target

    def extract_html(self, html: str) -> str:
        if "```" in html:
            html = html.split("```")[1]
        if html[:4] == "html":
            html = html[4:]
        html = html.strip()
        return html

    def pruning(self, node: dict[str, Any], now_depth: int, max_depth: int, min_area: int) -> dict[str, Any] | None:
        bbox: list = node["bbox"]
        area: int = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area < min_area:
            return None
        if now_depth >= max_depth:
            node["children"] = []
        else:
            for idx, cnode in enumerate(node["children"]):
                node["children"][idx] = self.pruning(cnode, now_depth + 1, max_depth, min_area)
            node["children"] = list(filter(lambda x: x, node["children"]))
        return node

    def gen(self, image: Image.Image, max_depth: int = 100, min_area: int = 100) -> tuple[str, str, list[Image.Image]]:
        imgs: list[Image.Image] = []
        # infer
        prediction_html = self.infer_bbox(image)

        # pruning
        bboxTree = Html2BboxTree(prediction_html, size=image.size)
        if bboxTree is None:
            logger.error("bboxTree is None. Returning empty string, empty list")
            return "", "", []
        bboxTree = self.pruning(bboxTree, 1, max_depth, min_area)
        if bboxTree is None:
            logger.error("bboxTree is None after pruning. Returning empty string, empty list")
            return "", "", []

        # iter leaf node to gen ctree by agent
        indexList: list = BboxTree2StyleList(bboxTree, skip_leaf=False)
        # only leaf filter
        indexList = list(filter(lambda x: not len(x["children"]), indexList))
        img_count: int = 0
        for item in indexList:
            bbox: list = item["bbox"]
            index_: str = item["index"]

            image_crop: Any = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

            if item["type"] == "img":
                imgs.append(image_crop)
                new_src: str = f"{img_count}.png"
                part_html: str = new_src
                img_count += 1
            else:
                part_html = self.agent_i2c.infer(image_crop)
                part_html = self.extract_html(part_html)

            target: dict = self.locateByIndex(bboxTree, index_)
            assert target is not None
            target["children"] = [part_html]

        html: str = BboxTree2Html(bboxTree, style=True)

        # optmize by agent
        html2: str = self.agent_optimize.infer(image, html)
        html2 = self.extract_html(html2)
        return html, html2, imgs


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.DEBUG)

    # Set environment variables
    API_KEY = os.getenv("API_KEY") or ""
    ENDPOINT = os.getenv("ENDPOINT") or ""
    API_KEY_GEMINI = os.getenv("API_KEY_GEMINI") or ""
    API_KEY_CLAUDE = os.getenv("API_KEY_CLAUDE") or ""

    os.environ["HTTP_PROXY"] = "http://127.0.0.1:27890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27890"

    # test
    output_dir = "./output"
    image = Image.open("test.png")

    inference = SimpleInference(
        llm_model="gpt-4o",
        API_KEY=API_KEY,
        ENDPOINT=ENDPOINT,
    )
    _, html, imgs = inference.gen(image)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/index.html", "w") as f:
        f.write(html)
    for idx, img in enumerate(imgs):
        img.save(f"{idx}.png")
