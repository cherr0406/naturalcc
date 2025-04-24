import logging

import torch
from transformers.models.pix2struct import Pix2StructForConditionalGeneration, Pix2StructProcessor

from .agents import AgentI2C, AgentOptimize
from .utils import BboxTree2Html, BboxTree2StyleList, Html2BboxTree, add_special_tokens, move_to_device

logger = logging.getLogger(__name__)


class SimpleInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = Pix2StructProcessor.from_pretrained("xcodemind/webcoder")
        self.processor: Pix2StructProcessor = processor[0] if isinstance(processor, tuple) else processor
        self.tokenizer = self.processor.tokenizer  # type: ignore

        model_bbox = Pix2StructForConditionalGeneration.from_pretrained(
            "xcodemind/webcoder",
            is_encoder_decoder=True,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model_bbox: Pix2StructForConditionalGeneration = model_bbox[0] if isinstance(model_bbox, tuple) else model_bbox
        add_special_tokens(self.model_bbox, self.tokenizer)

        self.agent_i2c = AgentI2C()
        self.agent_optimize = AgentOptimize()

    def infer_bbox(self, image):
        self.model_bbox.eval()
        with torch.no_grad():
            input = "<body bbox=["
            decoder_input_ids = self.tokenizer.encode(input, return_tensors="pt", add_special_tokens=True)[..., :-1]
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

    def locateByIndex(self, bboxTree, index):
        target = bboxTree
        for i in list(filter(lambda x: x, index.split("-"))):
            target = target["children"][int(i)]
        return target

    def extract_html(self, html):
        if "```" in html:
            html = html.split("```")[1]
        if html[:4] == "html":
            html = html[4:]
        html = html.strip()
        return html

    def pruning(self, node, now_depth, max_depth, min_area):
        bbox = node["bbox"]
        area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area < min_area:
            return None
        if now_depth >= max_depth:
            node["children"] = []
        else:
            for idx, cnode in enumerate(node["children"]):
                node["children"][idx] = self.pruning(cnode, now_depth + 1, max_depth, min_area)
            node["children"] = list(filter(lambda x: x, node["children"]))
        return node

    def gen(self, image, max_depth=100, min_area=100):
        imgs = []
        # infer
        prediction_html = self.infer_bbox(image)

        # draw bbox on image
        pBbox = Html2BboxTree(prediction_html, size=image.size)

        # pruning
        self.pruning(pBbox, 1, max_depth, min_area)

        # iter leaf node to gen ctree by agent
        bboxTree = Html2BboxTree(prediction_html, size=image.size)
        indexList = BboxTree2StyleList(bboxTree, skip_leaf=False)
        # only leaf filter
        indexList = list(filter(lambda x: not len(x["children"]), indexList))
        img_count = 0
        for item in indexList:
            bbox = item["bbox"]
            index_ = item["index"]

            image_crop = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

            if item["type"] == "img":
                imgs.append(image_crop)
                new_src = f"{img_count}.png"
                part_html = new_src
                img_count += 1
            else:
                part_html = self.agent_i2c.infer(image_crop)
                part_html = self.extract_html(part_html)

            target = self.locateByIndex(bboxTree, index_)
            assert target is not None
            target["children"] = [part_html]

        html = BboxTree2Html(bboxTree, style=True)

        # optmize by agent
        html2 = self.agent_optimize.infer(image, html)
        html2 = self.extract_html(html2)
        return html, html2, imgs


if __name__ == "__main__":
    import os

    from PIL import Image

    output_dir = "./output"
    image = Image.open("test.png")

    inference = SimpleInference()
    _, html, imgs = inference.gen(image)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/index.html", "w") as f:
        f.write(html)
    for idx, img in enumerate(imgs):
        img.save(f"{idx}.png")
