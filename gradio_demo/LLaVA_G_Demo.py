
# from .demo_modelpart import InferenceDemo
import gradio as gr
import os
import cv2

import torch
import numpy as np
from llava.eval.LLaVA_G_Eval import Evaluator_MM_Inter
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN

def get_image_name(dir_save="./gradio_demo/tmp_files", prefix="click_img_"):
    import os
    files = os.listdir(dir_save)
    file_orders = [int(file.split(".")[0][len(prefix):]) for file in files if file.endswith(".jpg") and file.startswith(prefix)]
    if len(file_orders) == 0:
        return os.path.join(dir_save, prefix + "0.jpg")
    else:
        return os.path.join(dir_save, prefix + str(max(file_orders) + 1) + ".jpg")
def preprocess_multi_conv(
    sources,
    tokenizer,
    has_image = False
):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    #TODO: multi-turn conversation.
    # conversations = []
    conv.messages = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
    conv_prompt = conv.get_prompt()
    conv_prompt = "ASSISTANT: ".join(conv_prompt.split("ASSISTANT: ")[:-1]) + "ASSISTANT:"
    conv_prompt = conv_prompt.replace("</s>", "")
    conversations = [conv_prompt]
    print("Input Prompt: ", conv_prompt)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
def filter_empty_box_mask(text, boxes_image, masks_image):
    def extract_text(sentence):
        # Use regular expression to find and extract the text and number
        import re
        pattern = r"<g_s>|<g_e> <seg>"
        cleaned_text = re.sub(pattern, '', sentence)
        return cleaned_text
    if len(boxes_image) == 0:
        return text, boxes_image, masks_image
    else:
        sub_texts = text.split(" <seg>")
        sub_texts_filtered = []
        boxes_image_filtered = []
        masks_image_filtered = []
        for box_per_gd, mask_per_gd, text_per_gd in zip(boxes_image, masks_image, sub_texts):
            text_per_gd += " <seg>"
            ind_nonempty_box = torch.where(box_per_gd.abs().sum(dim=1)>0)
            if len(ind_nonempty_box[0]) < box_per_gd.shape[0]:  # empty box encountered
                if len(ind_nonempty_box[0]) == 0:
                    text_per_gd = " " + " ".join(extract_text(text_per_gd).split())
                    sub_texts_filtered.append(text_per_gd)  # box is desperated
                    continue
                else:
                    box_per_gd = box_per_gd[ind_nonempty_box]
                    mask_per_gd = mask_per_gd[ind_nonempty_box]
                    boxes_image_filtered.append(box_per_gd)
                    masks_image_filtered.append(mask_per_gd)
                    sub_texts_filtered.append(text_per_gd)
            else:
                boxes_image_filtered.append(box_per_gd)
                masks_image_filtered.append(mask_per_gd)
                sub_texts_filtered.append(text_per_gd)
        sub_texts_filtered.append(sub_texts[-1])
        text_filtered = "".join(sub_texts_filtered)
        return text_filtered, boxes_image_filtered, masks_image_filtered

class InferenceDemo(object):
    def __init__(self, 
                 model_path, 
                 path_vision_cfg, 
                 path_inter_cfg, 
    ) -> None:
        self.model_backend = Evaluator_MM_Inter(
            model_path=model_path,
            path_vision_model_cfg=path_vision_cfg,
            path_inter_model_cfg =path_inter_cfg,
        )
        self.model_backend.data_mapper.preprocess = preprocess_multi_conv

    def hitory2datadict(self, history, text):
        def filter_valid_conversations(history):
            def delete_color(text):
                import re
                pattern = re.compile(r'<span style=.*?>(.*?)</span>', re.DOTALL)  
                
                # 使用re.sub()函数来删除匹配到的所有<span>标签，只保留括号内的内容  
                clean_text = pattern.sub(r'\1', text)  
                
                return clean_text

            valid_conversations = history[3:]
            valid_conversations = [aa for aa in valid_conversations if not (None in aa)]
            valid_conversations = [[delete_color(aa[0]), delete_color(aa[1])] for aa in valid_conversations]
            return valid_conversations
        valid_conversations = filter_valid_conversations(history) # history[3:]  # the first three conversations are not valid
        dataset_dict = {
            "file_name": history[1][0][0],
            "image_id": 0,
            "question_id": 0,
        }
        dataset_dict['conversations'] = []
        for valid_conv in valid_conversations:
            conv = [
                {
                    "from": "human", 
                    "value": valid_conv[0]
                },
                {
                    "from": "gpt", 
                    "value": valid_conv[1]
                }
            ]
            dataset_dict['conversations'].append([conv, None])
        conv = [
            {
                "from": "human", 
                "value": text
            },
            {
                "from": "gpt", 
                "value": "Placeholder."
            }
        ]
        dataset_dict['conversations'].append([conv, None])
        dataset_dict['conversations'][0][0][0]["value"] = DEFAULT_IMAGE_TOKEN + " " + dataset_dict['conversations'][0][0][0]["value"]

        return dataset_dict
    def inference(self, data_dict):
        # TODO: Implement data_mapper.
        data_dict = self.model_backend.data_mapper(data_dict)[0]
        #
        device = self.model_backend.model.device
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)
        
        response_text, response_boxes, response_mask, mask_inter = self.model_backend.evaluate_sample([data_dict])
        #
        response_text, response_boxes, response_mask = filter_empty_box_mask(response_text, response_boxes, response_mask)
        return response_text, response_boxes, response_mask, mask_inter


def generate_distinct_colors(count):
    import colorsys
    import random
    random.seed(0)
    hues = [i/count for i in range(count)]
    random.shuffle(hues)

    colors = []
    for hue in hues:
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        rgb = tuple(int(val * 255) for val in rgb)
        colors.append(rgb)

    return colors


def add_text(history, text, image, threshold_slider, temporature_slider, interaction_selector):
    # add a text to history stream. and leave the response as None for you to fill in bot.
    def response2stream(response, question):
        return [[question, response]]
    def post_process_text_response(text):
        def find_start_idxes(sentence, word):
            window_size = len(word)
            start_indexes = []
            assert len(sentence) > window_size
            if sentence == window_size:
                return [0]
            for start_index in range(len(sentence) - window_size):
                if sentence[start_index: start_index + window_size] == word:
                    start_indexes.append(start_index)
            return start_indexes
        def add_color_to_text(obj_id, text):
            color = colors[obj_id]
            text = f"<span style='color: rgb{color};'>{text}</span>"
            return text
        def format_sentence(splitted_sentence):
            joint_sentence = " ".join(splitted_sentence)
            return joint_sentence
        def extract_text(sentence):
            # Use regular expression to find and extract the text and number
            import re
            pattern = r"<g_s>|<g_e>"
            cleaned_text = re.sub(pattern, '', sentence)
            return cleaned_text
        
        text_pure = ""
        seg_start_index = find_start_idxes(text, "<seg>")
        if len(seg_start_index) > 0:
            count_obj = 0
            # text = text[:tail_start_index[0]]
            subtexts = text.split(" <seg>")
            for subtext in subtexts:
                if "<g_s>" in subtext:
                    # subtext += "<g_e>"
                    start_idx = find_start_idxes(subtext, "<g_s>")[0]
                    text_pure = format_sentence([text_pure, format_sentence(subtext[:start_idx].split())])
                    text_ = extract_text(subtext[start_idx:])
                    text_pure += add_color_to_text(count_obj, text_)
                    # text_pure = format_sentence([text_pure, format_sentence(text_.split())])
                    count_obj += 1
                else:
                    text_pure = format_sentence([text_pure, format_sentence(subtext.split())])
        else:
            text_pure = text
        return text_pure
    def post_process_gd_response(path_ori_image, gd_results_per_image):
        def unresize_box(box, width, height):
            # ori_size = max(width, height)
            # ratio = ori_size / size
            ratio = min(width, height) / max(width, height)
            if width > height:  # then the height dimension is padded, the y coordinates should be divided by ratio
                box[:, 1] = box[:, 1] / ratio
                box[:, 3] = box[:, 3] / ratio
            elif width < height:  # then the height dimension is padded, the y coordinates should be divided by ratio
                box[:, 0] = box[:, 0] / ratio
                box[:, 2] = box[:, 2] / ratio
            return box
        image = cv2.imread(path_ori_image)
        height, width = image.shape[:2]
        gd_results_per_image = [unresize_box(aa.detach().cpu(), width, height) for aa in gd_results_per_image]
        for gd_id, gd_result in enumerate(gd_results_per_image):
            bboxes = gd_result.cpu().tolist()
            for bbox in bboxes:
                # box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                bbox = [int(bbox[0]*width), int(bbox[1]*height), int(bbox[2]*width), int(bbox[3]*height)]
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[gd_id][::-1], 2)
                # print(f"Obj ID in Image: {gd_id}; Color: {colors[gd_id]}")
        path_save = get_image_name(prefix="grounding_img_")
        cv2.imwrite(path_save, image)
        return (path_save, )
    def post_process_masks(path_ori_image, mask_inter, path_gd_image, masks_gd, loc_inter, inter_type):
        def unresize_mask(mask, width, height):
            # ori_size = max(width, height)
            # ratio = ori_size / size
            import torch.nn.functional as F
            if width > height:  # then the height dimension is padded, the y coordinates should be divided by ratio
                # box[:, 1] = box[:, 1] / ratio
                # box[:, 3] = box[:, 3] / ratio
                mask = F.interpolate((mask[None, ...]).float(), size=[width, width], mode="nearest")[0]
                mask = mask[:, :height]
            elif width < height:  # then the height dimension is padded, the y coordinates should be divided by ratio
                mask = F.interpolate((mask[None, ...]).float(), size=[height, height], mode="nearest")[0]
                mask = mask[:, :, :width]
            return mask
        def unnormalize_inter(mask, loc):
            height, width, _ = mask.shape
            loc_x_mean, loc_y_mean, loc_w, loc_h = loc
            if height >= width:
                loc_x_mean = loc_x_mean / (width/height)
                loc_w = loc_w / (width/height)
            else:
                loc_y_mean = loc_y_mean / (height/width)
                loc_h = loc_h / (height/width)
            return [loc_x_mean-loc_w/2, loc_y_mean-loc_h/2, loc_x_mean+loc_w/2, loc_y_mean+loc_h/2]
        image = cv2.imread(path_ori_image)
        gd_image = cv2.imread(path_gd_image)
        returns = []
        if not (mask_inter is None):
            mask_ = (mask_inter[0][..., None] > 0).float().cpu().numpy()
            mask_ = cv2.resize(mask_, (max(image.shape[0], image.shape[1]), max(image.shape[0], image.shape[1])))
            mask_ = mask_[:image.shape[0], :image.shape[1]]
            mask_ = mask_[..., None] * np.array([155, 155, 155])[None, None, :]
            image = (image * 0.5 + mask_ * 0.5).astype(np.uint8)
            if inter_type.lower() == "box":
                loc_inter_unnormalized = [unnormalize_inter(mask_, loc_inter[0])]
                thickness = max(int(max(image.shape) / 1000 * 5), 1)
                print(thickness)
                cv2.rectangle(image, (int(loc_inter_unnormalized[0][0]*image.shape[1]), int(loc_inter_unnormalized[0][1]*image.shape[0])), (int((loc_inter_unnormalized[0][2])*image.shape[1]), int((loc_inter_unnormalized[0][3])*image.shape[0])), (255, 255, 255), thickness)
            elif inter_type.lower() == "click":
                loc_inter_unnormalized = [unnormalize_inter(mask_, loc_inter[0])]
                thickness = max(int(max(image.shape) / 1000 * 10), 1)
                print(thickness)
                cv2.circle(image, (int(((loc_inter_unnormalized[0][0] + loc_inter_unnormalized[0][2])/2)*image.shape[1]), int(((loc_inter_unnormalized[0][1] + loc_inter_unnormalized[0][3])/2)*image.shape[0])), thickness, (255, 255, 255), thickness=-1)
            path_save = get_image_name(prefix="seg_inter_")
            cv2.imwrite(path_save, image)
            returns.append((path_save, ))
        else:
            returns.append(None)
        if not (masks_gd is None):
            height, width = image.shape[:2]
            masks_gd = [unresize_mask(aa, width, height) for aa in masks_gd]
            colored_mask = torch.zeros((3, height, width), dtype=torch.long)
            for gd_id, gd_mask_result in enumerate(masks_gd):
                gd_mask_result = gd_mask_result.sum(dim=0, keepdim=True)
                colored_mask[:, gd_mask_result[0] > 0.5] = torch.tensor(colors[gd_id][::-1])[:, None]
            gd_image = (gd_image * 0.6 + colored_mask.permute(1,2,0).numpy() * 0.4).astype(np.uint8)
            path_save_gd = get_image_name(prefix="seg_gd_")
            cv2.imwrite(path_save_gd, gd_image)
            returns.append((path_save_gd, ))
        else:
            returns.append(None)
        return returns
    def mask2point(mask, inter_type):
        # mask = cv2.imread(path_mask)
        height, width = mask.shape[:2]
        ys, xs = np.where(mask[..., 0] == 255)
        if inter_type.lower() == "click":
            loc_x = xs.mean()#  / mask.shape[1]
            loc_y = ys.mean()#  / mask.shape[0]
            loc_x = loc_x / width
            loc_y = loc_y / height
            if height >= width:
                loc_x = loc_x * (width/height)
            else:
                loc_y = loc_y * (height/width)
            return torch.tensor([[loc_x, loc_y, 0.006, 0.006]])
        elif inter_type.lower() == "box":
            loc_x_min = xs.min() / width #  / mask.shape[1]
            loc_x_max = xs.max() / width #  / mask.shape[1]
            loc_y_min = ys.min() / height #  / mask.shape[0]
            loc_y_max = ys.max() / height #  / mask.shape[0]
            if height >= width:
                loc_x_min = loc_x_min * (width/height)
                loc_x_max = loc_x_max * (width/height)
            else:
                loc_y_min = loc_y_min * (height/width)
                loc_y_max = loc_y_max * (height/width)
            width = loc_x_max - loc_x_min
            height = loc_y_max - loc_y_min
            return torch.tensor([[(loc_x_min + loc_x_max)/2, (loc_y_min + loc_y_max)/2, width, height]])
    if len(history) < 3:
        response_text = "Please upload an image first."
    else:
        loc_inter = mask2point(image["mask"], interaction_selector)
        is_interactive = torch.isnan(loc_inter[0]).sum() == 0
        if is_interactive:  # do not need this. you can judge if there is interaction through image["mask"]
            input_data_dict = our_chatbot.hitory2datadict(history, text)
            input_data_dict["points"] = loc_inter
            input_data_dict["mode_inter"] = interaction_selector
        else:
            input_data_dict = our_chatbot.hitory2datadict(history, text)
            input_data_dict["points"] = None
            input_data_dict["mode_inter"] = None
        input_data_dict["matching_threshold"] = threshold_slider
        input_data_dict["temporature"] = temporature_slider
        response_text, response_gd, response_mask, mask_inter  = our_chatbot.inference(input_data_dict)
        # response = response[0]
    response_msks  = post_process_masks(history[1][0][0], mask_inter, history[1][0][0], response_mask, loc_inter, interaction_selector)
    if "<seg>" in response_text:
        response_gd = post_process_gd_response(response_msks[1][0], response_gd)
        response_msks[1] = list(response_msks[1])
        response_msks[1][0] = response_gd[0]
        response_msks[1] = tuple(response_msks[1])
    response_text  = post_process_text_response(response_text)
    history += response2stream(response_text, text)
    for response_msk in response_msks:
        if not (response_msk is None):
            history += response2stream(response_msk, None)
    return history, None
def add_image(history, image):
    print("LOG. Add Image Function is called.")
    path_input_img = get_image_name(prefix="tmp_input_img_")
    cv2.imwrite(path_input_img, image["image"][..., ::-1])
    # cv2.imwrite("tmp_input_mask.jpg", image["mask"])  # clear the mask.
    if len(history) > 0:
        history = [(None, "A new image recieved, I will clear the history conversations.")]
    else:
        history = [(None, None)]  # just to align with the above one, to determin where the image_path is.
    history = history + [((path_input_img, ), None)]
    history = history + [(None, "Let't talk about this image!")]
    return history

def add_interaction_click(history, image, interaction_selector):
    print("LOG. Add Interaction Function is called.")
    if interaction_selector.lower() == "box":
        history = history + [(None, "A more detailed box is specified, lets further talk about the region inside the box.")]
    elif interaction_selector.lower() == "click":
        history = history + [(None, "A more detailed click is specified, lets further talk about the region around the click.")]
    # cv2.imwrite("tmp_input_mask.jpg", image["mask"])
    
    mask = image["mask"][..., :3] * np.array([234, 176, 113])
    image_rgb = image["image"][..., ::-1]
    image_clicked = (image_rgb * 0.6 + mask * 0.4).astype(np.uint8)
    path_save = get_image_name(prefix="click_img_")
    cv2.imwrite(path_save, image_clicked)
    return history

def bot(history):
    yield history

def clear_history(history, txt, img):
    return None, None, None
def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question
    
def upvote_one(history):
    print("TODO: Implement upvote_one function.")
    pass
def downvote_one(history):
    print("TODO: Implement downvote_one function.")
    pass
def flag_one(history):
    print("TODO: Implement flag_one function.")
    pass
#? defined here for later renderring.
txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter, or upload an image. Append '(with grounding)' if you want to do grounding.",
    container=False,
)
with gr.Blocks() as demo:
    # Informations
    title_markdown = ("""
        # LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models
        [[Project Page]](todo) [[Paper]](todo) [[Code]](https://github.com/UX-Decoder/LLaVA-Grounding) [[Model]](todo)
    """)
    tos_markdown = ("""
                    
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """)
    models = [
        "LLaVA-Grounding-7B",
    ]
    interactions = [
        "Click",
        "Box"
    ]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gr.Markdown(title_markdown)
    with gr.Row():
        with gr.Column(min_width=300, scale=0.4):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False)
            img = gr.Image(
                type="numpy",
                # label="Image",
                height=220,
                tool="sketch", 
                interactive=True
            )
            img_upload_btn = gr.Button("Submit Image")
            with gr.Row():
                inter_upload_btn = gr.Button("Submit Interaction")
                interaction_selector = gr.Dropdown(
                    choices=interactions,
                    value=interactions[0] if len(interactions) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature")
                threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.1, interactive=True, label="Threshold")
    
            
            gr.Examples(examples=[
                [f"{cur_dir}/examples/meeting.jpg", "Describe the scene in detail. (with grounding)"],
                [f"{cur_dir}/examples/pizza.jpg", "Describe the scene in detail. (with grounding)"],
                
            ], inputs=[img, txt], label="Grounded Description Examples: ")
            gr.Examples(examples=[
                [f"{cur_dir}/examples/cow_motor.jpg", "Where is the object % and what is it doing?"],
                [f"{cur_dir}/examples/dog_sleep.jpg", "What is the object % doing and why?"],
                
            ], inputs=[img, txt], label="Visual Prompt Examples (Please draw clicks or boxes on the woman and dog for the two examples, respectively.): ")

        with gr.Column():
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=598
                # avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
            )  # the chatbot will provide a "history" stream of (input, output) tuples.

            with gr.Row():
                # txt.render()
                with gr.Column(scale=8):
                    txt.render()
                with gr.Column(scale=1, min_width=60):
                    submit_btn = gr.Button(value="Send")
                # btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
            #TODO: Enable these buttons.
            with gr.Row():
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
    txt.submit(add_text, [chatbot, txt, img, threshold, temperature, interaction_selector], [chatbot, txt], queue=False).then(
        bot, 
        chatbot, chatbot, 
        api_name="bot_text_response"
    )
    # .then(
    #     lambda: gr.Textbox(
    #         interactive=True,
    #         placeholder="Enter text and press enter, or upload an image. Append '(with grounding)' if you want do grounding.",
    #     ), 
    #     None, [txt], 
    #     queue=False
    # )
    submit_btn.click(fn=add_text, inputs=[chatbot, txt, img, threshold, temperature, interaction_selector], outputs=[chatbot, txt]).then(
        bot, 
        chatbot, chatbot, 
        api_name="submit_text"
    )
    img_upload_btn.click(fn=add_image, inputs=[chatbot, img], outputs=[chatbot], api_name="upload_image")
    inter_upload_btn.click(fn=add_interaction_click, inputs=[chatbot, img, interaction_selector], outputs=[chatbot], api_name="upload_inter")
    # buttons
    clear_btn.click(fn=clear_history, inputs=[chatbot, txt, img], outputs=[chatbot, txt, img], api_name="clear_all")
    regenerate_btn.click(fn=clear_response, inputs=[chatbot], outputs=[chatbot, txt], api_name="clear_last_response").then(
        add_text, [chatbot, txt, img, threshold, temperature, interaction_selector], [chatbot, txt], queue=False).then(
        bot, 
        chatbot, chatbot, 
        api_name="regenerate_response"
    )
    upvote_btn.click(fn=upvote_one, inputs=[], outputs=[], api_name="upvote_one")
    downvote_btn.click(fn=downvote_one, inputs=[], outputs=[], api_name="downvote_one")
    flag_btn.click(fn=flag_one, inputs=[], outputs=[], api_name="flag_one")

demo.queue()
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default=12124, type=str)
    argparser.add_argument("--model_path", default="", type=str)
    argparser.add_argument("--path_vision_cfg", default="configs/openseed/openseed_swint_lang_joint_2st_v2_data_end_with_interaction.yaml", type=str)
    argparser.add_argument("--path_inter_cfg", default="configs/semsam/idino_swint_1_part_data_llm_ref_feat_all_16_det_pretrainv1.yaml", type=str)
    args = argparser.parse_args()
    model_path = args.model_path
    colors = generate_distinct_colors(20)
    if not os.path.exists("./gradio_demo/tmp_files"):
        os.makedirs("./gradio_demo/tmp_files")
    our_chatbot = InferenceDemo(args.model_path, args.path_vision_cfg, args.path_inter_cfg)
    demo.launch(server_name=args.server_name, server_port=int(args.port))
    