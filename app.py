# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os
import argparse
import config
import gradio as gr

from model import ImgCaption_Model
from dataset import Vocabulary
from timeit import default_timer as timer
from utils import load_check_point_to_use

# Initialize parameters and parse the parameters
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', '-d', type=str, default=config.device, help='device to training')
    parse.add_argument('--save-path', '-s', type=str, default=config.save_path, help='number of batch size')
    parse.add_argument('--transform', default=config.transform, help='Compose transform of images')
    parse.add_argument('--embed-size', default=config.embed_size, help='Size of embedding')
    parse.add_argument('--hidden-size', default=config.hidden_size, help='Number of hidden nodes in RNN')
    parse.add_argument('--num-layer', default=config.num_layer, help='Number of layers lstm stack')
    parse.add_argument('--num-workers', default=config.num_workers, help='Number of core CPU use to load data')
    args = parse.parse_args()
    return args

# Load vocab file
vocab = Vocabulary()
vocab.read_vocab()

# Load arguments
args = get_args()

# Load model
model = ImgCaption_Model(args.embed_size, args.hidden_size, len(vocab), args.num_layer).to(args.device)

# Load saved weights
load_check_point_to_use(args.save_path + '/best.pt', model, args.device)

def caption(img):
    """Transforms, describe about image and returns caption and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image
    img = args.transform(img)

    # Put model into evaluation mode and describe image
    model.eval()
    prompt = " ".join(model.caption_image(img.unsqueeze(0), vocab))

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the caption and prediction time
    return prompt, pred_time


# Create title, description and article strings
def main():
    title = "Image Captioning 🖼➡️🆎"
    description = "A model describe about the picture"
    article = "Created on [GITHUB](https://github.com/vvduc1803/Image-Captioning)."

    # Create examples list from "examples/" directory
    example_list = [["examples/" + example] for example in os.listdir("examples")]

    # Create the Gradio demo
    demo = gr.Interface(fn=caption,  # mapping function from input to output
                        inputs=gr.Image(type="pil"),  # what are the inputs?
                        outputs=[gr.Textbox(label="Caption"), # what are the outputs?
                                 gr.Number(label="Prediction time (s)")],
                        # our fn has two outputs, therefore we have two outputs
                        # Create examples list from "examples/" directory
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

    # Launch the demo!
    demo.launch(server_name="127.0.0.1", server_port=1234, share=True)

if __name__ == '__main__':
    main()