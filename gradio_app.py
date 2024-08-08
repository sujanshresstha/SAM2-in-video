import os
import torch
import numpy as np
import cv2
import gradio as gr
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

# Enable automatic mixed precision for CUDA to improve performance and memory efficiency
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Check if the GPU supports TensorFloat-32 (TF32) precision for faster computations
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor = None

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to SAM2-in-video!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

def initialize_predictor(checkpoint):
    global predictor
    if checkpoint == "tiny":
        sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
    elif checkpoint == "small":
        sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
    else:
        raise ValueError("Invalid checkpoint")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def extract_first_frame(video_path):
    if not video_path:
        gr.Warning("No input video")
        return None, None, [], None, None

    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    video.release()

    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_dir = tempfile.mkdtemp()
        frame_path = os.path.join(temp_dir, "00000.jpg") 
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return frame_rgb, frame_rgb.copy(), [], None, temp_dir
    else:
        return None, None, [], None, None

def add_point_and_segment(image, original_image, points, point_type, mask, temp_dir, evt: gr.SelectData):
    if image is None or original_image is None or temp_dir is None:
        return image, original_image, points, mask, temp_dir

    x, y = evt.index[0], evt.index[1]
    points.append((x, y, point_type))

    # Prepare points for SAM2
    np_points = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    labels = np.array([1 if p[2] == "positive" else 0 for p in points], dtype=np.int32)

    # Initialize SAM2 state
    inference_state = predictor.init_state(video_path=temp_dir)
    predictor.reset_state(inference_state)

    # Perform segmentation
    _, _, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np_points,
        labels=labels,
    )

    # Update mask
    mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    # Create a copy of the original image to draw on
    image_with_points_and_mask = original_image.copy()

    # Apply mask
    if point_type == "positive":
        image_with_points_and_mask[mask] = image_with_points_and_mask[mask] * 0.7 + np.array([0, 255, 0]) * 0.3
    else:
        mask = ~mask  # Invert mask for negative points
        image_with_points_and_mask[mask] = image_with_points_and_mask[mask] * 0.7 + np.array([255, 0, 0]) * 0.3

    # Draw all points on the image
    for px, py, pt in points:
        color = (0, 255, 0) if pt == "positive" else (255, 0, 0)
        cv2.circle(image_with_points_and_mask, (px, py), 5, color, -1)

    return image_with_points_and_mask, original_image, points, mask, temp_dir

def clear_points(image, original_image, points, mask, temp_dir):
    if original_image is None:
        return None, None, [], None, temp_dir
    
    # Return the original image without points or mask
    return original_image.copy(), original_image, [], None, temp_dir

def segment_video(video_path, points, mask, checkpoint):
    if not video_path or not points:
        return None, None

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    frame_dir = os.path.join(video_dir, f"{video_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{frame_count:05d}.jpg"), frame)
        frame_count += 1
    video.release()

    frame_names = sorted([p for p in os.listdir(frame_dir) if p.endswith('.jpg')])

    #  Prepare points
    np_points = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    labels = np.array([1 if p[2] == "positive" else 0 for p in points], dtype=np.int32)

    inference_state = predictor.init_state(video_path=frame_dir)
    predictor.reset_state(inference_state)

    # Initaial annotation
    ann_frame_idx = 0
    ann_obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=np_points,
        labels=labels,
    )

    output_video_path = os.path.join(video_dir, "output_video.mp4")
    extracted_video_path = os.path.join(video_dir, "extracted_video.mp4")
    frame_rate = 30

    first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    height, width = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    extracted_writer = cv2.VideoWriter(extracted_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    cmap = plt.get_cmap("tab10")

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame = cv2.imread(os.path.join(frame_dir, frame_names[out_frame_idx]))
        extracted_frame = np.zeros_like(frame)

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            # For segmented object only
            extracted_frame[mask] = frame[mask]
            color = np.array(cmap(i % 10)[:3]) * 255
            # For output video with overlay
            frame[mask] = frame[mask] * 0.5 + color * 0.5

        video_writer.write(frame)
        extracted_writer.write(extracted_frame)

    video_writer.release()
    extracted_writer.release()

    return output_video_path, extracted_video_path

def sam2_in_video():
    app = gr.Blocks(js=js)

    with app:
        gr.Markdown("Video Segmentation with SAM2")
        with gr.Column():
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                image_output = gr.Image(label="First Frame", interactive=True)

            with gr.Row():
                extract_button = gr.Button("Extract First Frame")
                with gr.Row():
                    point_type = gr.Radio(["positive", "negative"], label="Point Type", value="positive")
                    clear_points_button = gr.Button("Clear Points")

        points = gr.State([])
        original_image = gr.State(None)
        mask = gr.State(None)
        temp_dir = gr.State(None)

        with gr.Row():
            checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus", "large"], value="tiny")
            segment_button = gr.Button("Segment Video")

        with gr.Row():
            video_output = gr.Video(label="Segmented Video")
            extracted_video_output = gr.Video(label="Extracted Segmented Object")

        extract_button.click(
            extract_first_frame,
            inputs=video_input,
            outputs=[image_output, original_image, points, mask, temp_dir],
            concurrency_limit=1
        )

        image_output.select(
            add_point_and_segment,
            inputs=[image_output, original_image, points, point_type, mask, temp_dir],
            outputs=[image_output, original_image, points, mask, temp_dir],
            concurrency_limit=1
        )

        clear_points_button.click(
            clear_points,
            inputs=[image_output, original_image, points, mask, temp_dir],
            outputs=[image_output, original_image, points, mask, temp_dir],
            concurrency_limit=1
        )

        segment_button.click(
            segment_video,
            inputs=[video_input, points, mask, checkpoint],
            outputs=[video_output, extracted_video_output],
            concurrency_limit=1
        )
        with gr.Tab(label='Video example'):
          gr.Examples(
              examples=["./assets/test.mp4"],
              inputs=[video_input],
          )

    app.launch(debug=True, share=True, max_threads=1)

if __name__ == "__main__":
    initialize_predictor("tiny")