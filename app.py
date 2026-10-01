import tempfile

import gradio as gr
import spaces
import torch

from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

print("Loading model:", MODEL_ID)

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")


@spaces.GPU(duration=120)
def generate_video(image, prompt, frames, guidance_scale, seed):

    if image is None:
        raise gr.Error("Upload gambar terlebih dahulu.")

    if not prompt.strip():
        raise gr.Error("Masukkan prompt gerakan.")

    image = Image.fromarray(image).convert("RGB")

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(int(seed))

    result = pipe(
        image=image,
        prompt=prompt,
        num_frames=int(frames),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )

    video_frames = result.frames[0]

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ).name

    export_to_video(
        video_frames,
        output_file,
        fps=16,
    )

    return output_file


with gr.Blocks(
    title="ANO VIDEO AI",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🎬 ANO VIDEO AI

        ### Image → Video Generator

        Upload gambar referensi,
        masukkan prompt gerakan,
        lalu generate video AI.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                label="Gambar Referensi",
                type="numpy",
            )

            prompt_input = gr.Textbox(
                label="Prompt Gerakan",
                placeholder=(
                    "The young soccer player takes "
                    "a short run-up and kicks the ball."
                ),
                lines=6,
            )

            frames_input = gr.Slider(
                minimum=41,
                maximum=81,
                value=81,
                step=4,
                label="Frames",
            )

            guidance_input = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=5.0,
                step=0.5,
                label="Guidance Scale",
            )

            seed_input = gr.Number(
                value=42,
                precision=0,
                label="Seed",
            )

            generate_button = gr.Button(
                "🚀 GENERATE VIDEO",
                variant="primary",
            )

        with gr.Column():

            video_output = gr.Video(
                label="Generated Video",
                format="mp4",
            )

    generate_button.click(
        fn=generate_video,
        inputs=[
            image_input,
            prompt_input,
            frames_input,
            guidance_input,
            seed_input,
        ],
        outputs=video_output,
    )


if __name__ == "__main__":
    demo.launch()
