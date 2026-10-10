import os
import re
import json
import math
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

from huggingface_hub import InferenceClient

from moviepy import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip().strip('"').strip("'")

VIDEO_WIDTH = 854
VIDEO_HEIGHT = 480
VIDEO_FPS = 30

OUTPUT_FILENAME = str(
    BASE_DIR / "AI_Tutor_Final.mp4"
)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

LLM_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

TEMP_DIR = Path(
    tempfile.mkdtemp(prefix="ai_tutor_")
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Tutor Video Generator",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 AI Tutor Video Generator")

st.write(
    "Create structured educational videos with AI-generated "
    "lessons, diagrams, illustrations, narration, animations, and an AI teacher avatar."
)

st.caption(
    "AI creates the lesson and teacher avatar. Python creates accurate educational text and diagrams."
)


# ============================================================
# TOKEN CHECK
# ============================================================

if not HF_TOKEN:

    st.error("❌ Hugging Face token was not found.")

    st.code(
        "HF_TOKEN=hf_your_token_here",
        language="text"
    )

    st.stop()

else:

    st.success("🔑 Hugging Face token detected.")


# ============================================================
# USER INPUT
# ============================================================

user_topic = st.text_input(
    "📚 Enter your topic:",
    placeholder=(
        "Examples: Photosynthesis, OSI Model, "
        "Newton's Third Law, Cloud Computing..."
    ),
)

audience = st.selectbox(
    "🎓 Target audience:",
    [
        "Beginner",
        "School Student",
        "College Student",
        "Engineering Student",
    ],
)

duration_preference = st.selectbox(
    "⏱️ Video length:",
    [
        "Short - about 1 minute",
        "Medium - about 2 minutes",
        "Detailed - about 3 minutes",
    ],
)

generate_button = st.button(
    "🚀 Generate AI Tutor Video",
    use_container_width=True,
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

def get_hf_client():

    return InferenceClient(
        provider="auto",
        api_key=HF_TOKEN,
    )


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```\s*$",
        "",
        text,
    )

    try:

        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "The AI did not return valid JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# GENERATE LESSON
# ============================================================

def generate_lesson(
    topic,
    audience,
    duration_preference,
):

    client = get_hf_client()

    if "1 minute" in duration_preference:

        scene_count = 5
        word_target = "180 to 250 words"

    elif "2 minutes" in duration_preference:

        scene_count = 7
        word_target = "300 to 420 words"

    else:

        scene_count = 9
        word_target = "500 to 650 words"

    system_prompt = f"""
You are an expert teacher, curriculum designer and
educational video director.

Create an accurate educational lesson for:

Topic: {topic}

Audience: {audience}

The lesson will become a narrated educational video.

The complete narration should contain approximately:
{word_target}

Create exactly {scene_count} scenes.

TEACHING RULES:

1. Start by clearly defining the topic.
2. Explain what it actually is.
3. Explain important components.
4. Explain how it works.
5. Explain the process step-by-step where applicable.
6. Give concrete examples.
7. Include formulas/equations where relevant.
8. Explain practical significance only after explaining
   the actual concept.
9. Never use vague filler.
10. Do not repeat ideas.
11. Do not assume prior knowledge.
12. Use technically accurate information.
13. Make the narration sound like a real teacher.

VISUAL RULES:

Each scene must have one of these visual types:

"illustration"
"diagram"
"process"
"equation"
"comparison"

Use "illustration" when a realistic or conceptual picture
is useful.

Use "diagram" when relationships between components
need to be shown.

Use "process" when something happens in sequential steps.

Use "equation" when a formula or mathematical relationship
is central.

Use "comparison" when two or more concepts need to be
compared.

VERY IMPORTANT:

FLUX will generate illustrations.

FLUX MUST NOT generate text.

Therefore, visual_prompt must describe ONLY the visual
objects and their arrangement.

Do NOT ask the image model to generate:
- words
- labels
- equations
- letters
- numbers
- captions
- titles

All important text will be added later by Python.

Return ONLY valid JSON.

Use exactly:

{{
    "title": "lesson title",
    "introduction": "short introduction",
    "scenes": [
        {{
            "scene_number": 1,
            "title": "scene title",
            "narration": "actual teacher narration",
            "key_point": "one short educational sentence",
            "visual_type": "illustration",
            "visual_prompt": "visual description without text",
            "labels": [
                "important label 1",
                "important label 2"
            ],
            "steps": [
                "step 1",
                "step 2",
                "step 3"
            ],
            "equation": ""
        }}
    ]
}}

For fields that are not needed, return an empty list or
empty string.
"""

    user_prompt = f"""
Create a complete educational video lesson about:

{topic}

Target audience:
{audience}

Video duration:
{duration_preference}

The lesson MUST genuinely teach the topic.

For example, if the topic is Photosynthesis, explain:
- definition
- inputs
- chlorophyll
- chloroplast
- process
- products
- equation
- importance

Do not simply say that it is important.

Adapt everything to the actual topic.
"""

    last_error = None

    for model in LLM_MODELS:

        try:

            status_message = st.empty()

            status_message.info(
                f"🧠 Creating lesson using {model}..."
            )

            response = client.chat_completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=4500,
                temperature=0.25,
            )

            content = response.choices[0].message.content

            if not content:

                raise RuntimeError(
                    "The AI returned an empty response."
                )

            lesson = extract_json(content)

            if "scenes" not in lesson:

                raise ValueError(
                    "No scenes were returned."
                )

            if not lesson["scenes"]:

                raise ValueError(
                    "The AI returned zero scenes."
                )

            for scene in lesson["scenes"]:

                required = [
                    "scene_number",
                    "title",
                    "narration",
                    "key_point",
                    "visual_type",
                    "visual_prompt",
                ]

                for field in required:

                    if field not in scene:

                        raise ValueError(
                            f"Scene missing: {field}"
                        )

                if "labels" not in scene:

                    scene["labels"] = []

                if "steps" not in scene:

                    scene["steps"] = []

                if "equation" not in scene:

                    scene["equation"] = ""

            status_message.success(
                f"✅ Lesson created using {model}"
            )

            return lesson

        except Exception as e:

            last_error = e

            st.warning(
                f"⚠️ Model unavailable: {model}"
            )

            st.caption(
                str(e)
            )

    raise RuntimeError(
        "All configured language models failed.\n\n"
        f"Last error: {last_error}"
    )


# ============================================================
# FONTS
# ============================================================

def get_font(
    size,
    bold=False,
):

    if bold:

        fonts = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ]

    else:

        fonts = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]

    for font in fonts:

        if os.path.exists(font):

            return ImageFont.truetype(
                font,
                size,
            )

    return ImageFont.load_default()


# ============================================================
# TEXT WRAPPING
# ============================================================

def wrap_text(
    draw,
    text,
    font,
    max_width,
):

    words = text.split()

    lines = []

    current = ""

    for word in words:

        test = (
            current + " " + word
        ).strip()

        bbox = draw.textbbox(
            (0, 0),
            test,
            font=font,
        )

        width = bbox[2] - bbox[0]

        if width <= max_width:

            current = test

        else:

            if current:

                lines.append(current)

            current = word

    if current:

        lines.append(current)

    return lines


# ============================================================
# BASE SLIDE
# ============================================================

def create_base_slide():

    return Image.new(
        "RGB",
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        (245, 248, 252),
    )


# ============================================================
# TITLE + KEY POINT + TEACHER AVATAR
# ============================================================

def add_header_footer(
    image,
    title,
    key_point,
    scene_number,
    teacher_avatar=None,
):

    draw = ImageDraw.Draw(
        image,
        "RGBA",
    )

    # Header
    draw.rectangle(
        (
            0,
            0,
            VIDEO_WIDTH,
            70,
        ),
        fill=(15, 23, 42, 235),
    )

    title_font = get_font(
        28,
        bold=True,
    )

    draw.text(
        (
            25,
            17,
        ),
        f"{scene_number}. {title}",
        fill=(255, 255, 255),
        font=title_font,
    )

    # Footer
    draw.rectangle(
        (
            0,
            VIDEO_HEIGHT - 80,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        fill=(15, 23, 42, 235),
    )

    key_font = get_font(
        20,
        bold=False,
    )

    lines = wrap_text(
        draw,
        key_point,
        key_font,
        VIDEO_WIDTH - 50,
    )

    y = VIDEO_HEIGHT - 68

    for line in lines[:2]:

        draw.text(
            (
                25,
                y,
            ),
            line,
            fill=(255, 255, 255),
            font=key_font,
        )

        y += 26

    # Paste Teacher Avatar in Bottom-Right Corner (Above Footer)
    if teacher_avatar:
        avatar_x = VIDEO_WIDTH - 145
        avatar_y = VIDEO_HEIGHT - 215
        image.paste(teacher_avatar, (avatar_x, avatar_y))

        # Border for the avatar box
        draw.rounded_rectangle(
            (avatar_x, avatar_y, avatar_x + 120, avatar_y + 120),
            radius=12,
            outline=(30, 64, 175),
            width=2,
        )

    return image


# ============================================================
# DRAW ARROW
# ============================================================

def draw_arrow(
    draw,
    start,
    end,
    width=5,
):

    draw.line(
        (
            start,
            end,
        ),
        fill=(30, 64, 175),
        width=width,
    )

    x1, y1 = start
    x2, y2 = end

    angle = math.atan2(
        y2 - y1,
        x2 - x1,
    )

    arrow_length = 14

    p1 = (
        x2 - arrow_length * math.cos(angle - math.pi / 6),
        y2 - arrow_length * math.sin(angle - math.pi / 6),
    )

    p2 = (
        x2 - arrow_length * math.cos(angle + math.pi / 6),
        y2 - arrow_length * math.sin(angle + math.pi / 6),
    )

    draw.polygon(
        [
            end,
            p1,
            p2,
        ],
        fill=(30, 64, 175),
    )


# ============================================================
# DIAGRAM VISUAL
# ============================================================

def create_diagram(
    scene,
    teacher_avatar=None,
):

    image = create_base_slide()

    draw = ImageDraw.Draw(
        image
    )

    title = scene["title"]

    labels = scene.get(
        "labels",
        [],
    )

    center_x = VIDEO_WIDTH // 2 - 40
    center_y = 250

    box_font = get_font(
        22,
        bold=True,
    )

    small_font = get_font(
        17,
        bold=False,
    )

    box_width = 250
    box_height = 80

    draw.rounded_rectangle(
        (
            center_x - box_width // 2,
            center_y - box_height // 2,
            center_x + box_width // 2,
            center_y + box_height // 2,
        ),
        radius=18,
        fill=(220, 235, 255),
        outline=(30, 64, 175),
        width=3,
    )

    center_text = title

    bbox = draw.textbbox(
        (0, 0),
        center_text,
        font=box_font,
    )

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        (
            center_x - tw / 2,
            center_y - th / 2,
        ),
        center_text,
        fill=(15, 23, 42),
        font=box_font,
    )

    positions = [
        (130, 150),
        (600, 150),
        (130, 330),
        (600, 330),
        (365, 125),
        (365, 375),
    ]

    for index, label in enumerate(labels[:6]):

        x, y = positions[index]

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=small_font,
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        padding_x = 15
        padding_y = 10

        draw.rounded_rectangle(
            (
                x - tw / 2 - padding_x,
                y - th / 2 - padding_y,
                x + tw / 2 + padding_x,
                y + th / 2 + padding_y,
            ),
            radius=12,
            fill=(255, 255, 255),
            outline=(100, 116, 139),
            width=2,
        )

        draw.text(
            (
                x - tw / 2,
                y - th / 2,
            ),
            label,
            fill=(15, 23, 42),
            font=small_font,
        )

        draw_arrow(
            draw,
            (x, y),
            (center_x, center_y),
            width=3,
        )

    return add_header_footer(
        image,
        scene["title"],
        scene["key_point"],
        scene["scene_number"],
        teacher_avatar=teacher_avatar,
    )


# ============================================================
# PROCESS VISUAL
# ============================================================

def create_process(
    scene,
    teacher_avatar=None,
):

    image = create_base_slide()

    draw = ImageDraw.Draw(
        image
    )

    steps = scene.get(
        "steps",
        [],
    )

    if not steps:

        steps = [
            scene["key_point"]
        ]

    small_font = get_font(
        16,
        bold=False,
    )

    usable_steps = steps[:5]

    start_y = 120

    box_width = 660
    box_height = 48

    spacing = 12

    for i, step in enumerate(
        usable_steps
    ):

        y = (
            start_y
            + i * (box_height + spacing)
        )

        draw.rounded_rectangle(
            (
                45,
                y,
                45 + box_width,
                y + box_height,
            ),
            radius=12,
            fill=(235, 242, 255),
            outline=(30, 64, 175),
            width=2,
        )

        draw.ellipse(
            (
                58,
                y + 7,
                92,
                y + 41,
            ),
            fill=(30, 64, 175),
        )

        draw.text(
            (
                69,
                y + 12,
            ),
            str(i + 1),
            fill=(255, 255, 255),
            font=get_font(
                15,
                bold=True,
            ),
        )

        lines = wrap_text(
            draw,
            step,
            small_font,
            570,
        )

        draw.text(
            (
                110,
                y + 13,
            ),
            lines[0][:80],
            fill=(15, 23, 42),
            font=small_font,
        )

        if i < len(usable_steps) - 1:

            draw_arrow(
                draw,
                (
                    375,
                    y + box_height,
                ),
                (
                    375,
                    y + box_height + spacing,
                ),
                width=3,
            )

    return add_header_footer(
        image,
        scene["title"],
        scene["key_point"],
        scene["scene_number"],
        teacher_avatar=teacher_avatar,
    )


# ============================================================
# EQUATION VISUAL
# ============================================================

def create_equation_scene(
    scene,
    teacher_avatar=None,
):

    image = create_base_slide()

    draw = ImageDraw.Draw(
        image
    )

    equation = scene.get(
        "equation",
        "",
    )

    if not equation:

        equation = scene["key_point"]

    equation_font = get_font(
        34,
        bold=True,
    )

    bbox = draw.textbbox(
        (0, 0),
        equation,
        font=equation_font,
    )

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.rounded_rectangle(
        (
            60,
            170,
            VIDEO_WIDTH - 180,
            310,
        ),
        radius=20,
        fill=(255, 255, 255),
        outline=(30, 64, 175),
        width=3,
    )

    draw.text(
        (
            (VIDEO_WIDTH - 120) / 2 - tw / 2,
            235 - th / 2,
        ),
        equation,
        fill=(15, 23, 42),
        font=equation_font,
    )

    return add_header_footer(
        image,
        scene["title"],
        scene["key_point"],
        scene["scene_number"],
        teacher_avatar=teacher_avatar,
    )


# ============================================================
# FLUX IMAGE
# ============================================================

def generate_flux_image(
    prompt,
    scene_number,
):

    client = get_hf_client()

    safe_prompt = f"""
Create a clean professional educational illustration.

{prompt}

The image is going to have labels added digitally later.

DO NOT generate:
- text
- words
- letters
- numbers
- captions
- titles
- equations
- labels
- watermarks

Use visual symbols, objects and diagrams only.

Style:
clean educational textbook illustration,
professional instructional design,
simple composition,
high visual clarity,
appropriate for a college educational video.
"""

    try:

        image = client.text_to_image(
            prompt=safe_prompt,
            model=IMAGE_MODEL,
        )

        return image.convert(
            "RGB"
        )

    except Exception as e:

        st.warning(
            f"⚠️ FLUX failed for scene "
            f"{scene_number}: {e}"
        )

        return Image.new(
            "RGB",
            (
                VIDEO_WIDTH,
                VIDEO_HEIGHT,
            ),
            (225, 232, 240),
        )


# ============================================================
# GENERATE TEACHER AVATAR
# ============================================================

def generate_teacher_avatar():
    try:
        avatar_prompt = (
            "Friendly college professor character avatar, professional portrait headshot, "
            "warm smiling expression, wearing professional attire, clean neutral background, "
            "vector illustration style, high quality instructional design"
        )
        avatar = generate_flux_image(avatar_prompt, 0)
        return avatar.resize((120, 120), Image.Resampling.LANCZOS)
    except Exception:
        fallback = Image.new("RGB", (120, 120), (220, 235, 255))
        return fallback


# ============================================================
# ILLUSTRATION SCENE
# ============================================================

def create_illustration_scene(
    scene,
    teacher_avatar=None,
):

    image = generate_flux_image(
        scene["visual_prompt"],
        scene["scene_number"],
    )

    image = image.resize(
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        Image.Resampling.LANCZOS,
    )

    return add_header_footer(
        image,
        scene["title"],
        scene["key_point"],
        scene["scene_number"],
        teacher_avatar=teacher_avatar,
    )


# ============================================================
# SCENE CREATION
# ============================================================

def create_scene_image(
    scene,
    teacher_avatar=None,
):

    visual_type = (
        scene["visual_type"]
        .lower()
        .strip()
    )

    if visual_type == "diagram":

        return create_diagram(
            scene,
            teacher_avatar=teacher_avatar,
        )

    elif visual_type == "process":

        return create_process(
            scene,
            teacher_avatar=teacher_avatar,
        )

    elif visual_type == "equation":

        return create_equation_scene(
            scene,
            teacher_avatar=teacher_avatar,
        )

    else:

        return create_illustration_scene(
            scene,
            teacher_avatar=teacher_avatar,
        )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_scene_image(
    image,
    scene_number,
):

    path = TEMP_DIR / (
        f"scene_{scene_number}.png"
    )

    image.save(
        path,
        format="PNG",
    )

    return str(path)


# ============================================================
# TEXT TO SPEECH
# ============================================================

def generate_voice(
    narration,
    scene_number,
):

    path = TEMP_DIR / (
        f"audio_{scene_number}.mp3"
    )

    tts = gTTS(
        text=narration,
        lang="en",
        tld="com",
        slow=False,
    )

    tts.save(
        str(path)
    )

    return str(path)


# ============================================================
# CREATE VIDEO CLIP
# ============================================================

def create_video_scene(
    image_path,
    audio_path,
):

    audio = AudioFileClip(
        audio_path
    )

    duration = audio.duration

    clip = ImageClip(
        image_path
    )

    clip = clip.with_duration(
        duration
    )

    clip = clip.resized(
        new_size=(
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        )
    )

    clip = clip.resized(
        lambda t: 1.0 + min(
            0.015,
            0.003 * t,
        )
    )

    clip = clip.cropped(
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        x_center=clip.w / 2,
        y_center=clip.h / 2,
    )

    clip = clip.with_audio(
        audio
    )

    return clip, audio


# ============================================================
# CREATE COMPLETE VIDEO
# ============================================================

def process_video_pipeline(
    lesson,
    progress,
    status,
):

    scenes = lesson["scenes"]

    video_clips = []

    audio_clips = []

    final_movie = None

    total = len(
        scenes
    )

    try:
        status.text("👩‍🏫 Generating AI teacher avatar...")
        teacher_avatar = generate_teacher_avatar()

        for index, scene in enumerate(
            scenes
        ):

            scene_number = index + 1

            progress.progress(
                int(
                    index / total * 90
                )
            )

            status.text(
                f"🎙️ Scene {scene_number}/{total}: "
                f"Creating narration..."
            )

            audio_path = generate_voice(
                scene["narration"],
                scene_number,
            )

            status.text(
                f"🎨 Scene {scene_number}/{total}: "
                f"Creating {scene['visual_type']} with AI teacher..."
            )

            scene_image = create_scene_image(
                scene,
                teacher_avatar=teacher_avatar,
            )

            image_path = save_scene_image(
                scene_image,
                scene_number,
            )

            status.text(
                f"🎬 Scene {scene_number}/{total}: "
                f"Animating..."
            )

            clip, audio = create_video_scene(
                image_path,
                audio_path,
            )

            video_clips.append(
                clip
            )

            audio_clips.append(
                audio
            )

        progress.progress(
            92
        )

        status.text(
            "🎞️ Combining all educational scenes..."
        )

        final_movie = concatenate_videoclips(
            video_clips,
            method="compose",
        )

        status.text(
            "🚀 Rendering final MP4..."
        )

        final_movie.write_videofile(
            OUTPUT_FILENAME,
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
        )

        progress.progress(
            100
        )

        status.success(
            "🎉 Video generation completed!"
        )

        return OUTPUT_FILENAME

    finally:

        if final_movie:

            try:
                final_movie.close()

            except Exception:
                pass

        for clip in video_clips:

            try:
                clip.close()

            except Exception:
                pass

        for audio in audio_clips:

            try:
                audio.close()

            except Exception:
                pass


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    if not TEMP_DIR.exists():

        return

    for file in TEMP_DIR.iterdir():

        try:

            if file.is_file():

                file.unlink()

        except Exception:
            pass

    try:

        TEMP_DIR.rmdir()

    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

if generate_button:

    if not user_topic.strip():

        st.warning(
            "⚠️ Please enter a topic first."
        )

        st.stop()

    progress = st.progress(0)

    status = st.empty()

    try:

        status.info(
            "🧠 Understanding the topic and planning the lesson..."
        )

        lesson = generate_lesson(
            user_topic.strip(),
            audience,
            duration_preference,
        )

        with st.expander(
            "📖 Preview AI Lesson"
        ):

            st.subheader(
                lesson.get(
                    "title",
                    user_topic,
                )
            )

            st.write(
                lesson.get(
                    "introduction",
                    "",
                )
            )

            for scene in lesson["scenes"]:

                st.markdown(
                    f"### Scene {scene['scene_number']}: "
                    f"{scene['title']}"
                )

                st.write(
                    scene["narration"]
                )

                st.caption(
                    f"Visual type: "
                    f"{scene['visual_type']}"
                )

                st.caption(
                    f"Key point: "
                    f"{scene['key_point']}"
                )

        output_file = process_video_pipeline(
            lesson,
            progress,
            status,
        )

        with open(
            output_file,
            "rb",
        ) as video_file:

            video_bytes = video_file.read()

        st.success(
            "🎉 Your educational video is ready!"
        )

        st.video(
            video_bytes,
            format="video/mp4",
        )

        st.download_button(
            "⬇️ Download AI Tutor Video",
            data=video_bytes,
            file_name="AI_Tutor_Video.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            "❌ Video generation failed."
        )

        st.exception(
            e
        )

    finally:

        cleanup()