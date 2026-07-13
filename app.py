"""
Multimodal Image Generation Studio
-----------------------------------
A Streamlit app that translates natural language text prompts into
digital artwork using Pollinations.ai's free text-to-image API
(no API key required), with a Groq-powered AI prompt enhancer.

Architecture follows the project blueprint:
1. Prompt Payload Formulation   -> aspect ratio -> exact pixel resolution
2. Network API Gateway          -> split-timeout policy (connect / read)
3. Security & Moderation Gates  -> graceful handling of blocked content
4. Transport Protocol           -> memory-safe chunked binary streaming
5. Integrity Verification       -> forced pixel-level decode (Pillow .load())
6. Automated Quality feedback   -> retry with exponential backoff + jitter
"""

import io
import os
import random
import time
import urllib.parse
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BASE_URL = "https://image.pollinations.ai/prompt/"

# Aspect ratio -> exact pixel payload (kept within safe generation limits)
ASPECT_RATIOS = {
    "1:1 Square (1024x1024) - Avatars, product grids": (1024, 1024),
    "16:9 Landscape (1344x768) - Web banners, presentations": (1344, 768),
    "9:16 Vertical (768x1344) - Mobile reels, wallpapers": (768, 1344),
    "4:3 Standard (1024x768) - Classic prints": (1024, 768),
    "3:4 Portrait (768x1024) - Posters": (768, 1024),
}

STYLE_PRESETS = {
    "None": "",
    "Cyberpunk": ", cyberpunk style, neon lights, futuristic",
    "Minimalism": ", minimalist style, clean lines, simple composition",
    "Watercolor": ", watercolor painting style, soft brush strokes",
    "Photorealistic": ", photorealistic, highly detailed, 8k",
    "Anime": ", anime style, vibrant colors, cel shading",
    "Oil Painting": ", oil painting style, textured brushwork, classical art",
}

MODELS = ["flux", "turbo", "flux-realism", "flux-anime", "flux-3d"]

GROQ_MODEL = "llama-3.3-70b-versatile"
ENHANCER_SYSTEM_PROMPT = (
    "You are a prompt engineer for text-to-image diffusion models. "
    "Rewrite the user's image description into a single, vivid, detailed prompt "
    "optimized for image generation: add composition, lighting, mood, and style "
    "details where helpful, while strictly preserving the user's original subject "
    "and intent. Do not add explanations, quotes, or preamble - output ONLY the "
    "enhanced prompt text, max 60 words."
)

OUTPUT_DIR = "generated_images"
CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 90
MAX_RETRIES = 4
CHUNK_SIZE = 65536  # 64 KiB, memory-safe streaming chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Core pipeline functions
# --------------------------------------------------------------------------


def build_payload_url(prompt: str, width: int, height: int, model: str, seed: int) -> str:
    """Phase 1: Prompt Payload Formulation - serialize params into the request URL."""
    encoded_prompt = urllib.parse.quote(prompt)
    params = f"?width={width}&height={height}&model={model}&seed={seed}&nologo=true"
    return f"{BASE_URL}{encoded_prompt}{params}"


def generate_image_with_retry(url: str, max_retries: int = MAX_RETRIES):
    """
    Phase 2-4: Network gateway + security gates + transport protocol.

    Uses a split connect/read timeout, exponential backoff with jitter,
    and memory-safe chunked streaming straight to bytes.

    Returns (success: bool, data_or_error: bytes | str, failure_type: str | None)
    failure_type in {"network", "inference", "moderation", None}
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                stream=True,
            )

            # Post-generation security gate (moderation / content policy)
            if response.status_code in (403, 451):
                return False, "Request blocked by content moderation policy.", "moderation"

            if response.status_code == 429:
                # Rate limited - back off and retry
                wait = _backoff_delay(attempt)
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                # Server-side / inference failure - retry
                wait = _backoff_delay(attempt)
                time.sleep(wait)
                continue

            response.raise_for_status()

            # Memory-safe chunked read into a buffer
            buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    buffer.write(chunk)

            return True, buffer.getvalue(), None

        except requests.exceptions.ConnectTimeout:
            # Network failure -> fail fast, minimal retry
            if attempt == max_retries:
                return False, "Could not connect to the image generation server.", "network"
            time.sleep(_backoff_delay(attempt))

        except requests.exceptions.ReadTimeout:
            # Inference failure -> the server is slow, worth retrying longer
            if attempt == max_retries:
                return False, "The server took too long to generate the image.", "inference"
            time.sleep(_backoff_delay(attempt))

        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                return False, f"Network error: {e}", "network"
            time.sleep(_backoff_delay(attempt))

    return False, "Exhausted all retry attempts.", "network"


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter to avoid hammering the server."""
    base = min(2 ** attempt, 20)
    jitter = random.uniform(0, 1)
    return base + jitter


def verify_image_integrity(image_bytes: bytes):
    """
    Phase 5: Integrity Verification.

    Forces a full pixel-level decode (not just header check) to catch
    truncated / corrupted streams that a shallow verify() would miss.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # forces full decode, raises OSError if truncated
        return True, img
    except (UnidentifiedImageError, OSError) as e:
        return False, str(e)


def save_image(image_bytes: bytes, prompt: str) -> str:
    """Phase 4 (output): write verified bytes safely to local storage."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
    filename = f"{timestamp}_{safe_prompt}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


def enhance_prompt(raw_prompt: str, api_key: str):
    """Use Groq (Llama 3.3) to expand a short prompt into a richer image description."""
    if Groq is None:
        return False, "Groq SDK not installed. Run: pip install groq"
    if not api_key:
        return False, "No Groq API key provided."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": ENHANCER_SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        enhanced = response.choices[0].message.content.strip()
        return True, enhanced
    except Exception as e:
        return False, f"Enhancer error: {e}"


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Multimodal Image Generation Studio", page_icon="🎨", layout="wide")

st.title("🎨 Multimodal Image Generation Studio")
st.markdown(
    "Translate natural language prompts into digital artwork using a free, "
    "no-API-key text-to-image engine — with production-grade error handling, "
    "retries, and binary integrity verification."
)

with st.sidebar:
    st.header("⚙️ Generation Parameters")

    aspect_choice = st.selectbox("Aspect Ratio", list(ASPECT_RATIOS.keys()))
    width, height = ASPECT_RATIOS[aspect_choice]
    st.caption(f"Resolution: {width} x {height} px  |  Pixel volume: {width*height:,}")

    style_choice = st.selectbox("Style Preset", list(STYLE_PRESETS.keys()))
    model_choice = st.selectbox("Model Engine", MODELS, help="flux = general purpose, turbo = fastest, flux-realism/anime/3d = specialized styles")

    num_images = st.slider("Number of Images", min_value=1, max_value=4, value=1)

    use_random_seed = st.checkbox("Random seed each generation", value=True)
    fixed_seed = None
    if not use_random_seed:
        fixed_seed = st.number_input("Seed", min_value=0, value=42)

    st.divider()
    st.subheader("✨ Prompt Enhancer")
    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com. Reads from .env (GROQ_API_KEY) if set.",
    )

    st.divider()
    st.caption("Architecture: split-timeout policy, exponential backoff + jitter, "
               "chunked streaming, forced pixel-level integrity decode.")

if "prompt_widget" not in st.session_state:
    st.session_state.prompt_widget = ""

# Apply any pending enhanced prompt BEFORE the widget is instantiated this run
if st.session_state.get("_pending_prompt") is not None:
    st.session_state.prompt_widget = st.session_state._pending_prompt
    st.session_state._pending_prompt = None

prompt = st.text_area(
    "Describe the image you want to generate",
    placeholder="e.g. A futuristic city skyline at sunset, flying cars, neon reflections on wet streets",
    height=100,
    key="prompt_widget",
)

col_a, col_b = st.columns([1, 1])
with col_a:
    generate_clicked = st.button("🚀 Generate", type="primary", use_container_width=True)
with col_b:
    enhance_clicked = st.button("✨ Enhance Prompt", use_container_width=True)

if enhance_clicked:
    if not prompt.strip():
        st.warning("Pehle kuch prompt likho enhance karne ke liye.")
    else:
        with st.spinner("Enhancing your prompt with Llama 3.3..."):
            success, result = enhance_prompt(prompt.strip(), groq_key)
        if success:
            st.session_state._pending_prompt = result
            st.rerun()
        else:
            st.error(result)

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        full_prompt = prompt.strip() + STYLE_PRESETS[style_choice]
        cols = st.columns(num_images)

        for i in range(num_images):
            seed = fixed_seed if fixed_seed is not None else random.randint(1, 999999)
            url = build_payload_url(full_prompt, width, height, model_choice, seed)

            with cols[i]:
                status_box = st.empty()
                status_box.info(f"Generating image {i+1}/{num_images}...")

                success, result, failure_type = generate_image_with_retry(url)

                if not success:
                    if failure_type == "moderation":
                        status_box.error("🚫 Blocked by content safety filter. Try a different prompt.")
                    elif failure_type == "network":
                        status_box.error(f"🔌 Network failure: {result}")
                    else:
                        status_box.error(f"⏱️ Generation failed: {result}")
                    continue

                is_valid, verified = verify_image_integrity(result)

                if not is_valid:
                    status_box.error(f"⚠️ Corrupted / truncated image data: {verified}")
                    continue

                filepath = save_image(result, prompt)
                status_box.empty()
                st.image(verified, caption=f"Seed: {seed}", use_container_width=True)

                with open(filepath, "rb") as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f.read(),
                        file_name=os.path.basename(filepath),
                        mime="image/png",
                        key=f"download_{i}_{seed}",
                    )
                st.caption(f"✅ Saved & verified: `{filepath}`")

st.divider()
with st.expander("ℹ️ How this pipeline works"):
    st.markdown("""
    1. **Prompt Payload Formulation** — your prompt + style preset is URL-encoded and mapped to an exact pixel resolution based on the chosen aspect ratio.
    2. **Network API Gateway** — requests use a split timeout: `3.05s` to establish connection, up to `90s` to wait for the (slow, GPU-bound) generation.
    3. **Security Gates** — HTTP `403`/`451` responses are treated as content-moderation blocks and surfaced clearly instead of crashing the app.
    4. **Transport Protocol** — image bytes are streamed in `64 KiB` chunks rather than loaded all at once, keeping memory usage safe for high-resolution assets.
    5. **Integrity Verification** — every image is forced through a full pixel-level decode (`Image.load()`), so truncated/corrupted downloads are caught and discarded instead of silently saved.
    6. **Resilience** — transient failures (`429`, `5xx`, timeouts) are retried automatically with exponential backoff + jitter, up to 4 attempts.
    """)
