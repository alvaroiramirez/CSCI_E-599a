"""
LLM Captioning and Classification Script
========================================

This script processes images from the LADI v2 dataset to detect disaster-related damages using the GPT-4o Vision API.

This script:
--------------
1. Loads and analyzes images from a defined source folder.
2. Sends images to the OpenAI API with rate-limiting (0.2s delay per request to avoid API blocking).
3. Extracts captions and structured binary labels indicating presence or absence of specific damage types.
4. Tracks analysis progress and saves results incrementally to prevent reprocessing.
5. Organizes output in folders based on damage classification.
6. Saves:
   - A JSON (analysis.json) file where we store the configuration used, detailed results per image, and summary.
   - A CSV (results.csv) with LADI v2-compliant binary labels and a per-image processing timestamp.
"""


# Imports
import os
import glob
import csv
import time
import base64
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Console setup
console = Console()


# CONSTANTS
# This variable controls whether to call the OpenAI API or not.
CALL_OPENAI_API = True

# Image folder
SOURCE_FOLDER = Path(__file__).resolve().parent.parent / "LADIv2"

# OpenAI API key
API_KEY_FILE = "/Users/alvaroramirez/Library/CloudStorage/OneDrive-Personal/estudio/Harvard/Classes/CSCI 599A/Code/API Keys/api_key.txt"

# Model
MODEL = "gpt-4o"

# Thresholds for flagging low-quality images, used for metadata only
# Images are still processed normally, but future cohorts could preprocess these images to obtain better results
QUALITY_SEAL_DARK_THRESHOLD = 20  # Threshold below which an image is flagged as very dark
QUALITY_SEAL_BLUR_THRESHOLD = 50  # Threshold below which an image is flagged as very blurry

# Delay between API requests (in seconds)
API_REQUEST_DELAY = 0.2

# Verbosity Setting 
VERBOSE = False  # Set to False to suppress detailed messages during processing

# Cost per token for OpenAI API usage
COST_PER_TOKEN = 0.000005 # I calculated this based on the API cost of $5 per 1M tokens

# Retry settings for API requests
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


# API Key Loading
def load_api_key(filepath):
    """Loads the API key from a specified file."""
    try:
        key_dir = Path(filepath).parent
        if not key_dir.exists():
            console.print(f"Error: The directory for the API key file does not exist: {key_dir}", style="bold red")
            console.print("Please ensure the directory structure is correct.", style="bold red")
            return None

        with open(filepath, 'r') as f:
            key = f.read().strip()
            if not key:
                console.print(f"Error: API key file '{filepath}' is empty.", style="bold red")
                return None
            return key
    except FileNotFoundError:
        console.print(f"Error: API key file '{filepath}' not found.", style="bold red")
        console.print(f"Please ensure the file exists at this exact path and contains your OpenAI API key.", style="bold red")
        return None
    except Exception as e:
        console.print(f"Error reading API key file '{filepath}': {e}", style="bold red")
        return None

# Load the API key
api_key = load_api_key(API_KEY_FILE)


# Helper Functions
def extract_binary_labels_from_caption(caption):
    labels = {
        "bridges_any": "0", "buildings_any": "0", "buildings_affected_or_greater": "0", "buildings_minor_or_greater": "0",
        "debris_any": "0", "flooding_any": "0", "flooding_structures": "0", "roads_any": "0", "roads_damage": "0",
        "trees_any": "0", "trees_damage": "0", "water_any": "0"
    }
    if not caption:
        return labels
    for line in caption.splitlines():
        if ':' in line:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                if key in labels and value in {"0", "1"}:
                    labels[key] = value
    return labels

def encode_image(image_path):
    """Encodes an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_image_dark(image_path, threshold=30):
    """Detects if an image is too dark based on average brightness."""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    brightness = np.mean(img_array)
    return brightness < threshold

def is_image_blurry(image_path, threshold=100):
    """Detects if an image is blurry based on Variance of Laplacian."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def analyze_image(image_path, current_api_key):
    """Analyzes a single image using the OpenAI API.
    
    Sends the image encoded in base64 along with a system prompt to guide the analysis.
    Returns the caption describing damages and the number of tokens used, or (None, 0) if an error occurs.
    Always returns total_wait_time as the third value.
    """
    total_wait_time = 0
    if not current_api_key:
        return None, 0, total_wait_time

    base64_image = encode_image(image_path)

    # Set up the headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {current_api_key}"
    }

    system_prompt = (
        "You are a FEMA disaster analyst reviewing aerial or ground images after natural disasters. "
        "Examine damages to buildings, roads, bridges, trees, and signs of flooding. "
        "Use these definitions:\n"
        "- bridges_any: at least one bridge present\n"
        "- buildings_any: any building visible\n"
        "- buildings_affected_or_greater: any damaged building (affected, minor, major, or destroyed)\n"
        "- buildings_minor_or_greater: damage at least minor\n"
        "- debris_any: visible debris (e.g., ruins, scattered material)\n"
        "- flooding_any: general water overflow not related to buildings\n"
        "- flooding_structures: buildings visibly flooded\n"
        "- roads_any: any road or highway present\n"
        "- roads_damage: roads visibly damaged or obstructed\n"
        "- trees_any: any trees visible\n"
        "- trees_damage: fallen or damaged trees\n"
        "- water_any: natural water bodies visible (not flood)\n\n"
        "Carefully examine the image to identify visible damage. Look attentively for any issues in buildings, trees, roads, bridges, and surrounding areas. Always report if:\n"
        "- Blue tarps indicate roof damage.\n"
        "- Any other hazards are visible.\n\n"
        "First, state the condition of buildings clearly:\n"
        "- If no buildings are present: \"No buildings detected.\"\n"
        "- If buildings are intact: \"Buildings appear in good condition.\"\n"
        "- If there are affected buildings, start your response with: \"Affected buildings.\"\n"
        "If the image is too blurry, dark, or unclear to determine damage reliably, start your response with: \"Not discernible.\"\n"
        "After that, briefly describe damage conditions as clearly and concisely as possible. "
        "At the end of your response, return the values for each label (from the definitions above) as 0 or 1 in this format:\n"
        "bridges_any: 0\n"
        "buildings_any: 1\n"
        "...\n"
        "water_any: 1"
    )

    # Prepare the user prompt with the image
    user_prompt = [
        {
            "type": "text",
            "text": "Analyze this image for visible disaster-related damages following a natural event, focusing on the condition of buildings, trees, poles, roads, and bridges."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

    # Prepare the payload for the API request
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            if CALL_OPENAI_API:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                analysis = response.json()
                caption = analysis['choices'][0]['message']['content']
                tokens_used = analysis['usage']['total_tokens']
                return caption, tokens_used, total_wait_time
            else:
                return None, 0, total_wait_time
        except requests.exceptions.RequestException as e:
            if response is not None and response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                delay = min(RETRY_DELAY * (2 ** attempt), 30)
                console.print(f"[yellow]Retry {attempt + 1}/{MAX_RETRIES} for image {Path(image_path).name}. Waiting {delay} seconds (total waited: {total_wait_time + delay} seconds)...[/yellow]")
                time.sleep(delay)
                total_wait_time += delay
                continue
            console.print(f"Error during API request for {Path(image_path).name}: {e}", style="bold red")
            if response is not None:
                console.print(f"Response status code: {response.status_code}", style="bold red")
                console.print(f"Response text: {response.text}", style="bold red")
            console.print(f"[bold red]API failed permanently after {attempt + 1} attempts and {total_wait_time} seconds waiting.[/bold red]")
            return None, 0, total_wait_time
        except KeyError as e:
            # Handle unexpected API response structure
            console.print(f"Error parsing API response for {Path(image_path).name}: Missing key {e}", style="bold red")
            console.print(f"Full response: {analysis}", style="bold red")
            console.print(f"[bold red]API failed permanently after {attempt + 1} attempts and {total_wait_time} seconds waiting.[/bold red]")
            return None, 0, total_wait_time
        except Exception as e:
            # Catch-all for any other exceptions
            console.print(f"An unexpected error occurred for {Path(image_path).name}: {e}", style="bold red")
            console.print(f"[bold red]API failed permanently after {attempt + 1} attempts and {total_wait_time} seconds waiting.[/bold red]")
            return None, 0, total_wait_time


# Main Execution
def main():
    """Main function to process all images in the source folder."""
    os.system('clear' if os.name == 'posix' else 'cls')
    console.print("[bold underline]Current Configuration:[/bold underline]", style="cyan")
    console.print(f"[bold]CALL_OPENAI_API:[/bold] {CALL_OPENAI_API}")
    console.print(f"[bold]SOURCE_FOLDER:[/bold] {SOURCE_FOLDER}")
    console.print(f"[bold]API_KEY_FILE:[/bold] {API_KEY_FILE}")
    console.print(f"[bold]MODEL:[/bold] {MODEL}")
    console.print(f"[bold]QUALITY_SEAL_DARK_THRESHOLD:[/bold] {QUALITY_SEAL_DARK_THRESHOLD}")
    console.print(f"[bold]QUALITY_SEAL_BLUR_THRESHOLD:[/bold] {QUALITY_SEAL_BLUR_THRESHOLD}")
    console.print(f"[bold]API_REQUEST_DELAY:[/bold] {API_REQUEST_DELAY}")
    console.print(f"[bold]VERBOSE:[/bold] {VERBOSE}")
    console.print(f"[bold]COST_PER_TOKEN:[/bold] {COST_PER_TOKEN}\n")
    if not CALL_OPENAI_API:
        console.print("⚠️ [bold yellow]API calls are disabled. The program will simulate analysis without sending data to OpenAI.[/bold yellow]")
    overall_start_time = time.time()
    if not api_key:
        console.print("API key could not be loaded. Exiting.", style="bold red")
        return
    base_dir = SOURCE_FOLDER
    source_dir = base_dir / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "analysis.json"
    csv_path = results_dir / "results.csv"
    processed_images = set()
    if csv_path.exists():
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "filename" in row and row["filename"]:
                    processed_images.add(row["filename"])
    if processed_images:
        console.print(f"Resuming from results.csv: {len(processed_images)} images already processed.", style="bold cyan")
        if VERBOSE:
            for img in sorted(processed_images):
                console.print(f"  • Skipping: {img}", style="dim")
    if not source_dir.exists():
        console.print(f"Error: Source directory '{source_dir}' not found.", style="bold red")
        console.print("Please ensure this folder exists and contains your images.", style="bold red")
        return
    elif not source_dir.is_dir():
        console.print(f"Error: The specified source path '{source_dir}' is not a directory.", style="bold red")
        return

    no_damages_dir = results_dir / "no_damages"
    with_damages_dir = results_dir / "with_damages"
    needs_review_dir = results_dir / "needs_review"
    not_discernible_dir = results_dir / "not_discernible"
    no_damages_dir.mkdir(exist_ok=True)
    with_damages_dir.mkdir(exist_ok=True)
    needs_review_dir.mkdir(exist_ok=True)
    not_discernible_dir.mkdir(exist_ok=True)
    image_files = glob.glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)
    image_files.sort()
    if not image_files:
        console.print(f"No JPG images found in the '{source_dir}' directory.", style="bold red")
        return
    if len(processed_images) >= len(image_files):
        console.print("All images already processed. Nothing to do.", style="bold green")
        return
    console.print()
    console.print(f"Found {len(image_files)} images to analyze in '{source_dir}'.", style="bold green")

    def safe_json_dump(data, filepath):
        class EnhancedEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.bool)):
                    return bool(obj)
                return super().default(obj)
        with open(filepath, mode='w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, cls=EnhancedEncoder)

    config_dict = {
        "CALL_OPENAI_API": CALL_OPENAI_API,
        "SOURCE_FOLDER": str(SOURCE_FOLDER),
        "API_KEY_FILE": API_KEY_FILE,
        "MODEL": MODEL,
        "QUALITY_SEAL_DARK_THRESHOLD": QUALITY_SEAL_DARK_THRESHOLD,
        "QUALITY_SEAL_BLUR_THRESHOLD": QUALITY_SEAL_BLUR_THRESHOLD,
        "API_REQUEST_DELAY": API_REQUEST_DELAY,
        "VERBOSE": VERBOSE,
        "COST_PER_TOKEN": COST_PER_TOKEN
    }

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
                analysis_results = json_data.get("analysis_results", {})
                session_start = json_data.get("session_start", datetime.now().isoformat())
        except Exception:
            analysis_results = {}
            session_start = datetime.now().isoformat()
    else:
        analysis_results = {}
        session_start = datetime.now().isoformat()


    safe_json_dump({
        "configuration": config_dict,
        "session_start": session_start,
        "analysis_results": analysis_results,
        "summary": {}
    }, json_path)

    csv_headers = [
        "filename", "bridges_any", "buildings_any", "buildings_affected_or_greater", "buildings_minor_or_greater",
        "debris_any", "flooding_any", "flooding_structures", "roads_any", "roads_damage",
        "trees_any", "trees_damage", "water_any", "timestamp"
    ]
    if not csv_path.exists():
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    results = {}
    images_per_subfolder = {}
    tokens_per_subfolder = {}
    subfolder_times = {}
    total_tokens_used = 0
    console.print()

    # Progress bar and per-image processing
    with Progress() as progress:
        task = progress.add_task("[cyan]Analyzing images...", total=len(image_files))
        for image_path in image_files:
            filename = Path(image_path).name
            if filename in processed_images:
                progress.update(task, advance=1)
                continue
            is_dark = is_image_dark(image_path, threshold=QUALITY_SEAL_DARK_THRESHOLD)
            is_blurry = is_image_blurry(image_path, threshold=QUALITY_SEAL_BLUR_THRESHOLD)
            caption, tokens_used, total_wait_time = analyze_image(image_path, api_key)

            if CALL_OPENAI_API and caption is None:
                console.print(f"\n[bold red]API call failed for image: {filename}. Halting execution to preserve data integrity.[/bold red]")
                console.print(f"[bold red]Total retries attempted: {MAX_RETRIES}[/bold red]")
                console.print(f"[bold red]Total wait time due to retries: {total_wait_time} seconds[/bold red]")
                return
            time.sleep(API_REQUEST_DELAY)
            if caption:
                caption_lower = caption.lower()
                if caption_lower.startswith("image not discernible for damage assessment."):
                    classification = "not_discernible"
                    needs_manual_review = "NO"
                else:
                    label_values = extract_binary_labels_from_caption(caption)
                    damage_indicators = [
                        "buildings_affected_or_greater", "buildings_minor_or_greater", "debris_any",
                        "flooding_any", "flooding_structures", "roads_damage", "trees_damage"
                    ]
                    has_damage = any(label_values[k] == "1" for k in damage_indicators)
                    classification = "with_damages" if has_damage else "no_damages"
                    needs_manual_review = "NO"
            else:
                classification = "failed_analysis"
                needs_manual_review = "YES"
            total_tokens_used += tokens_used
            subfolder_name = Path(image_path).parent.name
            if subfolder_name not in subfolder_times:
                subfolder_times[subfolder_name] = time.time()
            results[filename] = (caption, tokens_used, needs_manual_review, classification, is_dark, is_blurry)

            if classification == "no_damages":
                dest_folder = no_damages_dir
            elif classification == "with_damages":
                dest_folder = with_damages_dir
            elif classification == "not_discernible":
                dest_folder = not_discernible_dir
            else:
                dest_folder = needs_review_dir
            dest_path = dest_folder / filename
            if classification != "failed_analysis":
                with open(image_path, 'rb') as src_file, open(dest_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            images_per_subfolder[subfolder_name] = images_per_subfolder.get(subfolder_name, 0) + 1
            tokens_per_subfolder[subfolder_name] = tokens_per_subfolder.get(subfolder_name, 0) + tokens_used
            if VERBOSE:
                elapsed = time.time() - subfolder_times[subfolder_name]
                progress.console.log(f"[bold cyan]{filename}[/bold cyan] ({elapsed:.2f} sec)")
                progress.console.log(str(extract_binary_labels_from_caption(caption)))
            progress.update(task, advance=1)

            label_values = extract_binary_labels_from_caption(caption)
            row = [
                filename,
                label_values["bridges_any"],
                label_values["buildings_any"],
                label_values["buildings_affected_or_greater"],
                label_values["buildings_minor_or_greater"],
                label_values["debris_any"],
                label_values["flooding_any"],
                label_values["flooding_structures"],
                label_values["roads_any"],
                label_values["roads_damage"],
                label_values["trees_any"],
                label_values["trees_damage"],
                label_values["water_any"],
                datetime.now().isoformat()
            ]

            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    json_data = json.load(jf)
            except Exception:
                json_data = {
                    "configuration": config_dict,
                    "session_start": session_start,
                    "analysis_results": {},
                    "summary": {}
                }

            if "analysis_results" not in json_data or not isinstance(json_data["analysis_results"], dict):
                json_data["analysis_results"] = {}

            if filename not in json_data["analysis_results"]:
                json_data["analysis_results"][filename] = {
                    "caption": caption.replace("\n", " ") if caption else "",
                    "tokens_used": tokens_used,
                    "needs_manual_review": needs_manual_review,
                    "classification": classification,
                    "potentially_dark": bool(is_dark),
                    "potentially_blurry": bool(is_blurry),
                }

            safe_json_dump(json_data, json_path)


if __name__ == "__main__":
    main()
