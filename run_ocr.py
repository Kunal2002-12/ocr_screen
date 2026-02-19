from pathlib import Path
import argparse
import tempfile
from difflib import SequenceMatcher
from typing import Optional, List, Tuple
from tkinter import Tk, filedialog

from PIL import Image, UnidentifiedImageError
from paddleocr import PaddleOCR


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR from image path or popup, save text + visualization."
    )
    parser.add_argument("--image", help="Path to input image (optional if popup is used)")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code (default: en)")
    parser.add_argument("--expected", help="Optional expected .txt path for similarity check")
    return parser.parse_args()


def choose_image_from_popup() -> Optional[Path]:
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select image for OCR",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if not file_path:
            return None
        return Path(file_path).expanduser().resolve()
    except Exception as exc:
        print(f"Could not open file picker: {exc}")
        return None


def prepare_image_for_ocr(source: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    try:
        with Image.open(source) as img:
            # Normalize all Pillow-supported image types to PNG for OCR stability.
            rgb = img.convert("RGB")
            temp_path = Path(tempfile.gettempdir()) / f"{source.stem}_ocr_input.png"
            rgb.save(temp_path, format="PNG")
            return temp_path
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {source}") from exc


def save_visualizations(prediction_results, output_dir: Path):
    saved = 0
    for i, res_obj in enumerate(prediction_results, start=1):
        if hasattr(res_obj, "save_to_img"):
            try:
                res_obj.save_to_img(str(output_dir))
                saved += 1
                print(f"Saved visualization for result {i}")
            except Exception as exc:
                print(f"Failed to save visualization for result {i}: {exc}")
        else:
            print(f"Result {i} has no save_to_img(); skipping.")

    if saved == 0:
        print("No visualization files were saved.")


def extract_text(prediction_results) -> str:
    chunks: List[str] = []

    for i, res_obj in enumerate(prediction_results, start=1):
        item_text = None

        # Primary format: paddlex OCRResult with .json["res"]["rec_texts"]
        json_data = getattr(res_obj, "json", None)
        if isinstance(json_data, dict):
            res = json_data.get("res", {})
            if isinstance(res, dict):
                rec_texts = res.get("rec_texts", [])
                if isinstance(rec_texts, list):
                    meaningful = [t for t in rec_texts if isinstance(t, str) and t.strip()]
                    if meaningful:
                        item_text = "\n".join(meaningful)

        # Fallback format: list-style result
        if not item_text and isinstance(res_obj, list):
            fallback_lines: List[str] = []
            for line_info in res_obj:
                if isinstance(line_info, list) and len(line_info) == 2:
                    text_component = line_info[1]
                    if isinstance(text_component, tuple) and len(text_component) == 2:
                        text = text_component[0]
                        if isinstance(text, str) and text.strip():
                            fallback_lines.append(text.strip())
            if fallback_lines:
                item_text = "\n".join(fallback_lines)

        if item_text:
            chunks.append(item_text)
        else:
            print(f"No text extracted from result {i}")

    return "\n---\n".join(chunks)


def confidence_report(prediction_results, threshold: float) -> Tuple[float, List[Tuple[str, float]]]:
    scores: List[float] = []
    low_lines: List[Tuple[str, float]] = []

    for res_obj in prediction_results:
        json_data = getattr(res_obj, "json", None)
        if not isinstance(json_data, dict):
            continue

        res = json_data.get("res", {})
        if not isinstance(res, dict):
            continue

        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        if not isinstance(rec_texts, list) or not isinstance(rec_scores, list):
            continue

        for text, score in zip(rec_texts, rec_scores):
            if isinstance(text, str) and text.strip() and isinstance(score, (int, float)):
                score_f = float(score)
                scores.append(score_f)
                if score_f < threshold:
                    low_lines.append((text, score_f))

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, low_lines


def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "ocr_output_predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        source_image = Path(args.image).expanduser().resolve()
    else:
        source_image = choose_image_from_popup()
        if source_image is None:
            print("No image selected. Exiting.")
            return

    try:
        prepared_image = prepare_image_for_ocr(source_image)
    except Exception as exc:
        print(f"Input image error: {exc}")
        return

    print(f"Using OCR language: {args.lang}")
    print(f"Processing image: {source_image}")

    ocr = PaddleOCR(
        lang=args.lang,
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    try:
        prediction_results = ocr.predict(str(prepared_image))
    except Exception as exc:
        print(f"OCR execution failed: {exc}")
        return

    if not prediction_results:
        print("OCR returned no results.")
        return

    # Save visual outputs from PaddleOCR
    save_visualizations(prediction_results, output_dir)

    # Extract and save text
    final_text = extract_text(prediction_results)
    if not final_text:
        print("OCR finished but no text was extracted.")
        return

    print("\nRecognized Text:\n")
    print(final_text)

    txt_file = output_dir / f"{source_image.stem}_ocr.txt"
    txt_file.write_text(final_text, encoding="utf-8")
    print(f"\nText saved to: {txt_file}")

    # Optional expected-text similarity check
    if args.expected:
        expected_path = Path(args.expected).expanduser().resolve()
        if expected_path.exists():
            expected_text = expected_path.read_text(encoding="utf-8")
            similarity = SequenceMatcher(
                None,
                normalize_text(final_text),
                normalize_text(expected_text),
            ).ratio()
            print(f"Similarity with expected text file: {similarity:.2%}")
        else:
            print(f"Expected text file not found: {expected_path}")


if __name__ == "__main__":
    main()
