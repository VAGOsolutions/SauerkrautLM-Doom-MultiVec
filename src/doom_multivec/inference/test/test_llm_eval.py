"""Simple test harness for `llm_eval`.

Loads `test/image.png` (creates a small placeholder image if missing),
and sends it to the LLM via `query_llm_with_frames`. API key is taken
from the `LITELLM_API_KEY` environment variable or a `.env` file if
`python-dotenv` is installed.

Run:
    python test_llm_eval.py

"""
from pathlib import Path
import sys
import importlib.util
from PIL import Image

# Ensure package import works by adding the repo's `src` directory to sys.path
p = Path(__file__).resolve()
for ancestor in p.parents:
    if ancestor.name == 'src':
        sys.path.insert(0, str(ancestor))
        break
else:
    # Fallback: add parent-of-parent-of-parent (best-effort)
    sys.path.insert(0, str(p.parents[3]))

llm_eval_path = p.parents[1] / "llm_eval.py"
spec = importlib.util.spec_from_file_location("llm_eval", llm_eval_path)
assert spec is not None and spec.loader is not None
llm_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_eval)


def main():
    test_dir = Path(__file__).resolve().parent
    image_path = test_dir / "image.png"

    if not image_path.exists():
        print(f"{image_path} not found — creating a placeholder image...")
        img = Image.new('RGB', (64, 64), (200, 50, 50))
        img.save(image_path)

    # Load the image
    img = Image.open(image_path).convert('RGB')

    prompt = (
        "Describe the contents of the given image in detail."
    )

    print("Sending image to LLM (will use LITELLM_API_KEY from environment if available)...")
    try:
        resp = llm_eval.query_llm_with_frames([img], prompt, api_key=None)
    except Exception as e:
        print("Request failed:", e)
        return

    print("Raw response (truncated):")
    try:
        import json
        print(json.dumps(resp, indent=2))
    except Exception:
        print(str(resp)[:400])

    # Try to extract text reply
    try:
        text = llm_eval._extract_text_from_response(resp)
        print("\nModel reply:")
        print(text)
    except Exception as e:
        print("Failed to extract text from response:", e)


if __name__ == '__main__':
    main()
