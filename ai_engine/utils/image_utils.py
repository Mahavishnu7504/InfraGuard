import cv2
from pathlib import Path


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_result_image(image, filename="result.jpg"):

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(str(output_path), image)

    return str(output_path)