import os
import sys

DATASET_ROOT = r"C:\major_project\WeatherRemover\raindrop"

def generate_txt(split):
    input_dir = os.path.join(DATASET_ROOT, split, "input")
    gt_dir = os.path.join(DATASET_ROOT, split, "gt")
    txt_file = os.path.join(DATASET_ROOT, split, f"{split}.txt")

    print(f"Checking: {input_dir}")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] input folder not found: {input_dir}")
        sys.exit(1)

    if not os.path.isdir(gt_dir):
        print(f"[ERROR] gt folder not found: {gt_dir}")
        sys.exit(1)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if len(images) == 0:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)

    with open(txt_file, "w") as f:
        for img in images:
            if not os.path.exists(os.path.join(gt_dir, img)):
                print(f"[ERROR] Missing GT for {img}")
                sys.exit(1)
            f.write(f"input/{img}\n")

    print(f"[SUCCESS] {split}.txt created with {len(images)} entries")

if __name__ == "__main__":
    generate_txt("train")
    generate_txt("test")
