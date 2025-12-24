import os

gt_dir = r"C:\major_project\WeatherRemover\raindrop\test\gt"

for filename in os.listdir(gt_dir):
    if filename.endswith("_clean.png"):
        new_name = filename.replace("_clean", "_rain")
        src = os.path.join(gt_dir, filename)
        dst = os.path.join(gt_dir, new_name)

        if not os.path.exists(dst):
            os.rename(src, dst)
            print(f"{filename}  ->  {new_name}")
        else:
            print(f"[SKIP] {new_name} already exists")
