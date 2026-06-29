from pathlib import Path
import csv
import shutil
import re


PROJECT_ROOT = Path("/home/roolab/shs/RoboLabProjects/VLAwithAIF")
OPENVLA_ROOT = PROJECT_ROOT / "external" / "openvla-oft"
OUT_ROOT = PROJECT_ROOT / "outputs" / "libero_baseline"


def latest_file(pattern):
    files = list(OPENVLA_ROOT.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    latest_eval_log = latest_file("experiments/logs/EVAL-libero_spatial-openvla-*.txt")
    if latest_eval_log is None:
        raise FileNotFoundError("No official LIBERO eval log found.")

    # Example:
    # EVAL-libero_spatial-openvla-2026_06_29-11_16_55.txt
    match = re.search(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", latest_eval_log.name)
    run_id = match.group(1) if match else "latest"

    run_out = OUT_ROOT / run_id
    videos_out = run_out / "videos"
    run_out.mkdir(parents=True, exist_ok=True)
    videos_out.mkdir(parents=True, exist_ok=True)

    copied_eval_log = run_out / latest_eval_log.name
    shutil.copy2(latest_eval_log, copied_eval_log)

    # Copy matching rollout videos for this run timestamp
    video_files = list(OPENVLA_ROOT.glob(f"rollouts/**/*{run_id}*.mp4"))

    # Fallback: if timestamp matching fails, copy newest videos
    if not video_files:
        all_videos = list(OPENVLA_ROOT.glob("rollouts/**/*.mp4"))
        all_videos = sorted(all_videos, key=lambda p: p.stat().st_mtime, reverse=True)
        video_files = all_videos[:20]

    rows = []
    success_count = 0
    failure_count = 0

    for video in video_files:
        dst = videos_out / video.name
        shutil.copy2(video, dst)

        name = video.name
        success = None

        if "success=True" in name:
            success = True
            success_count += 1
        elif "success=False" in name:
            success = False
            failure_count += 1

        rows.append({
            "video": str(dst.relative_to(PROJECT_ROOT)),
            "success": success,
            "filename": name,
        })

    total = success_count + failure_count
    success_rate = success_count / total if total > 0 else None

    csv_path = run_out / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "success", "filename"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = run_out / "summary.md"
    with open(md_path, "w") as f:
        f.write("# LIBERO-Spatial OpenVLA-OFT baseline\n\n")
        f.write(f"Run ID: `{run_id}`\n\n")
        f.write(f"Official eval log: `{copied_eval_log.relative_to(PROJECT_ROOT)}`\n\n")
        f.write(f"Videos copied: `{len(video_files)}`\n\n")
        f.write(f"Successes: `{success_count}`\n\n")
        f.write(f"Failures: `{failure_count}`\n\n")
        if success_rate is not None:
            f.write(f"Success rate from video filenames: `{success_rate:.3f}`\n\n")
        else:
            f.write("Success rate from video filenames: `unknown`\n\n")

        f.write("## Videos\n\n")
        for row in rows:
            f.write(f"- `{row['video']}` — success: `{row['success']}`\n")

    print("Collected baseline results.")
    print("Run output:", run_out)
    print("Summary CSV:", csv_path)
    print("Summary MD:", md_path)
    print("Videos:", videos_out)
    print("Successes:", success_count)
    print("Failures:", failure_count)
    if success_rate is not None:
        print("Success rate:", round(success_rate, 3))


if __name__ == "__main__":
    main()
