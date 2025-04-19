import os
import json
import time
from data_crawling.track_extractor import extract_track_info, extract_genres, extract_duration, extract_extra_info
from data_crawling.downloader import download_audio_file

def process_audio_page(
    audio_url: str,
    driver,
    wait,
    index: int,
    output_dir: str
) -> dict:

    driver.get(audio_url)
    info = extract_track_info(driver, wait)
    file_url = info.get("fileUrl", "")
    metadata = {
        "audioName": info.get("title", "").strip(),
        "author": info.get("artistName", "").strip(),
        "genres": extract_genres(driver),
        "duration": extract_duration(driver),
        "instrumental": extract_extra_info(driver)[0],
        "ai_generated": extract_extra_info(driver)[1],
        "audio_url": audio_url
    }

    audio_fname = f"audio_{index:04d}.mp3"
    meta_fname = f"audio_{index:04d}.json"
    audio_fp = os.path.join(output_dir, "audio", audio_fname)
    meta_fp = os.path.join(output_dir, meta_fname)

    download_audio_file(file_url, audio_fp)
    with open(meta_fp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    time.sleep(0.5)
    return metadata
