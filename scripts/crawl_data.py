import sys
import os
import json
import time

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_crawling.config import BASE_SEARCH_URL, GENRES, OUTPUT_DIR, TOTAL_PAGES
from data_crawling.browser import init_driver
from data_crawling.menu_extractor import extract_audio_links_from_menu
from data_crawling.processor import process_audio_page

RESET_DRIVER_EVERY = 5  # Reset driver mỗi 5 trang

def get_existing_audio_urls_and_max_idx() -> tuple[set, int]:
    existing_urls = set()
    max_idx = 0
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith(".json") and fname.startswith("audio_"):
            try:
                idx = int(fname.replace("audio_", "").replace(".json", ""))
                max_idx = max(max_idx, idx)

                with open(os.path.join(OUTPUT_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    audio_url = data.get("audio_url", "")
                    if audio_url:
                        existing_urls.add(audio_url)
            except Exception:
                continue
    return existing_urls, max_idx

def crawl_pages_by_group(base_url: str, total_pages: int) -> list[str]:
    all_links = []

    for group_start in range(1, total_pages + 1, RESET_DRIVER_EVERY):
        group_end = min(group_start + RESET_DRIVER_EVERY - 1, total_pages)
        print(f"\n🚀 Đang crawl nhóm trang {group_start} → {group_end}")
        driver, wait = init_driver()

        for page in range(group_start, group_end + 1):
            url = f"{base_url}&page={page}"
            print(f"\n🧭 Truy cập: {url}")
            try:
                page_links = extract_audio_links_from_menu(url, driver, wait)
                print(f"✅ Trang {page}: {len(page_links)} audio được tìm thấy.")
                all_links.extend(page_links)
            except Exception as e:
                print(f"❌ Trang {page} lỗi → {e}")
        driver.quit()
        time.sleep(2)  # tránh bị chặn IP

    return all_links

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)

    existing_urls, current_max_idx = get_existing_audio_urls_and_max_idx()
    print(f"\n📁 File cao nhất: audio_{current_max_idx:04d}.mp3")
    print(f"🔍 Đã có {len(existing_urls)} audio được crawl trước đó.\n")

    idx = current_max_idx + 1

    for genre in GENRES:
        print(f"\n🎶 Bắt đầu với thể loại: {genre}")
        search_url = f"{BASE_SEARCH_URL}?quicksearch=&search-genre={genre}"

        all_audio_urls = crawl_pages_by_group(search_url, TOTAL_PAGES)
        print(f"\n🎧 Tổng số URL crawl được: {len(all_audio_urls)}")

        unique_urls = list(dict.fromkeys(all_audio_urls))
        print(f"✅ Sau khi loại trùng: {len(unique_urls)} audio")

        driver, wait = init_driver()
        for url in tqdm(unique_urls, desc=f"⬇️ Đang tải {genre}"):
            if url in existing_urls:
                print(f"⚠️ URL đã tồn tại: {url}")
                continue

            try:
                process_audio_page(url, driver, wait, idx, OUTPUT_DIR)
                existing_urls.add(url)
                idx += 1
                time.sleep(1)
            except Exception as e:
                print(f"❌ Lỗi khi tải {url}: {e}")
        driver.quit()

    print("\n✅ Đã crawl xong toàn bộ.")
