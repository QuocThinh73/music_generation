import os
from tqdm import tqdm
from config import BASE_SEARCH_URL, GENRES, OUTPUT_DIR, TOTAL_PAGES
from browser import init_driver
from menu_extractor import extract_audio_links_from_menu
from processor import process_audio_page


def loop_over_menu_pages(base_url: str, total_pages: int, driver, wait) -> list[str]:
    links = []
    for page in range(1, total_pages + 1):
        url = f"{base_url}?page={page}"
        page_links = extract_audio_links_from_menu(url, driver, wait)
        links.extend(page_links)
    return links


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)

    driver, wait = init_driver()
    idx = 1
    for genre in GENRES:
        search_url = f"{BASE_SEARCH_URL}?quicksearch=&search-genre={genre}"
        audio_urls = loop_over_menu_pages(search_url, total_pages=TOTAL_PAGES, driver=driver, wait=wait)
        for url in tqdm(audio_urls, desc=f"Downloading {genre}"):
            try:
                process_audio_page(url, driver, wait, idx, OUTPUT_DIR)
                idx += 1
            except Exception as e:
                print(f"Failed {url}: {e}")
    driver.quit()