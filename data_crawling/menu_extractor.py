import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


def extract_audio_links_from_menu(page_url: str, driver, wait) -> list[str]:
    driver.get(page_url)
    container = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.w-full.flex.flex-col.gap-3.pt-3")
        )
    )
    play_items = container.find_elements(By.CSS_SELECTOR, "div.play-item")
    links = []
    for item in play_items:
        try:
            link = item.find_element(By.CSS_SELECTOR, ".ptxt-track a").get_attribute("href")
            links.append(link)
        except Exception:
            continue
    return links