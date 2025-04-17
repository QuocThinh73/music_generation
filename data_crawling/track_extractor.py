import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from config import GENRES


def extract_track_info(driver, wait) -> dict:
    track_div = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-track-info]"))
    )
    return json.loads(track_div.get_attribute("data-track-info"))


def extract_genres(driver) -> list[str]:
    try:
        elem = driver.find_element(
            By.CSS_SELECTOR, "span.md\\:col-span-6.flex.flex-wrap.gap-3"
        )
        return [a.text.strip() for a in elem.find_elements(By.TAG_NAME, "a")]
    except Exception:
        return []


def extract_duration(driver) -> str:
    try:
        return driver.find_element(
            By.CSS_SELECTOR,
            "span.w-12.ml-auto.md\\:ml-0.col-span-2.inline-flex.justify-end.items-center"
        ).text.strip()
    except Exception:
        return ""


def extract_extra_info(driver) -> tuple[str, str]:
    instrumental, ai_generated = "No", "No"
    try:
        container = driver.find_element(
            By.CSS_SELECTOR, "div.px-8.py-2.bg-gray-light.flex.flex-col.divide-y.divide-gray"
        )
        rows = container.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.md\\:grid-cols-8.py-6")
        for row in rows:
            label = row.find_element(By.CSS_SELECTOR, "span.font-[500].md\\:col-span-2").text
            value = row.find_element(By.CSS_SELECTOR, "span.md\\:col-span-6").text
            if "Instrumental" in label:
                instrumental = value
            if "AI generated?" in label:
                ai_generated = value
    except Exception:
        pass
    return instrumental, ai_generated