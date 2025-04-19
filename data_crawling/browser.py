from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait

from data_crawling.config import CHROME_DRIVER_PATH, IMPLICIT_WAIT, WEB_DRIVER_DELAY

def init_driver():
    service = ChromeService(executable_path=CHROME_DRIVER_PATH)
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920x1080")

    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(IMPLICIT_WAIT)
    wait = WebDriverWait(driver, WEB_DRIVER_DELAY)
    return driver, wait
