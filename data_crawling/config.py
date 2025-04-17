CHROME_DRIVER_PATH = "/usr/bin/chromedriver"
WEB_DRIVER_DELAY = 10  # seconds for explicit waits
IMPLICIT_WAIT = 10     # seconds for implicit waits
BASE_SEARCH_URL = "https://freemusicarchive.org/search/"
GENRES = [
    "Instrumental", "Rock", "Soul-RnB", "Pop", "Electronic",
    "Classical", "Hip-Hop", "Folk", "Jazz", "Country",
    "Blues", "International", "Experimental", "Historic", "Novelty"
]
CRAWL_DELAY = 0.5  # seconds between requests
OUTPUT_DIR = "crawled_data"
AUDIO_DIR = "audio"
TOTAL_PAGES = 50  # Number of pages to crawl for each genre