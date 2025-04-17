# Data Crawling Package

## Quick Start

Chạy lệnh để thu thập dữ liệu:

```bash
python main.py
```

- Script sẽ tự động tạo thư mục `crawled_data/`
- Audio files được lưu trong `crawled_data/audio/`
- Metadata JSON được lưu trong `crawled_data/`

## Configuration (`config.py`)

| Tham số             | Mô tả                                                        |
|---------------------|--------------------------------------------------------------|
| `EDGE_DRIVER_PATH`  | Đường dẫn đến file `msedgedriver`                            |
| `WEB_DRIVER_DELAY`  | Thời gian (giây) chờ cho explicit waits của Selenium         |
| `IMPLICIT_WAIT`     | Thời gian (giây) chờ cho implicit waits của Selenium         |
| `BASE_SEARCH_URL`   | URL gốc cho các truy vấn tìm kiếm trên Free Music Archive    |
| `GENRES`            | Danh sách thể loại (genres) sẽ crawl                         |
| `CRAWL_DELAY`       | Độ trễ (giây) giữa mỗi lần xử lý track                       |
| `OUTPUT_DIR`        | Thư mục gốc lưu kết quả (mặc định: `crawled_data`)           |
| `AUDIO_DIR`         | Thư mục con dưới `OUTPUT_DIR` để lưu file MP3                |
| `TOTAL_PAGES`       | Số trang cần crawl cho mỗi thể loại                          |

## Module Structure

```
data_crawling/
├── config.py          # Thiết lập hằng số và tham số chung
├── browser.py         # Cấu hình và khởi tạo Selenium WebDriver
├── menu_extractor.py  # Thao tác lấy danh sách URL track từ trang listing
├── track_extractor.py # Thao tác trích xuất metadata chi tiết của track
├── downloader.py      # Hàm tải file MP3 về local
├── processor.py       # Điều phối trích xuất, tải file và lưu metadata
└── main.py            # Entry point: lặp qua genres/pages và chạy crawler
```