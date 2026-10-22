"""
wake_app.py - Automated keep-alive script for Streamlit application using Selenium.

Designed for scheduled execution (e.g., via GitHub Actions cron).
Visits the Streamlit app URL and clicks the "Yes, get this app back up!" button
if the app has gone to sleep due to inactivity.

Requirements:
    pip install selenium webdriver-manager

Usage:
    python wake_app.py
    python wake_app.py --url https://your-app.streamlit.app/
    APP_URL="https://your-app.streamlit.app/" python wake_app.py

Sample GitHub Actions workflow snippet (.github/workflows/wake_app.yml):
-------------------------------------------------------------------------
name: Keep App Alive

on:
  schedule:
    # Run every 6 hours (or adjust as needed)
    - cron: '0 */6 * * *'
  workflow_dispatch: # Allows manual trigger from Actions tab

jobs:
  wake:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install selenium webdriver-manager
      - name: Run wake script
        env:
          APP_URL: "https://trafficlens-ipfhyjvnrspjwazrw5gcmf.streamlit.app/"
        run: |
          python wake_app.py
"""

import argparse
import os
import sys
import time
from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

DEFAULT_APP_URL = "https://trafficlens-ipfhyjvnrspjwazrw5gcmf.streamlit.app/"


def log(message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Initializes and returns a Chrome WebDriver instance."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    try:
        # In modern Selenium (>= 4.6), ChromeDriverManager is built-in
        return webdriver.Chrome(options=chrome_options)
    except Exception as e:
        log(f"Standard Chrome initialization failed ({e}), attempting with webdriver-manager...")
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=chrome_options)
        except Exception as manager_error:
            log(f"webdriver-manager initialization also failed: {manager_error}")
            raise


def check_and_wake(driver: webdriver.Chrome, url: str, wait_timeout: int = 30) -> bool:
    """
    Loads target URL and triggers wake up if app has gone to sleep.

    Returns:
        bool: True if app is awake or successfully triggered wake up, False otherwise.
    """
    log(f"Navigating to {url} ...")
    driver.get(url)

    # Allow page scripts to initialize
    time.sleep(5)

    # Common Streamlit sleep button texts and selectors
    wake_button_selectors = [
        "//button[contains(., 'Yes, get this app back up!')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'get this app back up')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'wake')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'back up')]",
        "//div[contains(@class, 'viewer-badge')]//button",
    ]

    button_found = None
    for xpath in wake_button_selectors:
        try:
            elements = driver.find_elements(By.XPATH, xpath)
            for elem in elements:
                if elem.is_displayed():
                    button_found = elem
                    log(f"Found wake-up button with selector: {xpath}")
                    break
            if button_found:
                break
        except Exception:
            continue

    if button_found:
        log("App is currently SLEEPING. Clicking wake-up button...")
        try:
            button_found.click()
        except Exception:
            log("Standard click failed, executing JavaScript click fallback...")
            driver.execute_script("arguments[0].click();", button_found)

        log("Wake-up button clicked. Waiting for app to spin up...")
        time.sleep(15)

        # Check if loading spinner / app container appears
        try:
            WebDriverWait(driver, wait_timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="stAppViewContainer"], [data-testid="stApp"], .main, #root')
                )
            )
            log("App has initiated boot sequence successfully!")
            return True
        except TimeoutException:
            log("App wake-up requested, but initial load timed out. It may still be warming up.")
            return True

    # If no wake button was found, verify if app is already active
    log("No sleep screen detected. Checking if app is already active...")
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp, .main, #root')
            )
        )
        log("App is already AWAKE and responding normally.")
        return True
    except TimeoutException:
        # Check page title and body content for diagnostics
        title = driver.title
        log(f"Page title: {title}")
        body_text = driver.find_element(By.TAG_NAME, "body").text[:200]
        log(f"Body snippet: {body_text.replace(chr(10), ' ')}")

        # If it says 'Streamlit', it's active
        if "streamlit" in title.lower() or "trafficlens" in title.lower():
            log("Streamlit headers detected in title. App is active.")
            return True

        log("Warning: Could not conclusively confirm app active state.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wake up sleeping Streamlit application.")
    parser.add_argument(
        "--url",
        "-u",
        default=os.getenv("APP_URL", DEFAULT_APP_URL),
        help=f"Target Streamlit app URL (default: {DEFAULT_APP_URL})",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=2,
        help="Number of retry attempts if initial connection fails (default: 2)",
    )
    parser.add_argument(
        "--screenshot",
        "-s",
        action="store_true",
        default=os.getenv("SAVE_SCREENSHOT", "false").lower() in ("true", "1", "yes"),
        help="Save a screenshot (page_state.png) for debugging",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with GUI (useful for local debugging)",
    )

    args = parser.parse_args()
    target_url = args.url.strip()

    log(f"Starting keep-alive check for: {target_url}")

    driver = None
    success = False
    for attempt in range(1, args.retries + 1):
        log(f"Attempt {attempt}/{args.retries}...")
        try:
            driver = create_driver(headless=not args.no_headless)
            success = check_and_wake(driver, target_url)

            if args.screenshot and driver:
                screenshot_path = "page_state.png"
                driver.save_screenshot(screenshot_path)
                log(f"Saved debug screenshot to {screenshot_path}")

            if success:
                log("Keep-alive task completed successfully.")
                break
        except WebDriverException as e:
            log(f"WebDriver exception on attempt {attempt}: {e.msg if hasattr(e, 'msg') else e}")
            time.sleep(5)
        except Exception as e:
            log(f"Unexpected error on attempt {attempt}: {e}")
            time.sleep(5)
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = None

    if success:
        log("Finished with status: SUCCESS")
        return 0
    else:
        log("Finished with status: FAILED (App could not be verified as active)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
