import csv
import io
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..webdriver import create_chrome_driver

logger = logging.getLogger(__name__)

FORM_READY_TIMEOUT_SECONDS = 20
FORM_OPEN_RETRIES = 3
FORM_OPEN_RETRY_BACKOFF_SECONDS = 2


class DatasusTabnetForm:
    """Small Selenium wrapper around common DATASUS TABNET form interactions."""

    def __init__(
        self,
        headless=False,
        download_dir=None,
        page_load_wait=3,
        reset_wait=2,
        result_wait_seconds=180,
        form_ready_timeout_seconds=FORM_READY_TIMEOUT_SECONDS,
        form_open_retries=FORM_OPEN_RETRIES,
    ):
        self._owns_download_dir = download_dir is None
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="datasus_download_")
        self.driver = create_chrome_driver(headless=headless, download_dir=self.download_dir)
        self.page_load_wait = page_load_wait
        self.reset_wait = reset_wait
        self.result_wait_seconds = result_wait_seconds
        self.form_ready_timeout_seconds = form_ready_timeout_seconds
        self.form_open_retries = form_open_retries

    def open(self, url):
        last_error = None
        for attempt in range(1, self.form_open_retries + 1):
            try:
                self.driver.get(url)
                self._wait_for_form_ready()
                time.sleep(self.page_load_wait)
                return
            except TimeoutException as exc:
                last_error = exc
                self._stop_page_load()
            except WebDriverException as exc:
                last_error = exc
                self._stop_page_load()

            if attempt == self.form_open_retries:
                break

            time.sleep(min(5, attempt * FORM_OPEN_RETRY_BACKOFF_SECONDS))

        raise TimeoutException(f"Unable to open DATASUS form after {self.form_open_retries} attempts: {url}") from last_error

    def _wait_for_form_ready(self):
        WebDriverWait(self.driver, self.form_ready_timeout_seconds).until(
            EC.presence_of_element_located((By.XPATH, "//select[@name='Linha']"))
        )
        WebDriverWait(self.driver, self.form_ready_timeout_seconds).until(
            EC.presence_of_element_located((By.XPATH, "//select[@name='Incremento']"))
        )

    def _wait_for_select_options(self, select_name, minimum_options=1):
        WebDriverWait(self.driver, self.form_ready_timeout_seconds).until(
            EC.presence_of_element_located((By.XPATH, f"//select[@name='{select_name}']"))
        )
        WebDriverWait(self.driver, self.form_ready_timeout_seconds).until(
            lambda driver: len(
                driver.find_elements(By.XPATH, f"//select[@name='{select_name}']/option")
            )
            >= minimum_options
        )

    def _stop_page_load(self):
        try:
            self.driver.execute_script("window.stop();")
        except WebDriverException:
            pass

    def _click_option(self, xpath):
        self.driver.find_element(By.XPATH, xpath).click()

    def _option_text_xpath(self, select_name, text, exact=False):
        if exact:
            predicate = f"normalize-space(text())='{text}'"
        else:
            predicate = f"contains(normalize-space(text()), '{text}')"
        return f"//select[@name='{select_name}']/option[{predicate}]"

    def select_line(self, value):
        self._click_option(f"//select[@name='Linha']/option[@value='{value}']")

    def select_column(self, value):
        self._click_option(f"//select[@name='Coluna']/option[@value='{value}']")

    def select_line_text(self, text, exact=False):
        self._click_option(self._option_text_xpath("Linha", text, exact=exact))

    def select_column_text(self, text, exact=False):
        self._click_option(self._option_text_xpath("Coluna", text, exact=exact))

    def select_increment(self, value):
        self._click_option(f"//select[@name='Incremento']/option[@value='{value}']")

    def select_content_value(self, value):
        self.set_multiselect_values("Incremento", [value])

    def select_content_values(self, values):
        self.set_multiselect_values("Incremento", values)

    def select_all_content_values(self):
        self.set_all_multiselect_options("Incremento")

    def select_content_text(self, text, exact=False):
        for option in self.get_select_options("Incremento"):
            option_text = option["text"]
            if (exact and option_text == text) or ((not exact) and text in option_text):
                self.select_content_value(option["value"])
                return option["value"]
        raise NoSuchElementException(f"Could not find content option matching {text!r}")

    def select_option_value(self, value):
        self._click_option(f"//option[@value='{value}']")

    def select_options_with_value_fragment(self, fragment, pause_seconds=0.2):
        options = self.driver.find_elements(By.XPATH, f"//option[contains(@value, '{fragment}')]")
        for option in options:
            time.sleep(pause_seconds)
            option.click()

    def select_option_with_text(self, text):
        self._click_option(f"//option[contains(text(), '{text}')]")

    def select_option_with_exact_text(self, text):
        self._click_option(f"//option[normalize-space(text())='{text}']")

    def select_filter_option_text(self, select_names, text, exact=False):
        if isinstance(select_names, str):
            select_names = [select_names]
        for select_name in select_names:
            try:
                self._click_option(self._option_text_xpath(select_name, text, exact=exact))
                return select_name
            except NoSuchElementException:
                pass
        if exact:
            self.select_option_with_exact_text(text)
        else:
            self.select_option_with_text(text)
        return None

    def select_filter_option_texts(self, select_names, texts, exact=False, pause_seconds=0.2):
        for text in texts:
            self.select_filter_option_text(select_names, text, exact=exact)
            time.sleep(pause_seconds)

    def select_filter_option_value(self, select_name, value):
        self.set_multiselect_values(select_name, [str(value)])

    def get_select_options(self, select_name):
        self._wait_for_select_options(select_name)
        options = self.driver.find_elements(By.XPATH, f"//select[@name='{select_name}']/option")
        return [
            {
                "value": option.get_attribute("value"),
                "text": option.text.strip(),
                "selected": option.is_selected(),
            }
            for option in options
        ]

    def set_multiselect_values(self, select_name, values, clear_first=True):
        values = list(values)
        available_options = self.get_select_options(select_name)
        available_values = {option["value"] for option in available_options}
        missing_values = [value for value in values if value not in available_values]
        if missing_values:
            raise NoSuchElementException(
                f"Could not find values {missing_values!r} in select {select_name!r}"
            )

        script = """
            const select = document.querySelector(`select[name="${arguments[0]}"]`);
            if (!select) return null;
            const targets = new Set(arguments[1]);
            const clearFirst = arguments[2];
            for (const option of select.options) {
                if (clearFirst) {
                    option.selected = false;
                }
                if (targets.has(option.value)) {
                    option.selected = true;
                }
            }
            select.dispatchEvent(new Event('change', { bubbles: true }));
            return Array.from(select.options)
                .filter(option => option.selected)
                .map(option => option.value);
        """
        selected_values = self.driver.execute_script(script, select_name, values, clear_first)
        if selected_values is None:
            raise NoSuchElementException(f"Could not find select {select_name!r}")
        missing_after_selection = [value for value in values if value not in set(selected_values)]
        if missing_after_selection:
            raise RuntimeError(
                f"Failed to select values {missing_after_selection!r} in select {select_name!r}"
            )
        logger.debug(
            "Selected multiselect values: select=%s, count=%s",
            select_name,
            len(selected_values),
        )

    def set_all_multiselect_options(self, select_name):
        values = [option["value"] for option in self.get_select_options(select_name)]
        self.set_multiselect_values(select_name, values)

    def find_option_texts_by_fragments(self, select_name, fragments):
        option_texts = [option["text"] for option in self.get_select_options(select_name)]
        matched = []
        seen = set()
        for fragment in fragments:
            for option_text in option_texts:
                if fragment in option_text and option_text not in seen:
                    matched.append(option_text)
                    seen.add(option_text)
        return matched

    def find_options_by_values(self, select_name, values):
        value_set = {str(value) for value in values}
        matched = []
        for option in self.get_select_options(select_name):
            if option["value"] in value_set:
                matched.append(
                    {
                        "value": option["value"],
                        "text": option["text"],
                    }
                )
        return matched

    def open_dimension_picker(self, image_id, wait_seconds=1):
        self._click_option(f"//img[@id='{image_id}']")
        time.sleep(wait_seconds)

    def select_output_format_prn(self):
        WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@name='formato' and @value='prn']"))
        ).click()

    def select_output_format_table(self):
        WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@name='formato' and @value='table']"))
        ).click()

    def submit_query(self):
        WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@class='mostra']"))
        ).click()
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def wait_for_result_page(self):
        WebDriverWait(self.driver, self.result_wait_seconds).until(
            lambda driver: driver.find_elements(By.XPATH, "//pre")
            or driver.find_elements(By.XPATH, "//h2")
            or driver.find_elements(By.XPATH, "//a[contains(@href, '.csv')]")
        )
        message_elements = self.driver.find_elements(By.XPATH, "//h2")
        if message_elements:
            message_text = " ".join(element.text.strip() for element in message_elements if element.text.strip())
            if "Nenhum registro selecionado" in message_text:
                raise RuntimeError(f"DATASUS returned no selected records: {message_text}")

    def read_result_table(self):
        self.wait_for_result_page()
        raw_text = self.driver.find_element(By.XPATH, "//pre").text
        try:
            return pd.read_csv(io.StringIO(raw_text), sep=";", encoding="latin1")
        except pd.errors.ParserError:
            return self._parse_result_table_with_csv_reader(raw_text)

    def download_result_csv(self, destination_path):
        self.wait_for_result_page()
        download_dir = Path(self.download_dir)
        before_files = {path.name for path in download_dir.glob("*.csv")}
        csv_link = WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '.csv') and contains(., 'CSV')]"))
        )
        csv_link.click()
        downloaded_path = self._wait_for_downloaded_csv(before_files)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_path), destination)
        return str(destination)

    def _wait_for_downloaded_csv(self, before_files):
        download_dir = Path(self.download_dir)
        deadline = time.time() + self.result_wait_seconds
        while time.time() < deadline:
            if list(download_dir.glob("*.crdownload")):
                time.sleep(0.5)
                continue

            new_files = sorted(
                path for path in download_dir.glob("*.csv") if path.name not in before_files
            )
            if new_files:
                return new_files[-1]
            time.sleep(0.5)
        raise TimeoutException("Timed out waiting for DATASUS CSV download.")

    def _parse_result_table_with_csv_reader(self, raw_text):
        rows = list(csv.reader(io.StringIO(raw_text), delimiter=";", quotechar='"'))
        if not rows:
            return pd.DataFrame()

        header = rows[0]
        normalized_rows = []
        expected_width = len(header)
        for row in rows[1:]:
            if len(row) < expected_width:
                row = row + [None] * (expected_width - len(row))
            elif len(row) > expected_width:
                row = row[: expected_width - 1] + [";".join(row[expected_width - 1 :])]
            normalized_rows.append(row)
        return pd.DataFrame(normalized_rows, columns=header)

    def reset_query(self):
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])
        time.sleep(self.reset_wait)
        self.driver.find_element(By.XPATH, "//input[@class='limpa']").click()

    def close(self):
        self.driver.quit()
        if self._owns_download_dir:
            shutil.rmtree(self.download_dir, ignore_errors=True)
