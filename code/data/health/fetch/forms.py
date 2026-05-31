import io
import time

import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..webdriver import create_chrome_driver


class DatasusTabnetForm:
    """Small Selenium wrapper around common DATASUS TABNET form interactions."""

    def __init__(self, headless=False, download_dir=None, page_load_wait=3, reset_wait=2):
        self.driver = create_chrome_driver(headless=headless, download_dir=download_dir)
        self.page_load_wait = page_load_wait
        self.reset_wait = reset_wait

    def open(self, url):
        self.driver.get(url)
        time.sleep(self.page_load_wait)

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

    def select_content_text(self, text, exact=False):
        for select_name in ["Conteudo", "Incremento"]:
            try:
                self._click_option(self._option_text_xpath(select_name, text, exact=exact))
                return
            except NoSuchElementException:
                continue
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
        last_error = None
        for select_name in select_names:
            try:
                self._click_option(self._option_text_xpath(select_name, text, exact=exact))
                return select_name
            except NoSuchElementException as exc:
                last_error = exc
        if exact:
            self.select_option_with_exact_text(text)
        else:
            self.select_option_with_text(text)
        return None

    def select_filter_option_texts(self, select_names, texts, exact=False, pause_seconds=0.2):
        for text in texts:
            self.select_filter_option_text(select_names, text, exact=exact)
            time.sleep(pause_seconds)

    def get_select_options(self, select_name):
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
        script = """
            const select = document.querySelector(`select[name="${arguments[0]}"]`);
            if (!select) return false;
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
            return true;
        """
        found = self.driver.execute_script(script, select_name, values, clear_first)
        if not found:
            raise NoSuchElementException(f"Could not find select {select_name!r}")

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

    def open_dimension_picker(self, image_id, wait_seconds=1):
        self._click_option(f"//img[@id='{image_id}']")
        time.sleep(wait_seconds)

    def select_output_format_prn(self):
        self._click_option("//input[@name='formato' and @value='prn']")

    def submit_query(self):
        self.driver.find_element(By.XPATH, "//input[@class='mostra']").click()
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def read_result_table(self):
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
        raw_text = self.driver.find_element(By.XPATH, "//pre").text
        return pd.read_csv(io.StringIO(raw_text), sep=";", encoding="latin1")

    def reset_query(self):
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])
        time.sleep(self.reset_wait)
        self.driver.find_element(By.XPATH, "//input[@class='limpa']").click()

    def close(self):
        self.driver.quit()
