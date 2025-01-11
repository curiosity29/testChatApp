
# from langchain_core.tools import tool

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime


def select_date(driver, date, next_month_id_condition, date_condition, current_month = 0):
    """Args: 
            date: string of format 'yyyy-mm-dd'
            date_condition: string format with parameter date with input date (first input)
        Return:
            current month of datepicker
    """
    target_month = int(date.split("-")[1])
    if not current_month:
        current_month = int(datetime.now().month)
    next_count = target_month - current_month
    # print("next for " + str(next_count) + " times")
    for _ in range(next_count):
        time.sleep(0.5)
        driver.find_element(By.CSS_SELECTOR, next_month_id_condition).click()
    # print("finding with "+ date_condition.format(date = date))
    driver.find_element(By.CSS_SELECTOR, date_condition.format(date = date)).click()
    return target_month

# @tool
def get_price_by_dates(first_date, second_date):
    """Get the availability and price for each person (if available) of the hotel, using a date range from first_date to second_date.
        Inputs must be in type string of the format 'yyyy-mm-dd'.
        Output string is either a message that says the booked date is not available or there are error, or if it is then output a string consists of type of currency and its value then the number of people, e.g. 'HK$ 3 670 for 1 people' is 3670 Hong Kong dollar.
        Don't call this multiple times using the same input dates.
        Args:
            first_date: the start date of the booking period
            second_date: the end date of the booking period
        Return:
            dictionary with format:
            {
                available: bool,
                price: string
            }
        
    """
    # testing
    # return "VND 12 230 000 for 1 people"
    try:
        ##      Web structure id
        date_time_picker_id_condition = "[data-testid='searchbox-dates-container']"
        next_month_id_condition = "[aria-label='Следующий месяц']"
        # submit_class_id = "e4adce92df"
        date_condition = "[data-date='{date}']"
        price_class_name = "prco-valign-middle-helper"

        ##      Brow server
        # service = Service(executable_path="chromedriver.exe")
        # driver = webdriver.Chrome(service=service)
        driver = webdriver.Firefox()
        root_link = "https://www.booking.com/hotel/sk/palace-art-pezinok.ru.html"
        driver.get(root_link)
        driver.maximize_window()
        print("################                     opening datetime picker")
        ##      Open datetime picker
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0,1411.3333740234375)")
        time.sleep(0.5)
        driver.find_elements(By.CSS_SELECTOR, date_time_picker_id_condition)[1].click()
        # current_month = 1
        print("################                     picking date")
        time.sleep(0.5)
        ##      pick date_range
        current_month = select_date(driver, first_date, next_month_id_condition=next_month_id_condition, date_condition=date_condition)
        select_date(driver, second_date, next_month_id_condition=next_month_id_condition, date_condition=date_condition, current_month=current_month)

        ##      Submit
        time.sleep(0.5)
        print("################                     submit")
        # driver.find_element(By.CLASS_NAME, submit_class_id)
        driver.find_element(By.CSS_SELECTOR, ".a4c1805887 > .e4adce92df:nth-child(2)").click()
        time.sleep(0.5)
        driver.get(root_link)
        ##      Get the first price data price data
        price_text = driver.find_element(By.CLASS_NAME, price_class_name).text + " for 1 people" 
        print("################                     get price text: ", price_text)
    except Exception as e:
        print(e)
        ##  Price not found (not available)
        price_text = "The booked date range is not available or there are error"
    # driver.find_element(By.CSS_SELECTOR, "[aria-label={date}]".format(date = first_date)).click()
    # driver.find_element(By.CSS_SELECTOR, "[aria-label={date}]".format(date = second_date)).click()
    # print("searched price: ", price_text)
    driver.close()
    return price_text




if __name__ == "__main__":
    first_date = '2025-05-30'
    second_date = '2025-06-01'
    get_price_by_dates(first_date, second_date)