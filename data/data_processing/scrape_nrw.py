import requests
import zipfile
import io
import os
from pathlib import Path
from tqdm import tqdm
import time
import random
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import pickle

def read_nrw_url(url):
    html = requests.get(url)
    name = url.split("/")[-1]
    with io.open(Path("nrw_data")/name, "w", encoding="utf-8") as f:
        f.write(html.text)

def iterate_nrw():
    browser = webdriver.Firefox()
    browser.get('https://www.justiz.nrw.de/BS/nrwe2/index.php#solrNrwe')
    input("Start?")
    # Only after the start signal we will begin (so I can do the setup in the browser, i.e. accept cookies move to the right page)
    length = browser.find_element_by_id("anzahlGefunden")
    results, num_elems = int(length.text.split()[2]), 10
    num_pages = int(results/num_elems)
    try:
        with open(Path("Counter")/"nrw_counter.pkl", "rb") as f:
            # As we have already read the saved page +1 
            start_page = pickle.load(f)+1
    except FileNotFoundError:
        start_page = 1

    print(start_page)

    input("Go?")

    # We first need to move to the startpage
    """for page in range(1, start_page):
        try:
            actions = ActionChains(browser)
            page_button = browser.find_element_by_name("page"+str(page))
            actions.move_to_element(page_button)
            actions.click()
            actions.perform()
            time.sleep(0.3)
        except:
            time.sleep(1.2)
            continue"""

    for page in tqdm(range(start_page, num_pages), total=num_pages-start_page+1):
        actions = ActionChains(browser)
        page_button = browser.find_element_by_name("page"+str(page))
        actions.move_to_element(page_button)
        actions.click()
        actions.perform()

        results = browser.find_elements_by_class_name("einErgebnis")
        links = []
        for result in results:
            link = result.find_element_by_tag_name("a").get_attribute("href")
            links.append(link)

        for link in links:
            read_nrw_url(link)
            random_delay = random.random()/8
            time.sleep(0.1+random_delay)

        with open(Path("counter")/"nrw_counter.pkl", "wb") as f:
            pickle.dump(page, f)


if __name__ == "__main__":
    iterate_nrw()
