import requests
import zipfile
import io
import os
from pathlib import Path
from tqdm import tqdm
import time
import random
from selenium import webdriver
import pickle

def read_gb_url(url):
    # Extracts the xml verdict from gesetze-bayern.de, if a /Content/Zip/... url is given
    if not url.startswith("https://www.gesetze-bayern.de/Content/Zip/"):
        raise ValueError("Wrong URL name. Should start with https://www.gesetze-bayern.de/Content/Zip/")
    html = requests.get(url)
    try:
        zip_file = io.BytesIO(html.content)
        with zipfile.ZipFile(zip_file) as zip:
            for zip_info in zip.infolist():
                # bayportalrsp is the folder with the xml file of the verdict
                if zip_info.filename.startswith("bayportalrsp"):
                    save_path = Path("data")/os.path.basename(zip_info.filename)
                    file = zip.read(zip_info)
                    with open(save_path, "wb") as f:
                        f.write(file)
    except zipfile.BadZipFile:
        print(url, "not working...")

def is_document(link):
    start = link.startswith("https://www.gesetze-bayern.de/Content/Document/")
    end = link.endswith("?hl=true")
    return start and end

def convert_to_documentlinks():
    with open("links.pkl", "rb") as f:
        links = pickle.load(f)
    docs = []
    for link in links:
        start_length = len("https://www.gesetze-bayern.de/Content/Document/")
        end_length = len("?hl=true")
        docs.append("https://www.gesetze-bayern.de/Content/Zip/"+link[start_length:-end_length])
    print(len(docs))
    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

def iterate_gb():
    browser = webdriver.Firefox()
    browser.get('https://www.gesetze-bayern.de/')
    link_list = []
    input("Start?")
    # Only after the start signal we will begin (so I can do the setup in the browser, i.e. accept cookies move to the right page)
    elems = browser.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        link = elem.get_attribute("href")
        if is_document(link):
            link_list.append(link)
    max_page = 3588
    for i in range(2, max_page+1):
        browser.get('https://www.gesetze-bayern.de/Search/Page/'+str(i))
        elems = browser.find_elements_by_xpath("//a[@href]")
        for elem in elems:
            link = elem.get_attribute("href")
            if is_document(link):
                link_list.append(link)

    with open("links.pkl", "wb") as f:
        pickle.dump(link_list, f)

def scrape():
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)
    
    # max_iter is set, so we are not crawling the entire database and only 75% of the verdicts
    max_iter = int(len(docs)*0.75)
    i = 0
    with open("counter.pkl", "rb") as f:
        i = pickle.load(f)
    for link in tqdm(docs[i:]):
        with open("counter.pkl", "wb") as f:
            pickle.dump(i,f)
        delay = random.random()/4
        time.sleep(0.5+delay)
        read_gb_url(link)
        i += 1


if __name__ == "__main__":
    #iterate_gb()
    #convert_to_documentlinks()
    scrape()
