""" This whole script is used to preprocess the data and generate a unified data representation.
    The data will be stored via the following JSON format. If certain data does not exist the specific entry will contain an empty datastructure.
    {
        "id": str, # The assigned Aktenzeichen
        "date": str, # Format: YYYY-MM-DD
        "court": str, # Format: [type] [place] -> court type will always be the token before the first space
        "normchain": List[str], # Additional subsections might be added for each individual norm, most precise location in norm text is prefered
        "norms": List[placeholder], # All the norms used in the text with their specific placeholder found in the text
        "inst": List[str], # Contains the ids of the previous instances
        "keywords": List[str],
        "title": List[str],
        "guiding_principle": List[List[str]], # Contains the individual sentences from the guiding principle, we need this List-List structure as some verdicts have multiple GP
                                              # This list has two entries: first one are the sentences from the judge, second one are the sentences from the publisher
                                              # One/both can be empty list, which means they did not produce a guiding principle
        "tenor": List[str],
        "facts": List[str],
        "reasoning": List[str]
    }
    I choose to use a JSON Format, as some internet research suggested that it will be faster than using XML. The only advantage for XML is that you could directly
    embed the norms in the text, but then another extraction step would be necessary before the start of the training.
    For each source I need to create a different parser to get this information.
    The placeholder for the norm will be a global value in the following format: __norm###__ with ### replaced by a global identifier number. In the normchain no placeholder is
    used, as this makes it easier to retriev the used norms, in contrast to the text segments, where we want to embed the norms to vector representations.
"""
from pathlib import Path
import pickle
import os
import io
import re
from typing import List, Tuple
import json
from pprint import pprint
#import xml.etree.ElementTree as ET
from lxml import etree

from spacy.lang.de import German
from tqdm import tqdm

from sentences.sentence_segmenter import SentenceSegmenter

def process_nrw(source: Path, destination: Path, error_path: Path=Path("error_files")):
    """ Parses the verdicts from nrw into the JSON format """
    segmenter = SentenceSegmenter()
    nlp = German()
    nlp.add_pipe(segmenter)
    file_counter = get_file_counter(source)
    print(file_counter)
    # For debugging
    if source == Path("test_nrw_data"):
        file_counter[source] = 1
        
    files = [source/file for file in os.listdir(source)]

    print("Extracted files")

    # Restart from previous position
    for file in tqdm(files[file_counter[source]:], total=len(files)-file_counter[source]):
        # Other idea: Save the current process (how many files were processed etc.)
        # If anyting bad happens just throw an exception/use assertions 
        # Then we can investigate this file and further resume the processin
        html_string = read_file(file)
        
        parser = etree.HTMLParser()
        root = etree.fromstring(html_string, parser).find("body")
        divs = root.findall("div")
        divs = list(filter(lambda div: "class" in div.attrib and div.attrib["class"] == "maindiv" and len(div) > 0, divs))

        data = {
            "id": "",
            "date": "",
            "court": "", 
            "normchain": [], 
            "norms": [], 
            "inst": [], 
            "keywords": [],
            "title": [],
            "guiding_principle": [[], []], 
            "tenor": [],
            "facts": [],
            "reasoning": []
        }
        
        for div in divs:
            # Assumptions before processing: div has class "maindiv" and has at least one subelement
            # Gameplan: I think we can seperate the divs into two categories:
            #   - divs with "feldbezeichnung" and "feldinhalt"
            #   - divs with spans/p texts -> this should be the reasoning and facts
            # Assumption: No mixed divs -> TODO check this
            # 1. we need to differentiate between them
            # 2. write procedure for both of them
            # 2.1. should be staight forward via a state machine, which keeps track of the last "feldbezeichnung" and append the "feldinhalt" there
            # 2.2. we need to differentiate between the content -> facts and reasoning are seperated by a headline
            # 3. use generic method to extract text from p tags
            if div[0].tag == "div" and "class" in div[0].attrib and div[0].attrib["class"] == "feldbezeichnung":
                try:
                    new_data = process_div(div)
                    merge(data, new_data)
                except:
                    raise ValueError("Unknown div structure in: "+str(file)) 
            elif div[0].tag in ["span", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "div", "p", "br"]:
                try:
                    new_data = process_span(div)
                    merge(data, new_data)
                except:
                    raise ValueError("Unknown span structure in: "+str(file))   
            else:
                print(div[0].tag, div[0].attrib)
                raise ValueError("Unknown structure in: "+str(file))

        # Do sentence segmentation for all text segments
        data["guiding_principle"][0] = get_segment_sentences(nlp, data["guiding_principle"][0])
        data["guiding_principle"][1] = get_segment_sentences(nlp, data["guiding_principle"][1])

        # Some texts use non-breaking spaces for anonymization; we want to remove them
        data["tenor"] = list(map(lambda p: p.replace("\u00A0", ""), data["tenor"]))
        data["facts"] = list(map(lambda p: p.replace("\u00A0", ""), data["facts"]))
        data["reasoning"] = list(map(lambda p: p.replace("\u00A0", ""), data["reasoning"]))
        data["tenor"] = get_segment_sentences(nlp, data["tenor"])
        data["facts"] =  get_segment_sentences(nlp, data["facts"])
        data["reasoning"] = get_segment_sentences(nlp, data["reasoning"])   

        # For the files which we possibly need to check afterwards...
        if source != Path("test_nrw_data"): 
            if len(data["reasoning"]) == 0:
                with open("preprocessing_file.txt", "a", encoding="utf-8") as f:
                    f.write(str(file)+"\n")
                    print("File empty:" + str(file))         
        
        # Extract filename for saving file
        file_name = Path(file).name[:-len(".html")]
        save_json(data, destination/(file_name+".json")) 

        update_file_counter(source, file_counter)
            
def process_div(div: etree.Element) -> dict:
    """ Processes a div which consists only of divs with classes "feldbezeichnung" and "feldinhalt" """
    # Check that all subelements only have those two possible classes
    data = {}
    for elem in div:
        if elem.tag == "div" and "class" in elem.attrib and elem.attrib["class"] in ["feldbezeichnung", "feldinhalt"]:
            continue
        else:
            print(elem.tag, elem.attrib)
            raise ValueError("Div has wrong format.")

    # field will tell us which information is stored in the following fields. If we do not want to keep a "feldbezeichung", set it to "unimportant"
    field = None
    text2field = {
        "Datum": "date",
        "Gericht": "court",
        "Aktenzeichen": "id",
        "Schlagworte": "keywords",
        "Tenor": "tenor",
        "Vorinstanz": "inst",
        "Leitsätze": "guiding_principle",
        "Normen": "normchain",
        "Spruchkörper": "unimportant",
        "Sachgebiet": "unimportant",
        "Rechtskraft": "unimportant",
        "Nachinstanz": "unimportant",
        "Entscheidungsart": "unimportant",
        "ECLI": "unimportant"
    }
    for elem in div:
        if elem.attrib["class"] == "feldbezeichnung":
            if field is not None:
                raise ValueError("Wrong structure of divs...")
            for k in text2field:
                if elem.text.startswith(k):
                    field = text2field[k]
                    break
            if field is None:
                raise ValueError("Unknown field: "+elem.text)
        elif field is not None:
            # Individual data processing for all fields
            if field != "unimportant":
                if field == "date":
                    assert len(elem.text) == len("DD.MM.YYYY"), "Date has wrong format: "+elem.text
                    assert elem.text[0:2].isnumeric() and elem.text[2] == "." and elem.text[3:5].isnumeric() and elem.text[5] == "." and elem.text[6:].isnumeric(), "Date has wrong format: "+elem.text
                    data[field] = elem.text[6:]+"-"+elem.text[3:5]+"-"+elem.text[0:2]
                elif field == "court":
                    data[field] = elem.text
                elif field == "id":
                    data[field] = elem.text
                elif field == "tenor":
                    text = []
                    for p in elem.findall("p"):
                        text.append(nrw_process_paragraph(p))
                    if field not in data:
                        data[field] = text
                    else:
                        data[field].extend(text)
                elif field == "guiding_principle":
                    text = []
                    for p in elem.findall("p"):
                        text.append(nrw_process_paragraph(p))
                    
                    if len(text) > 0:
                        stored_gp = data.get(field, [[],[]])
                        last_sent = text[-1].strip().lower()
                        if "leitsatz" not in last_sent:
                            stored_gp[0].extend(text)
                        elif last_sent.endswith("Leitsatz vom Gericht"):
                            stored_gp[0].extend(text)
                        elif last_sent.endswith("kein leitsatz") or last_sent.endswith("kein leitsatz vorhanden"):
                            pass
                        elif "(redaktioneller leitsatz" in last_sent:
                            stored_gp[1].extend(text)
                        else:
                            raise ValueError("New guiding principle format"+text[-1])
                            stored_gp[1].extend(text)
                        
                        data[field] = stored_gp
                elif field == "normchain":
                    stored_nc = data.get(field, [])
                    stored_nc.append(elem.text)
                    data[field] = stored_nc
                elif field == "keywords":
                    if len(elem) > 0:
                        # We have some subelements
                        raise ValueError("Keywords wrong format")
                    else:
                        split = elem.text.splitlines()
                        words = []
                        for word in split:
                            words.append(word.strip())
                        words = list(filter(lambda word: len(word)>0, words))
                        data[field] = words
                elif field == "inst":
                    if len(elem) > 0:
                        # We have some subelements
                        raise ValueError("Instances wrong format")
                    else:
                        # Assumption: Before , we have the court name, afterwards the id -> we only want the id
                        split = elem.text.split(",")
                        if len(split) == 2:
                            temp_court = split[0].strip()
                            temp_id = split[1].strip()
                            # We are basically checking the possible id formats which are possible here -> id can be starting with numeric or mix of alphanumeric
                            if True:
                                assert temp_court.split(" ", maxsplit=1)[0].isalpha() and ((temp_id.split(" ", maxsplit=1)[0].isnumeric() or isroman(temp_id.split(" ", maxsplit=1)[0])) or 
                                    (temp_id.split(" ", maxsplit=1)[0].isalnum() and not temp_id.split(" ", maxsplit=1)[0].isalpha()) or
                                    (temp_id.split("-", maxsplit=1)[0].isalpha() and temp_id.split("-", maxsplit=1)[0].isupper()) or
                                    (temp_id.split("-", maxsplit=1)[0].isnumeric()) or
                                    (temp_id.split(" ", maxsplit=1)[0].isalpha() and temp_id.split(" ", maxsplit=1)[0].isupper()) or
                                    temp_id.split(" ", maxsplit=1)[0] in ["Ba", "BVGa", "Qu", "Ns", "StVK", "Vollz", "KLs", "LwG", "Lw", "GnR", "Ks", "Hagen", "StVG", "Qs", "Grundbuch"]), "Wrong previous instance "+ elem.text
                            text = split[1].strip()
                            if field not in data:
                                data[field] = [text]
                            else:
                                data[field].extend(text)
                        elif len(split) == 1 and (split[0].split(" ")[0].isnumeric() or isroman(split[0].split(" ")[0])):
                            text = split[0]
                            if field not in data:
                                data[field] = [text]
                            else:
                                data[field].extend(text)
                else:
                    raise ValueError("Unknown field: "+field)
            field = None
        else:
            raise ValueError("Wrong order of divs...")
    return data

def process_span(div: etree.Element) -> dict:
    """ Processes a div which contains the main text body of the verdict, i.e. contains the facts and reasoning """
    # Check that all subelements are spans
    tag_set = set(["span", "p", "ul", "ol", "li", "table", "tr",  "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "blockquote", "div", "img"])
    for elem in div:
        if elem.tag in tag_set:
            continue
        else:
            raise ValueError("Span has wrong format: "+elem.tag)
    # We extract the text from all the p elements and break based on the headlines "Gründe", "Tatbestand", "Entscheidungsgründe"
    # We might want to write paragarphs which only contain a single word into a file to manually check them
    facts = []
    reasoning = []
    current = reasoning
    for p in div:
        # We will ignore blockquotes, as they are anynomized
        if p.tag in ["span", "table", "br", "hr", "div", "img"]:
            continue

        text = nrw_process_paragraph(p)
        # We will now look at the first 42 characters (double the length of "Entscheidungsgründe" + buffer, as there are sometimes spaces between characters...)
        # to determine, if the paragraph contains a headline word
        temp = text[:42].lower().replace(" ", "")
        if temp.startswith("gründe") or temp.startswith("entscheidungsgründe"):
            current = reasoning
            continue
        elif temp.startswith("tatbestand"):
            current = facts
            continue

        current.append(text)
        if len(text.split(" ", maxsplit=1))==1 or len(text) < 5 or text[1] == " ":
            with open("short.text", "a", encoding="utf-8") as f:
                f.write(text)

    return {"facts": facts, "reasoning": reasoning}

def merge(data: dict, new_data: dict):
    """ Returns a merged variant of the both data entries """
    for k in new_data:
        if len(data[k]) == 0:
            data[k] = new_data[k]
        else:
            if k in ["id", "date", "court"]:
                raise ValueError("Multiple entries for "+k)
            elif k == "guiding_principle":
                data[k][0].extend(new_data[k][0])
                data[k][1].extend(new_data[k][1])
            else:
                data[k].extend(new_data[k])


def get_segment_sentences(nlp: German, segment: List[str]) -> List[str]:
    """ Do sentence segmentation on the segment """
    docs = nlp.pipe(segment)
    sentences = []
    for doc in docs:
        for s in doc.sents:
            sentences.append(s.text)
    return sentences

def nrw_process_paragraph(paragraph) -> str:
    """ Creates a continous string for the paragraph and replaces all the found norms with their placeholder 
    Returns:
        str -- continous version of the paragraph
    """
    # Use paragraph.iter() to iterate in document order over all subelements:
    texts = []
    
    for element in paragraph.iter():
        if element.tag in ["span", "table", "br", "hr", "blockquote"]:
            continue
        if element.text is not None:
            texts.append(element.text.strip())
        if element.tail is not None:
            texts.append(element.tail.strip())
    
    texts = " ".join(texts).strip()

    return texts

def save_json(dic, filepath):
    with io.open(filepath, "w+", encoding='utf-8') as f:
        json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

def read_file(path):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
                
    return text

def get_file_counter(source: Path):
    try:
        with open("file_counter_nrw.pkl", "rb") as f:
            # Counter is a dictionary that maps from each Path to its processed files number
            counter = pickle.load(f)
    except FileNotFoundError:
        counter = dict()

    if source not in counter:
        counter[source] = 0

    return counter

def update_file_counter(source, counter):
    counter[source] = counter[source] + 1
    with open("file_counter_nrw.pkl", "wb") as f:
        pickle.dump(counter, f)

def isroman(text: str) -> bool:
    """ Returns true only if text is a valid roman numeral """
    roman = set(['I','V','X','L','C','D','M','IV','IX','XL','XC','CD','CM'])
    i = 0
    while i < len(text):
        if i+1<len(text) and text[i:i+2] in roman:
            i+=2
        elif text[i] in roman:
            i+=1
        else:
            return False
    return True

if __name__ == "__main__":
    process_nrw(Path("nrw_data"), Path("processed_data_nrw"))
    #process_nrw(Path("test_nrw_data"), Path("test_nrw_json"))
