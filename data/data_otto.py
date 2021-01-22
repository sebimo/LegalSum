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
import xml.etree.ElementTree as ET

from spacy.lang.de import German
from tqdm import tqdm

from sentences.sentence_segmenter import SentenceSegmenter
from normdb import NormDatabase


class Norm:
    """ Wrapper class used to build up the text of xml nodes """

    def __init__(self, element: ET.Element, position: int):
        self.norms = [element]
        self.position = position

    def append(self, element: ET.Element):
        self.norms.append(element) 

    def finalize(self) -> Tuple[int, List[str]]:
        """ Returns (position, norm_refs) """
        # a norm always consists of two parts, i.e. we need to check which we are in. Additionally: each norm starts with the law (book) followed by the number and subnumbers etc.
        # Example: GG § 3, §§ 3,4 GG -> [GG § 3, GG § 4]
        # We want to normalize the norms, as this will make it easier later on, when we want to retrieve them from their source document
        refs = []
        for norm in self.norms:
            assert "PUBKUERZEL" in norm.attrib
            if "§" in norm.text:
                with open("preprocessing_norm.txt", "a", encoding="utf-8") as f:
                    f.write(norm.text+"\n")
                print("Norm has wrong format with §:" + norm.text)
                refs.append(norm.text.strip())
                continue
            # The last part of the text will always be a the norm if it is a word
            # if the last part of the norm is not a word, we will use the value from "PUBKUERZEL" instead which is always nonempty
            
            parts = norm.text.split()

            # We have to check, whether we are actually looking at a norm vs. some kind of Verordnunge etc.:
            # We can we do this: Verordnungen or Richtlininien are referenced by a mixture of numbers, alphabetic characters and slashes
            # -> they will not fulfill the second part of the if-statement
            if len(parts) > 0 and not (parts[-1].isalpha() or parts[-1].isnumeric()):
                # In this case we cannot really differiate between single and multiple norms ($ 3, 4 <-> $ 3), because we do not know what is the norm part which needs replication
                if parts[-1] != "ff.":
                    refs.append(" ".join(parts).strip())
                    continue

            norm_text = []
            found_normal_text = False
            for part in reversed(parts):
                if part.isalpha():
                    # We will only append roman numerals if there was no text before them
                    if isroman(part) and found_normal_text:
                        break
                    else:
                        norm_text.insert(0, part)
                        found_normal_text = True
                else:
                    break

            # Did we append roman numerals, but there is no norm reference found?
            if not found_normal_text and len(norm_text) > 0:
                norm_text = []
            
            # Have we even found a norm?
            if len(norm_text) > 0:
                # We need to check whether norm_text is the complete text or if there are still some parts missing
                if len(norm_text) < len(parts):
                    parts = norm_text + ["§"] + parts[:-len(norm_text)]
                else:
                    parts = norm_text
            else:
                parts.insert(0, norm.attrib["PUBKUERZEL"])
                parts.insert(1, "§")

            # Potentially splitting multi-norms:
            norm_text = " ".join(parts).strip()
            norm_split = norm_text.split("§")
            assert len(norm_split) == 2

            norm = norm_split[0].strip()
            sub_norm = norm_split[1].strip()
            sub_norm_split = sub_norm.split(",")

            norm_texts = []
            for sn in sub_norm_split:
                norm_texts.append(norm + " § " + sn.strip())

            refs.extend(norm_texts)
        
        return self.position, refs


def process_otto(source: Path, destination: Path, error_path: Path=Path("error_files")):
    """ Parses the verdicts from gesetze-bayern.de into the JSON format """
    #xsd = ValT.parse(Path("xsd")/"bayern.xsd")
    #schema = ValT.XMLSchema(xsd)
    if source == Path("test_otto_data"):
        normDB = NormDatabase(Path("norms_test.db"), create=False)
    else:
        normDB = NormDatabase(Path("norms.db"), create=False)
    segmenter = SentenceSegmenter()
    nlp = German()
    nlp.add_pipe(segmenter)
    file_counter = get_file_counter(source)
    print(file_counter)
    # For debugging
    if source == Path("test_otto_data"):
        file_counter[source] = 0
        files = [source/file for file in os.listdir(source)]
    else:
        files = []
        folders = ["OttoSchmidt_BGHV_2017-2020", "OttoSchmidt_Rechtsprechung_2010-2013", "OttoSchmidt_Rechtsprechung_2014-2016"]
        #folders = ["OttoSchmidt_BGHV_2017-2020"]

        # TODO change this path to your own OttoSchmidt path
        source = Path("..")/".."/"HiWi"/"Urteile"

        for folder in folders:
            filepath = source/folder
            for root, _, filenames in os.walk(filepath, topdown=True):
                for filename in filenames:
                    if os.path.basename(filename).endswith("xml"):
                        xml_filename = Path(root)/filename
                        files.append(xml_filename)

    print("Extracted files")

    # Restart from previous position
    for file in tqdm(files[file_counter[source]:], total=len(files)-file_counter[source]):
        # Other idea: Save the current process (how many files were processed etc.)
        # If anyting bad happens just throw an exception/use assertions 
        # Then we can investigate this file and further resume the processin
        xml_string = read_file(file)
        
        root = ET.fromstring(xml_string).find("BEITRAG/ENTSCHEIDUNG")

        # For the extraction of the smaller fields I decided against using additional functions as most of them are oneliners
        id = root.attrib["AZ"]
        date = root.attrib["DATUM"]
        if len(date) != 8:
            raise RuntimeError("Date has wrong format: "+ date)
        else:
            date = date[:4]+"-"+date[4:6]+"-"+date[6:]
        court = root.attrib["BEHOERDE"]

        try:
            normchain = otto_normchain(root.findall("NORM"))
        except AttributeError:
            print(file)
            raise ValueError("File is not correctly build.")
        found_norms = [normDB.register_norm(norm) for norm in normchain]
        normchain = found_norms.copy()
        keywords = [keyword.text for keyword in root.findall("META/SCHLAGWORT")]
        inst = [inst.text for inst in root.findall("VORINSTANZ/AKTENZEICHEN")]
        
        guiding_principle_paragraphs = []
        for segment in root.findall("LEITSATZ"):
            text, norms = otto_process_paragraph(segment, normDB)
            if len(text) > 0 and not text[0].startswith("Leitsatz nicht"):
                guiding_principle_paragraphs.extend(text)
                found_norms.extend(norms)

        tenor_paragraphs = []
        for segment in root.findall("ENTSCHEIDUNGSFORMEL"):
            text, norms = otto_process_paragraph(segment, normDB)
            tenor_paragraphs.extend(text)
            found_norms.extend(norms)
        
        facts_paragraphs = []
        for segment in root.findall("TATBESTAND"):
            text, norms = otto_process_paragraph(segment, normDB)
            facts_paragraphs.extend(text)
            found_norms.extend(norms)

        reasoning_paragraphs = []
        for segment in root.findall("GRUENDE"):
            text, norms = otto_process_paragraph(segment, normDB)
            reasoning_paragraphs.extend(text)
            found_norms.extend(norms)

        #assert len(facts_paragraphs) > 0 or len(reasoning_paragraphs) > 0
        #assert len(reasoning_paragraphs) > 0
        if len(reasoning_paragraphs) == 0:
            with open("preprocessing_file.txt", "a", encoding="utf-8") as f:
                f.write(str(file)+"\n")
                print("File empty:" + str(file))

        title_paragraphs = []

        # TODO Look if this is necessary
        """if len(facts_paragraphs) == 0:
            reasoning_paragraphs = []
            append_list = facts_paragraphs
            for paragraph in paragraph_list:
                if paragraph in ["II.", "B.", "III."]:
                    append_list = reasoning_paragraphs
                append_list.append(paragraph)
            if len(reasoning_paragraphs) == 0:
                reasoning_paragraphs = facts_paragraphs
                facts_paragraphs = []               
        else:
            reasoning_paragraphs = paragraph_list"""
        
        norms = list(set(found_norms))
        norms = {norm: normDB.placeholder2norm(norm) for norm in norms}

        # Possible segmentation of guiding principle into amtlich (from a judge) and redaktionell (from a publisher)
        # Criterion is whether they end with "(amtlicher Leitsatz)" or "(redaktioneller Leitsatz)"
        # If no annotation, we assume it is a guiding principle by a judge
        # Currently I have found one document with "amtlicher Leitsatz", i.e. we need to do the segmentation
        gp_judge = []
        gp_publisher = []
        for gp in guiding_principle_paragraphs:
            if gp.endswith("(amtlicher Leitsatz)"):
                gp_judge.append(gp[:-len("(amtlicher Leitsatz)")].strip())
            elif gp.endswith("(redaktioneller Leitsatz)"):
                gp_publisher.append(gp[:-len("(redaktioneller Leitsatz)")].strip())
            elif gp.endswith("Leitsatz)"):
                print(file)
                raise ValueError("We need to segment the guiding principle...")
            else:
                gp_judge.append(gp)


        # Do sentence segmentation for all text segments
        title_paragraphs = get_segment_sentences(nlp, title_paragraphs)
        gp_judge = get_segment_sentences(nlp, gp_judge)
        gp_publisher = get_segment_sentences(nlp, gp_publisher)
        tenor_paragraphs = get_segment_sentences(nlp, tenor_paragraphs)
        facts_paragraphs =  get_segment_sentences(nlp, facts_paragraphs)
        reasoning_paragraphs = get_segment_sentences(nlp, reasoning_paragraphs)            

        data = {
            "id": id,
            "date": date,
            "court": court, 
            "normchain": normchain, 
            "norms": norms, 
            "inst": inst, 
            "keywords": keywords,
            "title": title_paragraphs,
            "guiding_principle": [gp_judge, gp_publisher], 
            "tenor": tenor_paragraphs,
            "facts": facts_paragraphs,
            "reasoning": reasoning_paragraphs
        }
        
        # Extract filename for saving file
        file_name = Path(file).name[:-len(".xml")]
        save_json(data, destination/(file_name+".json")) 

        update_file_counter(source, file_counter)
            
def get_segment_sentences(nlp: German, segment: List[str]) -> List[str]:
    """ Do sentence segmentation on the segment """
    docs = nlp.pipe(segment)
    sentences = []
    for doc in docs:
        for s in doc.sents:
            sentences.append(s.text)
    return sentences

def otto_process_paragraph(paragraph, normDB: NormDatabase) -> (str, List[str]):
    """ Creates a continous string for the paragraph and replaces all the found norms with their placeholder 
    Returns:
        str -- continous version of the paragraph
        List[str] -- the placeholders of all found norms
    """
    # Use paragraph.iter() to iterate in document order over all subelements:
    texts = []
    current_text = []
    norms = []
    current_norms = None
    
    for element in paragraph.iter():
        # Each paragraph needs to be seperated from one another and then recursively added to the current_text (depending on subelements)
        if element.tag in ["P", "UL"]:
            if len(current_text) > 0:
                insert_norm(current_text, current_norms, norms, normDB)
                current_norms = None

                texts.append(" ".join(current_text).strip())
                current_text = []

        # When do we save the norm?
        # 1. As soon as we get the first tag, which is not a norm
        # 2. If we get another "VERWEIS-GS" and we already have a current_norms -> case if there are two norms directly following each other
        # We will have a problem if there are multiple VERWEIS-GS after another seperated with commas with reference the same law
        # Plan accumulate "VERWEIS-GS" until we find another tag or the tail is not ","?
        if element.tag == "VERWEIS-GS":
            if current_norms is None:
                current_norms = Norm(element, len(current_text))
            else:
                current_norms.append(element)
        elif element.tag in ["RZ", "TITEL", "LI"]:
            insert_norm(current_text, current_norms, norms, normDB)
            current_norms = None

            continue
        elif element.text is not None:
            # We will remove all spaces and join on them later on, as we can remove some possible edgecases that way
            insert_norm(current_text, current_norms, norms, normDB)
            current_norms = None

            insert_text = element.text.strip()
            if len(insert_text) > 0:
                current_text.append(insert_text)
        
        if element.tail is not None and not (element.tag == "VERWEIS-GS" and element.tail.strip() == ","):
            insert_norm(current_text, current_norms, norms, normDB)
            current_norms = None
            
            insert_text = element.tail.strip()
            if len(insert_text) > 0:
                current_text.append(element.tail.strip())

    if len(current_text) > 0:
        insert_norm(current_text, current_norms, norms, normDB)
        current_norms = None

        texts.append(" ".join(current_text).strip())

    norms = [norm.strip() for norm in norms]

    return (texts, norms)

def seperate_norms(norm_text: str) -> List[str]:
    """ Splits up the norm string into all its individual norms.
    Examples: "§ 3" -> ["§ 3"], "§§ 3, 4" -> ["§ 3", "§ 4"]
    Returns:
        List[str] -- all individual norms
    """
    # We only replace the first occurence
    norm_text = norm_text.replace("§§", "§", 1)
    norms = norm_text.split(",")
    processed_norms = [norms[0]]
    for norm in norms[1:]:
        processed_norms.append("§"+norm)
    for norm in processed_norms:
        assert norm.startswith("§ "), "Norm in wrong format:"+norm
    return processed_norms

def insert_norm(text: List[str], current_norms: Norm, norms: List[str], normDB: NormDatabase):
    """ We will insert all the norms at the specific position """
    if current_norms is not None:
        position, norm_refs = current_norms.finalize()
        norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
        norms.extend(norm_refs)

        norm_text = " ".join(norm_refs).strip() 
        text.insert(position, norm_text)

def otto_normchain(norm_nodes) -> List[str]:
    normchain = []
    for norm_node in norm_nodes:
        # We are saving the placeholders, not the norms!
        text = norm_node.text
        text = text.replace("§§", "§")
        if "," in text:
            norm_split = [t.strip() for t in text.split("§")]
            norm_split = [t for t in norm_split if len(t) > 0]
            if len(norm_split) == 2:
                subnorm_split = [t.strip() for t in norm_split[1].split(",")]
                subnorm_split = [t for t in subnorm_split if len(t) > 0]  
                for subnorm in subnorm_split:
                    normchain.append(norm_split[0] + " § " + subnorm)      
            else:
                with open("preprocessing_norm.txt", "a", encoding="utf-8") as f:
                    f.write(text+"\n")
                print("Normchain has wrong format:" + text)
        else:
            if len(text) > 0:
                normchain.append(text)
    return normchain

def save_json(dic, filepath):
    with io.open(filepath, "w+", encoding='utf-8') as f:
        json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

def read_file(path):
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
                
    return text

def get_file_counter(source: Path):
    try:
        with open("file_counter_otto.pkl", "rb") as f:
            # Counter is a dictionary that maps from each Path to its processed files number
            counter = pickle.load(f)
    except FileNotFoundError:
        counter = dict()

    if source not in counter:
        counter[source] = 0

    return counter

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

def update_file_counter(source, counter):
    counter[source] = counter[source] + 1
    with open("file_counter_otto.pkl", "wb") as f:
        pickle.dump(counter, f)

if __name__ == "__main__":
    process_otto(Path("..")/"HiWi"/"Urteile", Path("processed_data_otto"))
    #process_otto(Path("test_otto_data"), Path("test_otto_json"))
