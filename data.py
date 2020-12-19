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
#from lxml import etree as ValT

from spacy.lang.de import German
from tqdm import tqdm

from sentences.sentence_segmenter import SentenceSegmenter


class NormDatabase:
    """ Class keeping track of the placeholder: norm association """
    def __init__(self, path: Path, create: bool = False):
        if not create:
            with open(path, "rb") as f:
                self.norm2id, self.id2norm, self.max_id = pickle.load(f)
        else:
            raise ValueError("Created new norm database!!!")
            self.norm2id, self.id2norm, self.max_id = dict(), dict(), 0
        self.path = path

    def register_norm(self, norm):
        """ Makes a lookup for the norm (, creates a new id if necessary) and returns its id placeholder """
        if norm in self.norm2id:
            return self.norm2id[norm]
        else:
            self.max_id += 1
            placeholder = "__norm"+str(self.max_id)+"__"
            self.norm2id[norm] = placeholder
            self.id2norm[self.max_id] = norm
            return placeholder

    def placeholder2norm(self, placeholder):
        """ Converts a placeholder back to its initial norm """
        assert placeholder.startswith("__norm")
        assert placeholder.endswith("__")
        id = int(placeholder[len("__norm"):-len("__")])
        return self.id2norm[id]

    def __del__(self):
        print("Closing NormDatabase...")
        with open(self.path, "wb") as f:
            pickle.dump((self.norm2id, self.id2norm, self.max_id), f)


class Norm:
    """ Wrapper class used to build up the text of xml nodes """

    def __init__(self, element: ET.Element, position: int):
        # Assumption: We only have one abr per norm! -> every new norm will be displayed as a seperate tag in the xml
        # The outer element -> always save (text, tail), except for self.norm, as its tail belongs to the norm text
        self.norm = None
        # the abbreviation of the norm
        self.abr = (None, None)
        # reference for a specific norm paragraph -> we might have multiple of those
        self.num = []
        self.position = position
        self.process(element)

    def process(self, element: ET.Element):
        if element.tag == "verweis.norm":
            self.norm = element.text
        elif element.tag == "v.abk":
            self.abr = (element.text, element.tail)    
        elif element.tag == "v.norm":
            self.num.append((element.text, element.tail))
        else:
            raise ValueError("Wrong element with tag: "+element.tag)  

    def finalize(self) -> Tuple[int, List[str]]:
        """ Returns (position, norm_refs) """
        # a norm always consists of two parts, i.e. we need to check which we are in. Additionally: each norm starts with the law (book) followed by the number and subnumbers etc.
        # Example: GG § 3, §§ 3,4 GG -> [GG § 3, GG § 4]
        # We want to normalize the norms, as this will make it easier later on, when we want to retrieve them from their source document
        if self.abr[0] is None:
            print(self.norm, self.abr, self.num)
            raise ValueError("Norm is wrongly build")

        abr = self.abr[0].strip()
        followup = " " + self.abr[1].strip() if self.abr[1] is not None else ""

        text = self.norm if self.norm is not None else ""
        for norm in self.num:
            if norm[0] is not None:
                text += norm[0]
            if norm[1] is not None:
                text += norm[1]

        text = text.replace("§", "").strip()
        norm_list = [s.strip() for s in text.split(",")]

        refs = [abr + " § " + norm + followup for norm in norm_list]
        
        return self.position, refs


def process_ges_bay(source: Path, destination: Path):
    """ Parses the verdicts from gesetze-bayern.de into the JSON format """
    #xsd = ValT.parse(Path("xsd")/"bayern.xsd")
    #schema = ValT.XMLSchema(xsd)
    normDB = NormDatabase(Path("norms.db"), create=False)
    segmenter = SentenceSegmenter()
    nlp = German()
    nlp.add_pipe(segmenter)
    file_counter = get_file_counter(source)
    print(file_counter)
    # For debugging
    """if source == Path("test_data"):
        file_counter[source] = 0"""
    
    # Restart from previous position
    files = os.listdir(source)
    for file in tqdm(files[file_counter[source]:], total=len(files)-file_counter[source]):
        # We only want the original Beck format here
        #print(file)
        if not file.startswith("Y"):
            continue
        # Validate the schema of the file against the xsd
        #test_root = ValT.parse(source/file)
        #print(schema.validate(test_root))
        #
        # Other idea: Save the current process (how many files were processed etc.)
        # If anyting bad happens just throw an exception/use assertions 
        # Then we can investigate this file and further resume the processin

        xml_string = read_file(source/file)

        root = ET.fromstring(xml_string)
        meta = root.find("metadaten")

        # For the extraction of the smaller fields I decided against using additional functions as most of them are oneliners
        id = meta.find("aktenzeichen").text
        date = meta.find("entsch-datum").text
        court_node = meta.find("gericht")
        court = court_node.find("gertyp").text + " " + court_node.find("gerort").text
        normchain = ges_bay_normchain(meta.findall("norm"))
        found_norms = [normDB.register_norm(norm) for norm in normchain]
        normchain = found_norms.copy()

        keywords = [keyword.text for keyword in meta.findall("schlagwort")]
        inst = [inst.text for inst in meta.findall("vorinstanz/az")]

        text = root.find("textdaten")
        subsegments = text.findall("./")
        # We will seperate the aggregation of segments and their processing, i.e. first get all relevant text segments and their norms
        # Later segment them into segments in parallel. That means parsing the XML and the sentence segmentation are decoupled.
        tenor_paragraphs = []
        guiding_principle_paragraphs = []
        facts_paragraphs = []
        paragraph_list = []
        title_paragraphs = []
        for segment in subsegments:
            if segment.tag == "tenor":
                for paragraph in segment.findall("body/div/p"):
                    text, norms = ges_bay_process_paragraph(paragraph, normDB)
                    # There is one headline in the paragraphs:
                    if text != "Beschluss":
                        tenor_paragraphs.extend(text)
                        found_norms.extend(norms)
            elif segment.tag == "leitsatz":
                for paragraph in segment.findall("body/div/p"):
                    texts, norms = ges_bay_process_paragraph(paragraph, normDB)
                    # We have to remove the (Rn. xx) at the end, if its there
                    # Is it possible that we reference a norm here?
                    texts = clean_gp(texts)
                    guiding_principle_paragraphs.extend(texts)
                    found_norms.extend(norms)
            elif segment.tag == "gruende":
                # Here we have to be careful, as we still need to split facts and reasoning by the roman numerals I. and II. 
                for paragraph in segment.findall("body/div/p"):                    
                    texts, norms = ges_bay_process_paragraph(paragraph, normDB)
                    paragraph_list.extend(texts)
                    found_norms.extend(norms)
            elif segment.tag == "tatbestand":
                for paragraph in segment.findall("body/div/p"):
                    texts, norms = ges_bay_process_paragraph(paragraph, normDB)
                    facts_paragraphs.extend(texts)
                    found_norms.extend(norms) 
            elif segment.tag == "titelzeile":
                for paragraph in segment.findall("body/div/p"):
                    texts, norms = ges_bay_process_paragraph(paragraph, normDB)
                    title_paragraphs.extend(texts)
                    found_norms.extend(norms)
            elif segment.tag == "sonstosatz":
                for paragraph in segment.findall("body/div/p"):
                    texts, norms = ges_bay_process_paragraph(paragraph, normDB)
                    # We have to remove the (Rn. xx) at the end, if its there
                    # Is it possible that we reference a norm here?
                    texts = clean_gp(texts)
                    guiding_principle_paragraphs.extend(texts)
                    found_norms.extend(norms)
            elif segment.tag == "kurztext-land":
                pass 
            else:
                raise ValueError("Encountered unkown text segment in "+str(file)+": "+segment.tag)

        if len(facts_paragraphs) == 0:
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
            reasoning_paragraphs = paragraph_list
        
        norms = list(set(found_norms))
        norms = {norm: normDB.placeholder2norm(norm) for norm in norms}

        # Possible segmentation of guiding principle into amtlich (from a judge) and redaktionell (from a publisher)
        # Criterion is whether they end with "(amtlicher Leitsatz)" or "(redaktioneller Leitsatz)"
        # If no annotation, we assume it is a guiding principle by a judge
        gp_judge = []
        gp_publisher = []
        for gp in guiding_principle_paragraphs:
            if gp.endswith("(amtlicher Leitsatz)"):
                gp_judge.append(gp[:-len("(amtlicher Leitsatz)")].strip())
            elif gp.endswith("(redaktioneller Leitsatz)"):
                gp_publisher.append(gp[:-len("(redaktioneller Leitsatz)")].strip())
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
        file_name = file[:-len(".xml")]
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

def clean_gp(texts: List[str]) -> str:
    """ Removes following info from the guiding principle text:
        - (Rn. xx) at the end
    """
    # \Z stands for: match only at the end of the string
    processed_str = []
    for text in texts:
        m = re.search(r"\(Rn.\s[0-9]+\)\s*\Z", text)
        if m:
            processed_str.append(text[:m.start()])
        else:
            processed_str.append(text)
    return processed_str

def ges_bay_process_paragraph(paragraph, normDB: NormDatabase) -> (str, List[str]):
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
    # Flag to track, whether we are currently handling a reference to a different court ruling
    rspr = False
    
    for element in paragraph.iter():
        #print(element.tag, element.text, element.tail)
        # Each paragraph needs to be seperated from one another and then recursively added to the current_text (depending on subelements)
        if element.tag == "p":
            rspr = False
            if len(current_text) > 0:
                if current_norms is not None:
                    position, norm_refs = current_norms.finalize()
                    norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
                    norms.extend(norm_refs)

                    norm_text = " ".join(norm_refs).strip() 
                    insert_norm(current_text, norm_text, position)

                    current_norms = None

                texts.append(" ".join(current_text))
                current_text = []

        # TODO we need to look at <verweis.rspr>
        # When do we save the norm?
        # 1. As soon as we get the first tag, which is not a norm
        # 2. If we get another "verweis.norm" and we already have a current_norms -> case if there are two norms directly following each other
        if element.tag in ["verweis.norm", "v.abk", "v.norm"]:
            rspr = False
            if current_norms is None:
                assert element.tag == "verweis.norm"
                current_norms = Norm(element, len(current_text))
            elif element.tag == "verweis.norm":
                position, norm_refs = current_norms.finalize()
                norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
                norms.extend(norm_refs)

                norm_text = " ".join(norm_refs).strip() 
                insert_norm(current_text, norm_text, position)
                
                current_norms = Norm(element, len(current_text))
            else:
                current_norms.process(element)
        elif element.tag == "verweis.rspr":
            if current_norms is not None:
                position, norm_refs = current_norms.finalize()
                norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
                norms.extend(norm_refs)

                norm_text = " ".join(norm_refs).strip() 
                insert_norm(current_text, norm_text, position)

                current_norms = None
            rspr = True
            for t in element.itertext():
                current_text.append(t.strip())
        elif element.tag in ["v.gericht", "v.entscheidungstyp", "v.datum", "v.az"]:
            # In this case we already have appended all the text of this element
            if rspr:
                continue
            else:
                if current_norms is not None:
                    position, norm_refs = current_norms.finalize()
                    norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
                    norms.extend(norm_refs)

                    norm_text = " ".join(norm_refs).strip() 
                    insert_norm(current_text, norm_text, position)

                    current_norms = None

                current_text.append(element.text.strip())
        elif element.text is not None:
            rspr = False
            # We will remove all spaces and join on them later on, as we can remove some possible edgecases that way
            if current_norms is not None:
                position, norm_refs = current_norms.finalize()
                norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
                norms.extend(norm_refs)

                norm_text = " ".join(norm_refs).strip() 
                insert_norm(current_text, norm_text, position)

                current_norms = None

            current_text.append(element.text.strip())
        
        # TODO For the norm case only allow this if we are in the "verweis.norm" case
        if element.tail is not None and element.tag not in ["v.abk", "v.norm"]:
            current_text.append(element.tail.strip())

    if len(current_text) > 0:
        if current_norms is not None:
            position, norm_refs = current_norms.finalize()
            norm_refs = [normDB.register_norm(norm) for norm in norm_refs]
            norms.extend(norm_refs)

            norm_text = " ".join(norm_refs).strip() 
            insert_norm(current_text, norm_text, position)

            current_norms = None

        texts.append(" ".join(current_text).strip())

    norms = [norm.strip() for norm in norms]

    return (texts, norms)

def insert_norm(text: List[str], norms: str, position: int) -> List[str]:
    """ We will insert all the norms at the specific position """
    text.insert(position, norms.strip())

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

def ges_bay_normchain(norm_nodes) -> List[str]:
    normchain = []
    for norm_node in norm_nodes:
        # We are saving the placeholders, not the norms!
        text = []
        for t in norm_node.find("normgliederung/enbez").itertext():
            text.append(t)
        text = "".join(text)
        text = text.strip("\n\t\r\v")

        # We have to split the norms, as sometimes multiple norm numbers are included from the same norm
        multi_norm = text.split(",")
        if len(multi_norm) > 1:
            multi_norm[0] = multi_norm[0].replace("§§", "§")
            normchain.append(multi_norm[0])
            norm_tld = multi_norm[0].split("§")[0].strip()
            for norm_num in multi_norm[1:]:
                norm_num = norm_num.replace("§", "").strip()
                if len(norm_num) > 0:
                    norm = norm_tld + " § " + norm_num.strip()
                    normchain.append(norm)
        else:
            if len(text) > 0:
                normchain.append(text)
    return normchain

def save_json(dic, filepath):
    with io.open(filepath, "w+", encoding='utf-8') as f:
        json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

def read_file(path):
    with open(path, "r") as f:
        text = f.read()
                
    return text

def get_file_counter(source: Path):
    try:
        with open("file_counter.pkl", "rb") as f:
            # Counter is a dictionary that maps from each Path to its processed files number
            counter = pickle.load(f)
    except FileNotFoundError:
        counter = dict()

    if source not in counter:
        counter[source] = 0

    return counter

def update_file_counter(source, counter):
    counter[source] = counter[source] + 1
    with open("file_counter.pkl", "wb") as f:
        pickle.dump(counter, f)

if __name__ == "__main__":
    process_ges_bay(Path("data"), Path("processed_data"))
    #process_ges_bay(Path("test_data"), Path("test_json"))