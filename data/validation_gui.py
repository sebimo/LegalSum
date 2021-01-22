# We want to use this GUI to create a filter list for all files with non-empty Guiding Principles
# Every other file will be discarded as it adds no further value for the GR
import os
from pathlib import Path
import pickle
import io
import json
import time

from tkinter import Scrollbar, Text, Frame, Tk, StringVar, Entry
from tkinter.font import Font
import tkinter

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.gr1 = TextBox(master=self)
        self.gr2 = TextBox(master=self)
        self.tenor = TextBox(master=self)
        self.facts = TextBox(master=self)
        self.reasoning = TextBox(master=self)
        self.grid_propagate(True)

        self.folder = "processed_data_nrw"
        self.files = []

        self.file_dict = {}
        self.file_name = None
        self.prev_file = None

        self.init_window()

    def init_window(self):
        """
        Initializes the GUI. Also loads all file names, which where not processed in the last application run
        """
        self.gr1.grid(row=0,column=0,sticky="NW",padx=0,pady=0,ipadx=0,ipady=0)
        self.gr2.grid(row=1,column=0,sticky="SW",padx=0,pady=0,ipadx=0,ipady=0)
        self.tenor.grid(row=0,column=1,sticky="NWES",padx=0,pady=0,ipadx=0,ipady=0)
        self.facts.grid(row=1,column=1,sticky="NWES",padx=0,pady=0,ipadx=0,ipady=0)
        self.reasoning.grid(row=2,column=1,sticky="NWES",padx=0,pady=0,ipadx=0,ipady=0)

        widget = [self.gr1, self.gr2, self.tenor, self.facts, self.reasoning]
        for w in widget:
            w.init_window()

        # TODO on-close we have to save the current list
        self.files = self.load_filenames()
        print("Remaining files:", len(self.files))
        self.next_file()

        self.focus_set()

        # Bind the followup commands to the keys
        self.bind("<Key>", self.keyhandle)
        self.bind("<w>", lambda _: self.flag_file())
        self.bind("<d>", lambda _: self.post_process_file())
        self.bind("<a>", lambda _: self.previous_file())
        self.bind("<space>", lambda _: self.next_file())
        self.bind("<Left>", lambda _: self.previous_file())
        self.bind("<Up>", lambda _: self.flag_file())
        self.bind("<Down>", lambda _: self.next_file())
        self.bind("<Right>", lambda _: self.post_process_file())

    def keyhandle(self, event):
        print(vars(event))

    def load_filenames(self):
        file_list = []
        if os.path.exists(Path("filter_files")/("filter_"+ self.folder +"_remaining_files.pkl")):
            with open(Path("filter_files")/("filter_"+ self.folder +"_remaining_files.pkl"), "rb") as f:
                file_list = pickle.load(f)
        else:
            file_list = os.listdir(Path(self.folder))
        return file_list

    def next_file(self):
        # Add to recent_files list, to reload if we need to go back 
        try:   
            file_name = self.files.pop(0)
            while file_name is None or file_name == "":
                file_name = self.files.pop(0)
            with io.open(Path(self.folder)/file_name, "r", encoding='utf-8') as f:
                file_dict = json.load(f)
                if len(file_dict["guiding_principle"][0])==0 and len(file_dict["guiding_principle"][1])==0:
                    self.next_file()
                else:
                    self.file_dict = file_dict
                    self.prev_file = self.file_name
                    self.file_name = file_name
        except IndexError:
            self.file_name = None
            self.file_dict = {
                "guiding_principle": [[], []],
                "tenor": [],
                "facts": [],
                "reasoning": [],
            }

        self.redraw()  

    def previous_file(self):
        self.files.insert(0, self.file_name)
        self.file_name = ""
        self.files.insert(0, self.prev_file)
        self.prev_file = None
        self.next_file()

    def flag_file(self):
        # Add the filename to the filterlist
        with open(Path("filter_files")/("filter_"+ self.folder +"_files.txt"), "a+") as f:
            f.write(self.file_name+"\n")
        # Load the next file
        self.next_file()

    def post_process_file(self):
        # Add the filename to the postprocesslist
        with open(Path("filter_files")/("post_"+ self.folder +"_files.txt"), "a+") as f:
            f.write(self.file_name+"\n")
        # Load the next file
        self.next_file()

    def redraw(self):
        self.gr1.redraw('\n'.join(self.file_dict["guiding_principle"][0]))
        self.gr2.redraw('\n'.join(self.file_dict["guiding_principle"][1]))
        self.tenor.redraw('\n'.join(self.file_dict["tenor"]))
        self.facts.redraw('\n'.join(self.file_dict["facts"]))
        self.reasoning.redraw('\n'.join(self.file_dict["reasoning"]))

    def save_files(self):
        self.files.insert(0, self.file_name)
        with open(Path("filter_files")/("filter_"+ self.folder +"_remaining_files.pkl"), "wb") as f:
            pickle.dump(self.files, f)

class TextBox(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.text = Text(self, height=20)
        self.scrollbar = Scrollbar(self, command=self.text.yview)

        self.text['yscrollcommand'] = self.scrollbar.set

    def init_window(self):
        self.scrollbar.pack(side=tkinter.RIGHT)
        self.text.pack(side=tkinter.LEFT)

    def redraw(self, content: str):
        self.text.delete(1.0, 'end')
        self.text.insert('end', content)


if __name__ == "__main__":
    root = Tk()

    font = Font(family="bitstream charter", size=13)

    app = Window(root)
    app.pack()
    root.mainloop()
    app.save_files()