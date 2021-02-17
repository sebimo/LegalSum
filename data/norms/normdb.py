from pathlib import Path
import pickle

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

class NormDBStub(NormDatabase):

    def __init__(self):
        self.norm2id, self.id2norm, self.max_id = dict(), dict(), 0

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
        print("Closing NormDatabaseStub...")
