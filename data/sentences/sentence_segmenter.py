import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gensim.models import KeyedVectors

#TODO is the assumption correct, that we do not have any of those: ' ', '\t', '\v', '\r', '\f';
# \n is kept, but could be removed by translating the doc to numpy array and back
special = [".", ",", ";", ":", "*", "#", "!", "?", "\"" "(", ")", "{", "}", "[", "]", "-", "_", "\n"]

class SentenceSegmenter:

    def __init__(self, embedding_size=100, scope_before=7, scope_after=7, path=["sentences"]):
        word_embedding_path = os.path.join(*path, "word2vec_full.wv")
        model_path = os.path.join(*path, "model")
        self.word_model = KeyedVectors.load(word_embedding_path, mmap='r')
        self.model = GruWindow()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        self.model = self.model.to('cuda')
        self.embedding_size = 100
        self.scope_before = scope_before
        self.scope_after = scope_after

    def __call__(self, doc):
        """
        Predicts for every token whether it is considered a sentence boundary
        Arguments:
            token_list {List: Str} -- List of tokens
        Returns:
            {List: Bool} -- Annotations for every token on SBD, where True means Sentence Boundary
        """
        token_list = [token.text for token in doc]
        self.model.eval()
        dataset = WindowDataset(token_list, self.word_model, self.embedding_size,
                                scope_before=self.scope_before, scope_after=self.scope_after)
        # Using a DataLoader to make the datatransfer more efficient (more samples in one batch) when detaching to numpy
        dataloader = DataLoader(dataset, batch_size=1028, pin_memory=False)
        sentence_boundary = []
        for window in dataloader:
            window = window.cuda()
            output = self.model.forward(window)
            output = output.cpu().detach().numpy()
            x = output.reshape(-1)
            ret = [True if i > 0.5 else False for i in x]
            for t in ret:
                sentence_boundary.append(t)

        # Now we have to set token.is_sent_start = True for all sentence boundaries. All other need token.is_sent_start = False
        # !CAREFUL: the model produces predictions for sentence terminating tokens, so we have to assign this boundary
        # to the first following Alphanumeric token!
        sentence_end = False
        for i, (token, boundary) in enumerate(zip(token_list, sentence_boundary)):
            if not boundary and not sentence_end:
                doc[i].is_sent_start = False
            elif boundary:
                doc[i].is_sent_start = False
                sentence_end = True
            elif sentence_end:
                if token in special:
                    doc[i].is_sent_start = False
                else:
                    doc[i].is_sent_start = True
                    sentence_end = False
        return doc


class GruWindow(nn.Module):

    def __init__(self, d_in=100, d_h=64, num_layers=4, bidirectional=True, scope_before=7, scope_after=7):
        """
        Creates the NN Model used for sentence segmentation. The given parameters are the ones used by 
        the current model.
        Arguments:
            d_in {int} -- Size of the word embeddings
            d_h {int} -- Size of the hidden dimension used between the GRUs
            num_layers {int} -- Number of subsequent GRU layers before the classification layer
            bidirectional {bool} -- Is the model using bidirectional layers?
            scope_before {int} -- Number of tokens before the current position used for the prediction
            scope_after {int} -- Number of tokens after the current position used for the prediction
        """
        super(GruWindow, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        """
        Produces the output of the Neural Network.
        Arguments:
            tok {torch.Tensor[batch size, window length, embedding size]} -- The current input
        Returns:
            {torch.Tensor[batch size]} -- Predictions for each window
        """
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class WindowDataset(Dataset):

    def __init__(self, token, vector_model, embedding_size, scope_before=7, scope_after=7):
        """
        Creation of the Dataset used for the predictions. We are using the dataset in order
        to abstract away the transformations from tokens to word embeddings.
        Arguments:
            token {List: String} -- All the training tokens
            target {List: Bool}: The corresponding targets based on the sbd annotations
            embedding_size {int} -- Size of the word embeddings
            scope_before {int} -- Number of tokens before the current position used for the prediction
            scope_after {int} -- Number of tokens after the current position used for the prediction
        """
        self.token = []
        self.embedding_size = embedding_size
        self.scope_before = scope_before
        self.scope_after = scope_after
        for i, tok in enumerate(token):
            # Input accumulation
            token_tensor = torch.from_numpy(vector_model[tok].copy()) if tok in vector_model else torch.zeros(self.embedding_size, dtype=torch.float32)
            self.token.append(token_tensor)

    def __getitem__(self, idx):
        """
        Returns a tensor with the embeddings of a token with its surrounding tokens (given by
        scope_before and scope_after). Unknown words are replaced with a zero vector. If a 
        token is at the beginning/end of a document the tensor is padded with zero vectors
        to get same sized windows for each sample.
        Arguments:
            idx {int} -- The index for a sample in the dataset
        Returns:
            {torch.Tensor[window length, embedding size]} -- torch Tensor representing the
            window window
        """
        scope_start = idx - self.scope_before
        # We need + 1 because otherwise the last element in the window would not be included
        # scope_end is used in the range function as the delimiter
        scope_end = idx + self.scope_after + 1
        token = list()
        for s in range(scope_start, 0):
            token.append(torch.zeros(self.embedding_size, dtype=torch.float32))
        scope_start = max(scope_start, 0)
        for s in range(scope_start, idx):
            token.append(self.token[s])
        token.append(self.token[idx])
        for e in range(idx+1, min(self.__len__(), scope_end)):
            token.append(self.token[e])
        for e in range(self.__len__(), scope_end):
            token.append(torch.zeros(self.embedding_size, dtype=torch.float32))
        token = [torch.unsqueeze(t, dim=0) for t in token]
        output_token = torch.cat(token, dim=0)

        return output_token

    def __len__(self):
        return len(self.token)

if __name__ == "__main__":
    sent = SentenceSegmenter(path=[])
    text = """Die Antragstellerin ist eine politische Partei und begehrt im Wege des einstweiligen
Rechtsschutzes die Verpflichtung des Rundfunk Berlin-Brandenburg (rbb), einer öffentlich-
rechtlichen Rundfunkanstalt, anlässlich der Europawahl einen von ihr eingereichten
Wahlwerbespot auf den zugeteilten Sendeplätzen auszustrahlen. Der Wahlwerbespot enthalte in seiner
Gesamtschau einen evidenten und schwerwiegenden Verstoß gegen § 130 Abs. 1 Nr. 2
StGB, da er Migrantinnen und Migranten pauschal als Kriminelle diffamiere und eine
Zweiteilung der Gesellschaft in Deutsche und (kriminelle) Ausländer propagiere. 
Ergibt die Prüfung im Eilrechtsschutzverfahren, dass eine Verfassungsbeschwerde
offensichtlich begründet wäre, läge in der Nichtgewährung von Rechtsschutz der schwere
Nachteil für das gemeine Wohl im Sinne des § 32 Abs. 1 BVerfGG (vgl. BVerfGE 111,
147 <153>)."""
    import spacy
    from spacy.lang.de import German
    nlp = German()
    without_special_cases = {}
    nlp.tokenizer.rules = without_special_cases
    nlp.add_pipe(sent)
    doc = nlp(text)