# define and replace the Arabic tokenizer
import spacy
from spacy.tokens import Doc

class MsaTokenizer:
    def __init__(self, vocab):
        from camel_tools.disambig.mle import MLEDisambiguator
        from camel_tools.tokenizers.morphological import MorphologicalTokenizer
        self.vocab = vocab
        self.nlp = spacy.blank("ar")
        mle_msa = MLEDisambiguator.pretrained('calima-msa-r13')
        self.msa_atb_tokenizer = MorphologicalTokenizer(disambiguator=mle_msa, scheme='atbtok')

    def __call__(self, text):
        doc = self.nlp(text)
        forms = self.msa_atb_tokenizer.tokenize([t.text for t in doc])
        words = []
        for form in forms:
            # Separate prefixes
            splitted = form.rsplit('+_')
            while len(splitted) > 1:
                words.append(splitted[0])
                splitted = splitted[1:]
            form = splitted[0]
            # Separate suffixes
            splitted = form.rsplit('_+')
            while len(splitted) > 1:
                words.append(splitted[0])
                splitted = splitted[1:]
            words.append(splitted[0])
        spaces = [True] * len(words)
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False
        return Doc(self.vocab, words=words, spaces=spaces)

def create_msa_tokenizer():

    @spacy.registry.tokenizers("msa_tokenizer")
    def create_tokenizer(nlp):
        return MsaTokenizer(nlp.vocab)

    return create_tokenizer
