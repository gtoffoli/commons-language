# module derived from:
# https://github.com/asterbini/spacy-babelnet/blob/master/spacy_babelnet/babelnet_annotator.py

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.tokens import Span
from spacy.matcher import Matcher

class TermsAnnotator():

    def __init__(self, model, glossary_concepts):
        print()
        self.model = model
        self.matcher = Matcher(model.vocab)
        for concept_id, lang_dict in glossary_concepts:
            terms = lang_dict[model.lang]
            self.matcher.add(concept_id, [self.term_to_pattern(term) for term in terms])
         # Span.set_extension('terms', default=None, force=True)

    def __call__(self, doc: Doc):
        self.doc = doc
        self.matches = self.matcher(doc, as_spans=True)
        if self.matches:
            doc.spans['GLOSSARY'] = self.matches
        return doc

    def token_to_pattern(self, token):
        pos = token.pos_
        pattern = {'POS': pos}
        if pos == 'ADP':
            pattern['OP'] = '?'
        elif pos == 'DET':
            pattern['OP'] = '?'
        else:
            pattern['LEMMA'] = token.lemma_
        return pattern

    def term_to_pattern(self, term):
        doc = self.model(term)
        return [self.token_to_pattern(token) for token in doc]

    def list_matches(self):
        for span in self.matches:
            print(self.model.vocab.strings[span.label], span.start, span.end, span.text)
