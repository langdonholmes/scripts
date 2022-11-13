import re
from tqdm.auto import tqdm

from spacy_alignments import get_alignments
from spacy.tokens import Span
nlp = spacy.blank("en")

def align_ents(docs,
               clean_func,
              ):
    '''Align document entities before and after some alterations to the text.'''
    
    new_docs = []
    for doc in tqdm(docs):
        new_doc = nlp(clean_func(doc.text))

        a2b, _ = get_alignments([t.text for t in doc],
                                [t.text for t in new_doc])

        new_ents = []
        for e in doc.ents:
            st = a2b[e.start][0]
            end = a2b[e.end][0] if a2b[e.end] else a2b[e.end-1][0] + 1
            
            while st >= end:
                end += 1
                
            new_e = Span(new_doc,
                     st,
                     end,
                     label=e.label_)
            
            new_ents.append(new_e)

        labels = [[new_e.start_char, new_e.end_char, new_e.label_] for new_e in new_ents]
        new_doc.set_ents(new_ents)        
        new_docs.append(new_doc)
        
        # if doc.ents:
        #     print(list(zip(doc.ents, new_doc.ents)))
        if len(doc.ents) != len(new_doc.ents):
            print(doc.ents)
            print(new_doc.ents)
            print(doc[:10])
            print(new_doc[:10])
            print(a2b)
            
    return new_docs