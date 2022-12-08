from pathlib import Path
from tqdm.auto import tqdm
import typer

import spacy
from spacy.tokens import Doc, DocBin
from .extensions import *
    
def get_docs(texts,
             out_file,
             model_name = 'en_core_web_trf',
             additional_components = [],
             id_text_tuples = True,
             gpu = True):
    '''
    nlp: a spacy model
    texts: an iterable of (name, text documents) tuples or just text_documents
    out_file: the destination of the DocBin
    returns --> a list of docs, pulled from DocBin or generated via spacy pipeline    
    '''
    
    nlp = spacy.load(model_name)
    out_file = Path(out_file)
    
    if not Doc.has_extension('name'):
        Doc.set_extension('name', default=None)
        
    additional_components.append('doc_cleaner')
    
    for component in additional_components:
        if not component in nlp.pipe_names:
            nlp.add_pipe(component)
    
    db = DocBin(store_user_data=True)
        
    if out_file.exists():
        print('Retrieving from existing Docbin file')
        db = db.from_disk(out_file)
    else:
        print('Creating new Docbin file')
        for bundle in tqdm(nlp.pipe(texts, as_tuples=id_text_tuples), total=len(texts)):
            if id_text_tuples:
                doc, name = bundle
                doc._.name = name
            else:
                doc = bundle            
            db.add(doc)
            
        db.to_disk(out_file)
        
    return list(tqdm(db.get_docs(nlp.vocab), total=len(texts)))

if __name__ == "__main__":
    typer.run(get_docs)