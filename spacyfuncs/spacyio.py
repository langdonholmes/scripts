import warnings
from pathlib import Path
from typing import Iterable, List

import spacy
import srsly
import typer
from spacy.tokens import Doc, DocBin
from tqdm.auto import tqdm

if not Doc.has_extension('name'):
    Doc.set_extension('name', default=None)
nlp = spacy.blank('en')

app = typer.Typer()


def _to_docbin(docs: Iterable[Doc], out_path: Path) -> None:
    '''Helper function that sends an iterable of docs to a DocBin on disk
    '''
    db = DocBin(store_user_data=True)
    for doc in docs:
        db.add(doc)
    db.to_disk(Path(out_path))
    
def _convert(text_lines: Iterable[dict]) -> Iterable[Doc]:
    '''Convert entity annotation from Doccano format to spaCy v3 .spacy format.
    '''
    for line in text_lines:
        text = line['data']
        doc = nlp.make_doc(text)
        doc._.name = line['id']
        ents = []
        for start, end, label in line['label']:
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                msg = f"Document: {line['id']}\nEntity [{start}, {end}, {label}] does not align with token boundaries.\nOriginal entity was '{doc.text[start:end]}'"
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                msg += f"\nAttempting to set entity as '{span}'"
                warnings.warn(msg)
            ents.append(span)
        doc.ents = ents
        yield doc
        
@app.command()
def get_docs(texts: Iterable = typer.Argument(..., help='An iterable of (name, text) tuples or just text tuples'),
             out_file: str = typer.Argument(..., help='The location of the DocBin'),
             model_name: str = typer.Argument('en_core_web_trf', help='The name of the spacy model'),
             additional_components: Iterable[str] = typer.Argument([], help='Additional components to add to the spaCy pipeline'),
             id_text_tuples: bool = typer.Argument(True, help='Whether the texts are tuples of iter[tuple[name, text]]. If False, texts is iter[text].'),
             ) -> List[Doc]:
    '''Creates a DocBin file from a list of texts.'''
    
    out_path = Path(out_file)
    nlp = spacy.load(model_name)
        
    additional_components.append('doc_cleaner')
    for component in additional_components:
        if not component in nlp.pipe_names:
            nlp.add_pipe(component)
        
    if out_path.exists():
        print('Retrieving from existing Docbin file')
        db = DocBin(store_user_data=True).from_disk(out_path)
        return list(tqdm(db.get_docs(nlp.vocab), total=len(texts)))
    else:
        print('Creating new Docbin file')
        docs = []
        for bundle in tqdm(nlp.pipe(texts, as_tuples=id_text_tuples), total=len(texts)):
            if id_text_tuples:
                doc, name = bundle
                doc._.name = name
            else:
                doc = bundle            
            docs.append(doc)
        _to_docbin(docs, out_path)
        return(docs)

@app.command()
def convert(in_file: str = typer.Argument(..., help='The location of the jsonl file'),
            train_percent: int = typer.Argument(0, help='The percentage of the data to use for training. Set to 0 to skip splitting data.'),
            ) -> None:
    '''Converts a jsonl file to a train/dev/test split of .spacy files.
    '''
    from sklearn.model_selection import train_test_split

    input_path = Path(in_file)
    
    if input_path.suffix == '.jsonl':
        raw = list(srsly.read_jsonl(input_path))
        if any([key not in raw[0].keys() for key in ['id', 'data', 'label']]):
            warnings.warn("Raw data is expected to be json lines with id, data, label keys.")
            return
        docs = list(_convert(raw))
    elif input_path.suffix == '.spacy':
        docs = list(DocBin().from_disk(input_path).get_docs(nlp.vocab))
    else:
        warnings.warn("Unknown filetype. '.spacy' and '.jsonl' are supported.")
        return
    
    train, _remains = train_test_split(docs, train_size=train_percent/100, random_state=0)
    dev, test = train_test_split(_remains, train_size=0.5, random_state=0)
    _to_docbin(train, 'corpus/train.spacy')
    _to_docbin(dev, 'corpus/dev.spacy')
    _to_docbin(test, 'corpus/test.spacy')
    
if __name__ == "__main__":
    app()
    
