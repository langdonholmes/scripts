import typer
import warnings
from tqdm.auto import tqdm

import spacy
from spacy.tokens import DocBin

def main(model_loc, in_bin, out_bin):
    nlp = spacy.load(model_loc)
    
    docs = list(DocBin().from_disk(in_bin).get_docs(nlp.vocab))
    
    (
        DocBin()
        .add(
          list(longformer
               .pipe(tqdm([doc.text
                       for doc in docs]))
              )
          )
        .to_disk(out_bin)
    )

if __name__ == "__main__":
    typer.run(main)