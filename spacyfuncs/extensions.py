from spacy.language import Language

@Language.component('line_annotator')
def line_annotator(doc):
    '''
    Doc --> Doc
    ! Assumes that all lines are separated by a single \n token !
    '''
    line_num = 0
    line_start = 0
    spans = []
    line_texts = []
    for t in doc:
        if t.text == '\n':
            span = doc[line_start:t.i+1]
            span._.line_no = line_num
            if span.text in line_texts:
                span._.repeated = True
            line_texts.append(span.text)
            spans.append(span)
            line_start = t.i+1
            line_num += 1
    doc.spans['lines'] = spans
    return doc
