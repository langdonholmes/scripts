def true_positive_by_overlap(corpus, annotator_source):
    corpus_tp = corpus_fp = corpus_fn = 0
    
    for doc in corpus:
        true_labels = list(doc.ents)
        auto_labels = list(doc.spans[annotator_source])

        if true_labels or auto_labels:
            doc_tp, doc_fp, doc_fn = overlapper(auto_labels, true_labels)
            corpus_tp += doc_tp
            corpus_fp += doc_fp
            corpus_fn += doc_fn
            
    recall = corpus_tp / (corpus_tp + corpus_fn)
    precision = corpus_tp / (corpus_tp + corpus_fp)
    print(f'True Positives: {corpus_tp}')
    print(f'False Positives: {corpus_fp}')
    print(f'False Negatives: {corpus_fn}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')

def overlapper(xs, ys):
    # xs are a set of labelled spans
    # ys are treated as the gold labels
    # stats for this document
    doc_tp = doc_fp = doc_fn = 0

    i = j = 0
    while i < len(xs) and j < len(ys):
        x = xs[i]
        y = ys[j]

        # the intersection of the two annotations, if they overlap:
        intersection = range(
            max(x.start, y.start),
            min(x.end, y.end)
        )

        if intersection:
            # step both lists                
            j += 1
            i += 1
            doc_tp += 1

        # no intersection, step the earlier's list forward
        elif x.start < y.start:
            i += 1
            doc_fp += 1
        else:
            j += 1
            doc_fn += 1
            
    doc_fp += len(xs[i:]) # remaining xs are false positives
    doc_fn += len(ys[j:]) # remaining ys are false negatives

    return doc_tp, doc_fp, doc_fn