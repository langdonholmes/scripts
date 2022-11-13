import re

linebreaks = re.compile('\s*[\r\n\f\v]+\s*') # one or more linebreaks and any surrounding whitespace
not_linebreaks = re.compile('[ \t]+') # one or more whitespace that is not a linebreak

replace_linebreaks = lambda text: re.sub(linebreaks, '\n', text)
replace_not_linebreaks = lambda text: re.sub(not_linebreaks, ' ', text)
fix_whitespace = lambda text: replace_not_linebreaks(replace_linebreaks(text))