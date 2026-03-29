# fix_praw.py
c = open('ingestion/sources.py', encoding='utf-8').read()
old = 'import praw as _praw\nimport logging as _logging\n_logging.getLogger("praw").setLevel(_logging.ERROR)'
new = 'import praw as _praw'
c = c.replace(old, new, 1)
open('ingestion/sources.py', 'w', encoding='utf-8').write(c)
print('REVERTED')