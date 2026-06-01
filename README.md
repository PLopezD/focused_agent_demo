pip install tavily-python && pip freeze > requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pgrep -a lang


tradeoff 
    - why gpt-4o-mini?
    - ive done rag differently - with embeddings
    
learnings
    - i had never done the PII stuff

with more time 
    - made embeddings to go along with my rag architecture 
    - improve flow of tool selection, right now doing string search is not great
    - summarization after 15-20 messages 
    - spent more time with our database patterns including using something like postgres 
