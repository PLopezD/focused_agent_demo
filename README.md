pip install pytest pytest-asyncio && pip freeze > requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest tests/test_auth.py -v
pytest tests/ -v -s --tb=short
pytest tests/ -v


pgrep -a lang
pgrep -li lang



  docker build -t langchain-music-store .
  docker run -p 8501:8501 langchain-music-store




tradeoff:
    - why gpt-4o-mini?
    - ive done rag differently - with embeddings
    - semantic search on queries for routtings w llms vs keyword search
    - Predictability vs. Agent Autonomy
    
learnings:
    - i had never done the PII stuff

with more time 
    - made embeddings to go along with my rag architecture 
    - improve flow of tool selection, right now doing string search is not great
    - summarization after 15-20 messages 
    - spent more time with our database patterns including using something like postgres 
    - code cleanup (class inheritence, be more DRY, various middlewares)


PLD todo 
- find out why rag is not further filtering when it has a good answer --- xxxx
- write dockerfile --- xxx
- add pii middleware --- xxx
- add eval dataset --- 
- sort out music suggestion algo - xxxx


- CALL BANDON (can be pushed)