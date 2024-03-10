# TECHIN 513 Final Project  - Tale
## The core solution to the communication challenges of aphasia is an AI-powered tool that excels in context-aware, real-time conversation analysis and assistance.
### Key Features Emphasizing Context-Aware Technology 
- Context-Based Speech Analysis: Utilizing Azure Event Hubs and Functions alongside Redis, our system excels in processing speech data in real-time. It adeptly adapts to and understands the context of conversations, offering timely and relevant assistance.
- Context-Aware Word Retrieval/sentance completion: Our AI tool goes beyond standard word prediction, finely tuned to grasp the specifics of conversational context. This leads to precise suggestions for words or phrases, thus greatly improving communication fluidity for aphasia patients.
- Listening and Learning Mechanism: More than just a reactive tool, our system continuously learns from user interactions. This ongoing adaptation tailors its assistance to each individual’s unique communication patterns.
- Progress Tracking Feature: A significant addition to our tool is the comprehensive tracking of patient progress. This feature records and analyzes speech improvements over time, offering detailed reports. These insights can be invaluable during visits to doctors or therapists, providing a clear picture of the user's journey and advancements in managing aphasia.

## How to Run
Open the terminal and run the following commands:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
flask run
```

## What's Included

- `app.py`: The main application made with Streamlit, pyannote, and OpenAI.
