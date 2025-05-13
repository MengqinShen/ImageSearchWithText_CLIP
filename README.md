# ImageSearch_CLIP_Unsplash
Search photos on Unsplash Dataset using natural language descriptions. The search is powered by OpenAI's CLIP model. This can be a model implemented in cell phone "Photos".

In this project: 
1. Images and metadata was processed with pretrained CLIP model for image-text embedding and zero-shot retrieval.
2. Implemented a RAG LLM system to return semantically relevant videos based on user queries.
3. for user interface, a webui using Gradio was built to enable user access and search image easily.

You can run main.py to get the resluts plotted from LLM- RAG or run webui.py to get a link and interact with your local image dataset. 
