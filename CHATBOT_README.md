# 🖼️ RAG Image Search Chatbot

A conversational chatbot interface for your RAG (Retrieval-Augmented Generation) image search system. This chatbot uses CLIP for image-text similarity search and Gemini for intelligent image ranking and selection.

## ✨ Features

- **Chat Interface**: Natural conversation with the image search system
- **Image Display**: Visual results showing found images in a grid layout
- **CLIP Integration**: Uses OpenAI's CLIP model for semantic image search
- **Gemini Ranking**: Leverages Google's Gemini model for intelligent image selection
- **Real-time Search**: Instant results as you type your queries
- **User-friendly UI**: Clean, modern interface built with Gradio

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r chatbot_requirements.txt
```

### 2. Ensure Data is Available

Make sure you have the image dataset and features available. If not, run:

```bash
python main.py
```

This will download and process the Unsplash dataset.

### 3. Launch the Chatbot

```bash
python run_chatbot.py
```

Or directly:

```bash
python chatbot_simple.py
```

The chatbot will open in your browser at `http://localhost:7860`

## 💡 How to Use

1. **Type your query**: Describe the image you're looking for in natural language
2. **Press Enter or click Search**: The system will process your request
3. **View results**: Images will be displayed in a grid on the right side
4. **Clear chat**: Use the "Clear Chat" button to start over

### Example Queries

- "two birds flying above water"
- "sunset over mountains"
- "people walking in the city"
- "flowers in a garden"
- "cat sitting on a windowsill"
- "beach with palm trees"

## 🔧 Technical Details

### Architecture

- **Frontend**: Gradio web interface
- **Image Search**: CLIP (Contrastive Language-Image Pre-training)
- **Image Ranking**: Google Gemini 1.5 Flash
- **Image Display**: Matplotlib with base64 encoding for web display

### Components

- `chatbot_simple.py`: Main chatbot implementation
- `run_chatbot.py`: Launcher script with error handling
- `chatbot_requirements.txt`: Required Python packages

### Data Flow

1. User inputs text query
2. CLIP encodes query and finds similar images
3. Gemini ranks and selects best matches
4. Images are displayed in a grid layout
5. Results are shown in the chat interface

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r chatbot_requirements.txt
   ```

2. **Data Not Found**: Ensure the data directory exists and contains images
   ```bash
   python main.py  # This will download data if needed
   ```

3. **Port Already in Use**: Change the port in the launch call
   ```python
   demo.launch(server_port=7861)  # Use different port
   ```

4. **Memory Issues**: Reduce batch sizes in data processing if needed

### Error Messages

- **"System not ready"**: Check console for initialization errors
- **"No images found"**: Try different query terms
- **"Couldn't display images"**: Check if image files exist in data/photos/

## 🔒 Security Notes

- The chatbot runs locally by default
- No data is sent to external servers (except Gemini API calls)
- Keep your Gemini API key secure
- The interface is accessible from your local network

## 📱 Customization

### Changing the Interface

Edit `chatbot_simple.py` to modify:
- UI layout and styling
- Number of displayed images
- Search parameters
- Response messages

### Adding Features

- Save chat history
- Export search results
- Batch processing
- Different ranking algorithms

## 🤝 Contributing

Feel free to improve the chatbot by:
- Adding better error handling
- Improving the UI/UX
- Adding new search features
- Optimizing performance

## 📄 License

This chatbot is part of your RAG image search project. Make sure to comply with the licenses of the underlying models (CLIP, Gemini) and datasets (Unsplash).

## 🆘 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure data files are available
4. Check your Gemini API key configuration

---

**Happy Image Searching! 🎉** 