import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
from pathlib import Path
import sys
import tempfile

# Add current directory to path for imports
sys.path.append('.')

try:
    from data_downloader import download_data
    from data_processor import process_data
    from model_image_search import image_search
    from gemini_ranker import gemini_rank
    print("All modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required modules are available")

class FixedRAGChatbot:
    def __init__(self):
        self.file_path = "data"
        self.version = "lite"
        self.feature_file = f"{self.file_path}/{self.version}/features/features.npy"
        self.photo_ids_file = None
        self.photo_features_file = None
        self.initialized = False
        self.initialize_data()
    
    def initialize_data(self):
        """Initialize the data and models"""
        try:
            if not Path(self.feature_file).exists():
                print("Downloading and processing data...")
                photo_metadata = download_data(version="lite", data_path="data", threads_count=16)
                photo_ids_file, photo_features_file = process_data(photo_metadata, self.file_path, version="lite", batch_size=16)
                self.photo_ids_file = photo_ids_file
                self.photo_features_file = photo_features_file
            else:
                self.photo_ids_file = "data/lite/features/photo_ids.csv"
                self.photo_features_file = "data/lite/features/features.npy"
            
            self.initialized = True
            print("Data initialized successfully!")
        except Exception as e:
            print(f"Error initializing data: {e}")
            self.initialized = False
    
    def search_images(self, query):
        """Search for images using the existing pipeline"""
        if not self.initialized:
            return [], [], "System not initialized. Please check the console for errors."
        
        try:
            # Search for images using CLIP
            best_photo_ids_raw = image_search(
                self.photo_ids_file, 
                self.photo_features_file, 
                query, 
                10  # results_count
            )
            
            if not best_photo_ids_raw:
                return [], [], f"No images found for '{query}'"
            
            # Rank images using Gemini
            image_ids = gemini_rank(
                best_photo_ids_raw, 
                self.file_path, 
                self.version, 
                query, 
                4  # results_count_final
            )
            
            if not image_ids:
                return best_photo_ids_raw, [], f"Found {len(best_photo_ids_raw)} images but couldn't rank them for '{query}'"
            
            return best_photo_ids_raw, image_ids, f"Found {len(image_ids)} relevant images for '{query}'"
            
        except Exception as e:
            error_msg = f"Error searching for images: {str(e)}"
            print(error_msg)
            return [], [], error_msg
    
    def create_image_grid(self, best_photo_ids, image_ids, query):
        """Create a grid of images and return the file path"""
        if not best_photo_ids or not image_ids:
            return None
        
        try:
            rows = 2
            display_cols = math.ceil(len(image_ids) / rows)
            
            fig, axes = plt.subplots(rows, display_cols, figsize=(12, rows * 3))
            
            # Handle single image case
            if len(image_ids) == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.flatten()
            
            for i, image_id in enumerate(image_ids):
                if image_id <= len(best_photo_ids):
                    photo_id = best_photo_ids[image_id - 1]
                    photo_image_path = f"{self.file_path}/{self.version}/photos/{photo_id}.jpg"
                    
                    if os.path.exists(photo_image_path):
                        img = mpimg.imread(photo_image_path)
                        axes[i].imshow(img)
                        axes[i].set_title(f"Photo ID: {photo_id}", fontsize=8)
                        axes[i].axis("off")
                    else:
                        axes[i].text(0.5, 0.5, f"Image not found: {photo_id}", 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].axis("off")
            
            # Hide unused subplots
            for i in range(len(image_ids), len(axes)):
                axes[i].axis("off")
            
            plt.suptitle(f'Images found for: "{query}"', fontsize=16)
            plt.tight_layout()
            
            # Save to temporary file
            temp_file = f"search_results_{hash(query) % 10000}.png"
            plt.savefig(temp_file, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return temp_file
            
        except Exception as e:
            print(f"Error creating image grid: {e}")
            return None
    
    def chat(self, message, history):
        """Main chat function"""
        if not message.strip():
            return "", history
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        try:
            # Search for images
            best_photo_ids, image_ids, status_msg = self.search_images(message)
            
            if best_photo_ids and image_ids:
                # Create image grid
                image_file = self.create_image_grid(best_photo_ids, image_ids, message)
                
                if image_file and os.path.exists(image_file):
                    response = f"{status_msg} Here are the images:"
                    history.append({"role": "assistant", "content": response})
                    return "", history, image_file
                else:
                    response = f"{status_msg} But I couldn't display them. Please try again."
                    history.append({"role": "assistant", "content": response})
                    return "", history, None
            else:
                history.append({"role": "assistant", "content": status_msg})
                return "", history, None
                
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}. Please try again."
            history.append({"role": "assistant", "content": response})
            return "", history, None

def create_chatbot_interface():
    """Create the Gradio interface"""
    chatbot = FixedRAGChatbot()
    
    with gr.Blocks(title="RAG Image Search Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ RAG Image Search Chatbot")
        gr.Markdown("Chat with me to search for images! I'll use CLIP and Gemini to find the most relevant images for your queries.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    height=600,
                    show_label=False,
                    container=True,
                    bubble_full_width=False,
                    type="messages"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Describe the image you're looking for...",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("Search", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat")
                
                # Status indicator
                status_text = gr.Textbox(
                    value="System ready" if chatbot.initialized else "System not ready - check console",
                    label="Status",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📸 Search Results")
                image_output = gr.Image(
                    label="Found Images",
                    height=400,
                    show_label=True
                )
        
        # Event handlers
        def user_input(message, history):
            return chatbot.chat(message, history)
        
        def clear_chat():
            return [], None
        
        # Connect events
        msg.submit(user_input, [msg, chatbot_interface], [msg, chatbot_interface, image_output])
        submit_btn.click(user_input, [msg, chatbot_interface], [msg, chatbot_interface, image_output])
        clear_btn.click(clear_chat, outputs=[chatbot_interface, image_output])
        
        # Example queries
        gr.Markdown("### 💡 Example queries:")
        gr.Markdown("- 'two birds flying above water'")
        gr.Markdown("- 'sunset over mountains'")
        gr.Markdown("- 'people walking in the city'")
        gr.Markdown("- 'flowers in a garden'")
        
        # Instructions
        gr.Markdown("### 📋 Instructions:")
        gr.Markdown("1. Type your image description in the text box")
        gr.Markdown("2. Press Enter or click Search")
        gr.Markdown("3. Wait for the system to find and display relevant images")
        gr.Markdown("4. Use the Clear Chat button to start over")
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861) 