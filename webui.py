from model_image_search import image_search
from gemini_ranker import gemini_rank
from data_downloader import download_data
from data_processor import process_data

import gradio as gr
from pathlib import Path
from PIL import Image

# Store state: search results and mode
search_results = gr.State([])
search_mode = gr.State("lite")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 AI Image Search Assistant")
    query_input = gr.Textbox(label="What picture do you want to search?")

    with gr.Row():
        lite_btn = gr.Button("Lite")
        big_btn = gr.Button("Big")

        # Add sliders to control number of images
    num_images_search = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                  label="Number of images to fetch from search (1-20)")
    num_images_rerank = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                  label="Number of images to fetch from Gemini re-ranking (1-10)")

    with gr.Row():
        show_search_btn = gr.Button("Show Top X (Search Results)")
        show_rerank_btn = gr.Button("Show Top X (LLM Reranked)")

    gallery = gr.Gallery(label="Search Results", columns=4, height="auto")

    # Button logic
    def set_mode(mode):
        return gr.update(value=mode)

    def run_search(query_input,mode,num_images_search):
        file_path = "data"
        # mode = "lite"
        feature_file = f"{file_path}/{mode}/features/features.npy"
        if not Path(feature_file).exists():
            photo_metadata = download_data(version="lite", data_path="data", threads_count=16)
            photo_ids_file, photo_features_file = process_data(photo_metadata, file_path, version="lite", batch_size=16)
        else:
            photo_ids_file = "data/lite/features/photo_ids.csv"
            photo_features_file = "data/lite/features/features.npy"
        best_photo_ids_raw = image_search(photo_ids_file, photo_features_file, query_input, num_images_search)
        image_paths = []
        for i, photo_id in enumerate(best_photo_ids_raw):
            photo_image_path = f"{file_path}/{mode}/photos/{photo_id}.jpg"
            image_paths.append(Image.open(photo_image_path))
        return image_paths,best_photo_ids_raw # update both state and top 10 display

    def show_search(images):
        return images

    def show_rerank(best_photo_ids_raw, version, query_input,num_images_rerank):
        file_path = "data"
        image_ids = gemini_rank(best_photo_ids_raw, file_path, version, query_input, num_images_rerank)
        result = []
        for i, image_id in enumerate(image_ids):
            photo_id = best_photo_ids_raw[image_id - 1]
            photo_image_path = f"{file_path}/{version}/photos/{photo_id}.jpg"
            result.append(Image.open(photo_image_path))
        return result


    def set_mode_lite():
        return "lite"


    def set_mode_big():
        return "big"


    lite_btn.click(fn=set_mode_lite, inputs=[], outputs=search_mode)
    big_btn.click(fn=set_mode_big, inputs=[], outputs=search_mode)

    # Run search when either mode is selected and query entered
    lite_btn.click(fn=run_search, inputs=[query_input, search_mode], outputs=[search_results, gallery])
    big_btn.click(fn=run_search, inputs=[query_input, search_mode], outputs=[search_results, gallery])

    show_search_btn.click(fn=show_search, inputs=[search_results, query_input], outputs=gallery)
    show_rerank_btn.click(fn=show_rerank, inputs=[search_results, query_input], outputs=gallery)

demo.launch()
    # file_path = "data"
    # version = "lite"
    # feature_file = f"{file_path}/{version}/features/features.npy"
    # if not Path(feature_file).exists():
    #     photo_metadata = download_data(version="lite", data_path="data",threads_count=16)
    #     photo_ids_file, photo_features_file = process_data(photo_metadata,file_path, version="lite", batch_size=16)
    # else:
    #     photo_ids_file = "data/lite/features/photo_ids.csv"
    #     photo_features_file = "data/lite/features/features.npy"

#
#
# def chatbot_interface(query):
#     top_images = search_images(query, top_k=5)
#     ranked_images = rerank_with_llm(query, top_images)
#     return ranked_images[:3]
#
# gr.Interface(fn=chatbot_interface, inputs=gr.Textbox(), outputs=gr.Gallery(label="Top 3 Images")).launch()
#
#
# file_path = "data"
#     version = "lite"
#     feature_file = f"{file_path}/{version}/features/features.npy"
#     if not Path(feature_file).exists():
#         photo_metadata = download_data(version="lite", data_path="data",threads_count=16)
#         photo_ids_file, photo_features_file = process_data(photo_metadata,file_path, version="lite", batch_size=16)
#     else:
#         photo_ids_file = "data/lite/features/photo_ids.csv"
#         photo_features_file = "data/lite/features/features.npy"
#     results_count = 10
#     results_count_final = 4
#     search_query = input("What picture you want to search?")
#     # search_query = "two birds flying"
#     best_photo_ids_raw = image_search(photo_ids_file, photo_features_file, search_query, results_count)
#     image_ids = gemini_rank(best_photo_ids_raw,file_path,version,search_query,results_count_final)
#     # print(best_photo_ids_raw)