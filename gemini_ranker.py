import google.generativeai as genai
import re


def gemini_rank(best_photo_ids_raw,file_path,version,search_query,results_count_final):
    # Your Gemini API key
    genai.configure(api_key="your-key")
    model = genai.GenerativeModel("gemini-1.5-flash")
    images = []
    for i, photo_id in enumerate(best_photo_ids_raw):
        photo_image_path = f"{file_path}/{version}/photos/{photo_id}.jpg"
        with open(photo_image_path, "rb") as f:
            img_data = f.read()
        images.append({
            "mime_type": "image/jpeg",
            "data": img_data
        })

    #     img_data = Image.open(photo_image_path)
    #     images.append({
    #     "name": photo_id,
    #     "data": img_data
    # })

    query = f"Select the {results_count_final} most relevant images that look like {search_query}.Return only the image id of the selected images. "
    #     query = [{"role": "user", "parts": [
    #     f"Here are {results_count} images. Please select {results_count_final} images that best match the prompt {search_query}. Return only the image names of the selected images."
    # ]}]
    # 📤 Send to Gemini
    response = model.generate_content(
        [query] + images
    )
    if response.candidates is None:
        print("Can't find images aligned with search query. Try again")
        image_ids = []
    else:
        print(response.text)
        image_ids = list(map(int, re.findall(r'\d+', response.text)))

    return image_ids
