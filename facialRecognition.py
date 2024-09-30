import cv2
import google.generativeai as genai

genai.configure(api_key="")

def imagePreprocessing(image):
    _, buffer = cv2.imencode('.png', image)
    im_buffer = buffer.tobytes()
    return im_buffer

def geminiRequest(image):
    image_bytes = imagePreprocessing(image)
    
    tempPNG = {
        'mime_type': 'image/png',
        'data': image_bytes
    }
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = """
    Analyse this photo taken from a webcam. Is the person looking at or very near 
    their screen (within reason to account for camera angle)? What emotion from this 
    list would they best fit into to? LIST: Happy, Sad, Angry, Neutral. Please respond 
    with a boolean for attention and a string for emotion ONLY. Example: True, Happy"""
    
    response = model.generate_content([tempPNG,prompt])
    return response

def facialRecognitionFeed():
    camera = cv2.VideoCapture(0)
    ret, curFrame = camera.read()
    
    if not ret:
        exit()
    else:
        response = geminiRequest(curFrame)

    camera.release()
    return response
    
def mood():
    output = facialRecognitionFeed()
    output = output.text
    if "Happy" in output:
        return "Happy"
    elif "Sad" in output:
        return "Sad"
    elif "Angry" in output:
        return "Angry"
    elif "Neutral" in output:
        return "Neutral"
    else:
        return "Neutral"
    
if __name__ == "__main__":
    print(mood())
