from fer.fer import FER 

emotion_detector = FER(mtcnn=True)

def predict_emotion(image):
    result = emotion_detector.detect_emotions(image)
    
    if not result:
        return {"emotion": "No face detected"}

    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    confidence = emotions[top_emotion]

    return {"emotion": top_emotion, "confidence": round(confidence, 2)}
