from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np
from collections import defaultdict

# Load your encoded data
symptom_vectors = pickle.load(open("symptom_vectors.pkl", "rb"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup FastAPI app
app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: List[str]

def infer_disease(symptoms: List[str], top_n=3):
    all_results = []
    for symptom in symptoms:
        query_vector = model.encode(symptom, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_vector, symptom_vectors["vectors"])[0]
        top_indices = np.argsort(similarities.numpy())[::-1][:top_n]
        for i in top_indices:
            all_results.append({
                "disease": symptom_vectors["diseases"][i],
                "matched_symptom": symptom_vectors["symptoms"][i],
                "similarity": float(similarities[i])
            })

    # Group and average by disease
    disease_scores = defaultdict(list)
    for r in all_results:
        disease_scores[r["disease"]].append(r["similarity"])

    averaged = [{"disease": d, "score": np.mean(scores)} for d, scores in disease_scores.items()]
    averaged = sorted(averaged, key=lambda x: x["score"], reverse=True)

    return averaged[:top_n]

@app.post("/predict_disease")
def predict(request: SymptomRequest):
    predictions = infer_disease(request.symptoms)

    suggestions = {
        "Flu": {
            "medications": ["Congestal", "Comtrex"],
            "follow_up": "Do you also have muscle aches or chills?"
        },
        "Common Cold": {
            "medications": ["Congestal", "Comtrex"],
            "follow_up": "Do you have sneezing or red eyes?"
        },
        "Bronchitis": {
            "medications": ["Mucosol", "Bronchicum"],
            "follow_up": "Do you have mucus production or chest discomfort?"
        }
    }

    results = []
    for pred in predictions:
        disease = pred["disease"]
        results.append({
            "disease": disease,
            "confidence": round(pred["score"], 2),
            "medications": suggestions.get(disease, {}).get("medications", []),
            "follow_up_question": suggestions.get(disease, {}).get("follow_up", "")
        })

    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


