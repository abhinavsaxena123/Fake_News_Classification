from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

# Parent folder of 'app'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Saved_Models")


# Load RoBERTa model & tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained(os.path.join(SAVED_MODELS_DIR, "roberta_tokenizer"))
roberta_model = RobertaForSequenceClassification.from_pretrained(os.path.join(SAVED_MODELS_DIR, "roberta_model"))
roberta_model.eval()

# Load DistilBERT model & tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(SAVED_MODELS_DIR, "distilbert_tokenizer"))
distilbert_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(SAVED_MODELS_DIR, "distilbert_model"))
distilbert_model.eval()

label_map = {0: "Fake News", 1: "Real News"}

##################################################################################

def predict_roberta(text):
    # Tokenize input text
    inputs = roberta_tokenizer(text, return_tensors="pt", 
                truncation=True, padding=True, 
                max_length=512
                )
    
    # Forward pass (disable gradient calculation for efficiency)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]

###################################################################################

def predict_distilbert(text):
    # Tokenize input text
    inputs = distilbert_tokenizer(text, return_tensors="pt", 
                truncation=True, padding=True, 
                max_length=512
                )
    
    # Forward pass (disable gradient calculation for efficiency)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]
