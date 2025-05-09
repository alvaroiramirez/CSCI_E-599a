import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np

def get_bert_embedding(text, model, tokenizer):
    """
    Get BERT embeddings for a text string.
    Returns the mean of all token embeddings from the last hidden layer.
    """
    # Add special tokens and convert to tensor
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get model output (without computing gradients)
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Get the embeddings from the last hidden state
    # Shape: [batch_size, sequence_length, hidden_size]
    token_embeddings = output.last_hidden_state
    
    # Create attention mask to ignore padding tokens
    attention_mask = encoded_input['attention_mask']
    
    # Expand attention mask to same dimensions as token_embeddings
    # and use it to zero out padding token embeddings
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    masked_embeddings = token_embeddings * expanded_mask
    
    # Sum embeddings along sequence dimension and divide by number of non-padding tokens
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.sum(attention_mask, dim=1, keepdim=True)
    sentence_embedding = summed / counts
    
    return sentence_embedding[0].numpy()  

def compare_captions_with_bert(caption1, caption2):
    """
    Compare two captions using BERT embeddings and cosine similarity.
    Returns similarity score between 0 and 1, where 1 means identical.
    """
    #pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    #embeddings for both captions
    embedding1 = get_bert_embedding(caption1, model, tokenizer)
    embedding2 = get_bert_embedding(caption2, model, tokenizer)
    
    #cosine similarity (convert to similarity from distance)
    similarity = 1 - cosine(embedding1, embedding2)
    
    return similarity

original_caption = "Major damage, a building reduced to a pile of rubble and surrounded by debris"
generated_caption = "major damage, collapsed building"

similarity_score = compare_captions_with_bert(original_caption, generated_caption)
print(f"Similarity score: {similarity_score:.4f}")