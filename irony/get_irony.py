import pandas as pd
from transformers import pipeline
import torch

def main():
    print("Loading dataset train_2022.csv ...")
    df = pd.read_csv("train_2022.csv")

    # Initialize the pipeline
    print("Loading model cardiffnlp/twitter-roberta-base-irony ...")
    device = 0 if torch.cuda.is_available() else -1
    irony_classifier = pipeline(
        "text-classification", 
        model="cardiffnlp/twitter-roberta-base-irony", 
        tokenizer="cardiffnlp/twitter-roberta-base-irony", 
        truncation=True, 
        max_length=512,
        device=device
    )

    print("Running inference on TEXT column...")
    texts = df['TEXT'].tolist()
    
    # Process in batches or all at once
    results = irony_classifier(texts)

    irony_scores = []
    irony_labels = []
    
    for res in results:
        # labels are usually 'irony' or 'non_irony' (some models use 'LABEL_1', 'LABEL_0')
        # for cardiffnlp/twitter-roberta-base-irony: 0 -> non_irony, 1 -> irony
        label_str = res['label']
        score = res['score']
        
        if '1' in label_str or label_str.lower() == 'irony':
            irony_scores.append(score)
            irony_labels.append(1)
        else:
            irony_scores.append(1 - score)
            irony_labels.append(0)

    # 註解: 新增預測標籤與反諷分數 (irony_score 為預測是反諷的機率)
    df['irony_pred'] = irony_labels
    df['irony_score'] = irony_scores

    print("Saving to irony.csv ...")
    df.to_csv("irony.csv", index=False)
    print("Process complete!")

if __name__ == "__main__":
    main()
