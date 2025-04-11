import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm as tqdm
from evaluate import load
from transformers import MarianMTModel, MarianTokenizer

print('Starting Script')

sentences = pd.read_csv('1millsentences.csv')

print('Read File')

# Preparing Training Data
train_X = sentences['english'][:int(len(sentences) * .7)] # English only for input
train_y = sentences['spanish'][:int(len(sentences) * .7)] # Spanish for target

# Preparing Testing Data
test_X = sentences['english'][int(len(sentences) * .7) : int(len(sentences) * .9)]
test_y = sentences['spanish'][int(len(sentences) * .7) : int(len(sentences) * .9)]

# Preparing Validation Data
val_X = sentences['english'][int(len(sentences) * .9) : int(len(sentences))]
val_y = sentences['spanish'][int(len(sentences) * .9) : int(len(sentences))]

train_X = np.array(train_X)
train_y = np.array(train_y)

test_X = np.array(test_X)
test_y = np.array(test_y)

val_X = np.array(val_X)
val_y = np.array(val_y)

print('Loaded Data')

model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)

print('Loaded Tokenizer')

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        self.mask = []
        
        # Creating a dataset of inputs to model, and their outputs
        for data_X, data_y in zip(X, y):
            
            # Handles if the input file has Non-string like objects
            if isinstance(data_X, str) and isinstance(data_y, str):
                
                # Tokenize text
                inputs = tokenizer(data_X, max_length=128, padding='max_length', truncation=True)
                with tokenizer.as_target_tokenizer():
                    targets = tokenizer(data_y, max_length=128, padding='max_length', truncation=True)
                
                # Add tokenized text to our data sets
                self.input_ids.append(inputs['input_ids'])
                self.target_ids.append(targets['input_ids'])
                self.mask.append(inputs['attention_mask'])
                
        
        # Convert to tensors
        self.input_ids = torch.tensor(self.input_ids)
        self.target_ids = torch.tensor(self.target_ids)
        self.mask = torch.tensor(self.mask)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input': self.input_ids[index],
            'target': self.target_ids[index],
            'mask'  : self.mask[index]
        }

train_dataset = TextDataset(train_X, train_y, tokenizer)
test_dataset = TextDataset(test_X, test_y, tokenizer)
val_dataset = TextDataset(val_X, val_y, tokenizer)

batch_size = 16

# Create DataLoaders 
print('Starting Data Loader')
train_loader = DataLoader(train_dataset, batch_size=batch_size)
print('Train Data Loaded')
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print('Test Data Loaded')
val_loader = DataLoader(val_dataset, batch_size=batch_size)
print('Val Data Loaded')

print('Data Preprocessed')

model_name = 'Helsinki-NLP/opus-mt-en-es'
model = MarianMTModel.from_pretrained(model_name)
base_model = MarianMTModel.from_pretrained(model_name)

print('Loaded Model')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Loaded Device as: {device}')

model.to(device)    
base_model.to(device)

def train(model, training_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in training_loader:
        ids = data['input'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, labels=targets, attention_mask=mask)
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.zero_grad()
        optimizer.step()
        
        total_loss += loss.item()
        
        
    return total_loss / len(training_loader)

def val(model, val_loader, tokenizer, bertscore, device):
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for data in val_loader:
            targets = data['target'].to(device, dtype = torch.long)
            ids = data['input'].to(device, dtype = torch.long)
            generated_ids = model.generate(input_ids = ids)
            generated_targs = model.generate(input_ids = targets)
            
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(generated_targs, skip_special_tokens=True)
            
            # Add each prediction/reference pair individually
            for pred, ref in zip(predictions, references):
                all_predictions.append(pred)
                all_references.append(ref) 
            
        results = bertscore.compute(predictions=all_predictions, references=all_references, device=device, lang='es')
    return results['f1']

def test(model, test_loader, tokenizer, bertscore, device):
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for data in test_loader:
            targets = data['target'].to(device, dtype = torch.long)
            ids = data['input'].to(device, dtype = torch.long)
            generated_ids = model.generate(input_ids = ids)
            generated_targs = model.generate(input_ids = targets)
            
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(generated_targs, skip_special_tokens=True)
            
            # Add each prediction/reference pair individually
            for pred, ref in zip(predictions, references):
                all_predictions.append(pred)
                all_references.append(ref) 
                
        results = bertscore.compute(predictions=all_predictions, references=all_references, lang='es', device=device)
    return results

optim = torch.optim.AdamW(model.parameters(), lr=.0001)
NUM_EPOCH = 4
bertscore = load("bertscore")

print('Training Starting')

for epoch in range(NUM_EPOCH):
    loss = train(model, train_loader, optim, device)
    torch.save(model.state_dict(), "fine_tuned_en_es.bin")
    model.config.to_json_file("config.json")
    
    f1_score = val(model, val_loader, tokenizer, bertscore, device)
    print(f'Epoch: {epoch+1} \nBERTScore F1: {np.mean(f1_score)}\nTraining Loss: {loss}')
    
print('Training Done')
print('Evaluation Starting on training set')

f1_score_based_model = test(base_model, test_loader, tokenizer, bertscore, device)
f1_score_finetuned = test(model, test_loader, tokenizer, bertscore, device)

print('')
print('-------------------------------')
print('Evaluation on Test Set Complete')
print('-------------------------------')
print('-------------------------------')
print('         Base MT Model         ')
print('-------------------------------')
print(f"F1         : {np.mean(f1_score_based_model['f1'])}")
print(f"Recall     : {np.mean(f1_score_based_model['recall'])}")
print(f"Precision  : {np.mean(f1_score_based_model['precision'])}")
print('-------------------------------')
print('        Fine Tuned Model       ')
print('-------------------------------')
print(f"F1         : {np.mean(f1_score_finetuned['f1'])}")
print(f"Recall     : {np.mean(f1_score_finetuned['recall'])}")
print(f"Precision  : {np.mean(f1_score_finetuned['precision'])}")
print('-------------------------------')


torch.save(model.state_dict(), "fine_tuned_en_es.bin")
model.config.to_json_file("config.json")



