from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import string
import emoji
from bs4 import BeautifulSoup
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set display options for pandas
pd.set_option("display.max_columns", None)


class TextPreprocessing:
    '''
    TextPreprocessing class provides methods for cleaning and preprocessing text data.

    Attributes:
        contraction_mapping (dict): Mapping of contractions to their expanded forms.
        punct (list): List of punctuation characters.
        punct_mapping (dict): Mapping of special punctuation characters to their replacements.
        mispell_dict (dict): Dictionary of common misspellings and their corrections.

    Methods:
        clean_text: Cleans text by removing emojis, links, punctuation, and special characters.
        clean_contractions: Expands contractions in the text.
        clean_special_chars: Cleans special characters from the text.
        correct_spelling: Corrects common spelling errors.
        remove_space: Removes extra spaces from the text.
        pipeline: Applies a series of text cleaning steps in a pipeline.
    '''

    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                           "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                           "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am",
                           "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                           "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                           "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                           "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                           "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                           "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                           "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                           "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                           "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                           "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                           "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s': 'america', 'e.g': 'for example'}

    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
             '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
             '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
             '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
             '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                     "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                     'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!': ' '}

    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                    'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                    'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                    'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                    'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                    'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                    'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                    'demonetisation': 'demonetization'}

    @staticmethod
    def clean_text(text):
        '''
        Clean emoji, Make text lowercase, remove text in square brackets,
        remove links, remove punctuation and remove words containing numbers.
        '''
        # Convert emoji to text representation
        text = emoji.demojize(text)
        # Remove emoji characters
        text = re.sub(r'\:(.*?)\:', '', text)
        # Convert text to lowercase
        text = str(text).lower()
        # Remove text in square brackets
        text = re.sub('\[.*?\]', '', text)
        # Remove links
        text = re.sub('https?://\S+|www\.\S+', '', text)
        # Remove html tags
        text = BeautifulSoup(text, 'lxml').get_text()
        # Remove newline characters
        text = re.sub('\n', '', text)
        # Remove words containing numbers
        text = re.sub('\w*\d\w*', '', text)
        # Remove all characters except (a-z, A-Z, ".", "?", "!", ",", "'")
        text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
        return text

    @staticmethod
    def clean_contractions(text, mapping):
        '''Clean contraction using contraction mapping'''
        # Replace special characters with apostrophe
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")

        # Iterate through contraction mapping
        for word in mapping.keys():
            # Replace each contraction with its expanded form
            if word in text:
                text = text.replace(word, mapping[word])

        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # Add space between words and punctuation
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        # Remove extra spaces
        text = re.sub(r'[" "]+', " ", text)

        return text

    @staticmethod
    def clean_special_chars(text, punct, mapping):
        '''Cleans special characters present (if any)'''
        # Replace special characters with their mapped values
        for p in mapping:
            text = text.replace(p, mapping[p])

        # Add space around punctuation characters
        for p in punct:
            text = text.replace(p, f' {p} ')

        # Replace special unicode characters with their
        # corresponding replacements
        specials = {'\u200b': ' ', '…': ' ... ',
                    '\ufeff': '', 'करना': '', 'है': ''}
        for s in specials:
            text = text.replace(s, specials[s])

        return text

    @staticmethod
    def correct_spelling(text, dic):
        '''Corrects common spelling errors'''
        # Iterate through dictionary of corrections
        # and replace misspelled words
        for word in dic.keys():
            text = text.replace(word, dic[word])
        return text

    @staticmethod
    def remove_space(text):
        '''Removes awkward spaces'''
        # Strip leading and trailing spaces
        text = text.strip()
        # Split text into list of words and remove any empty elements
        text = text.split()
        # Join the list of words with a single space between each word
        return " ".join(text)

    @staticmethod
    def pipeline(text):
        '''Cleaning and parsing the text.'''
        # Apply each cleaning step sequentially
        text = TextPreprocessing.clean_text(text)
        text = TextPreprocessing.clean_contractions(
            text, TextPreprocessing.contraction_mapping)
        text = TextPreprocessing.clean_special_chars(
            text, TextPreprocessing.punct, TextPreprocessing.punct_mapping)
        text = TextPreprocessing.correct_spelling(
            text, TextPreprocessing.mispell_dict)
        text = TextPreprocessing.remove_space(text)
        return text


class BERTTestDataset(Dataset):
    '''
    BERTTestDataset class represents a PyTorch dataset for testing BERT models.

    Args:
        df (DataFrame): DataFrame containing the summaries.
        tokenizer: Tokenizer to encode the text.
        max_len (int): Maximum length of the input sequence.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Retrieves an item from the dataset at the given index.
    '''

    def __init__(self, df, tokenizer, max_len):
        '''
        Initializes the BERTTestDataset.

        Parameters:
            df (DataFrame): DataFrame containing the summaries.
            tokenizer: Tokenizer to encode the text.
            max_len (int): Maximum length of the input sequence.
        '''
        self.df = df
        self.max_len = max_len
        self.text = df.summary
        self.tokenizer = tokenizer

    def __len__(self):
        '''Returns the length of the dataset.'''
        return len(self.df)

    def __getitem__(self, index):
        '''
        Retrieves an item from the dataset at the given index.

        Parameters:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the input IDs, attention mask,
            and token type IDs.
        '''
        text = self.text[index]
        # Tokenize the text and encode it as input IDs,
        # attention mask, and token type IDs
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


class BERTClass(torch.nn.Module):
    '''
    BERTClass represents a fine-tuned RoBERTa model for genre classification.

    Attributes:
        roberta (AutoModel): Pre-trained RoBERTa model.
        fc (torch.nn.Linear): Linear layer for classification.

    Methods:
        __init__: Initializes the BERTClass.
        forward: Performs a forward pass through the model.
    '''
    
    def __init__(self):
        '''
        Initializes the BERTClass.

        This class defines a fine-tuned RoBERTa model for genre classification.

        '''
        super(BERTClass, self).__init__()
        # Load the pre-trained RoBERTa model
        self.roberta = AutoModel.from_pretrained('roberta-base')
        # Add a linear layer for classification
        self.fc = torch.nn.Linear(768, 10)  # 768 is the hidden size of RoBERTa

    def forward(self, ids, mask, token_type_ids):
        '''
        Forward pass of the BERTClass.

        Parameters:
            ids (torch.Tensor): Input IDs.
            mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.

        Returns:
            torch.Tensor: Output logits for genre classification.
        '''
        # Pass the inputs through the RoBERTa model
        _, features = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
            )
        # Pass the features through the linear layer
        output = self.fc(features)
        return output


class ModelUtils:
    '''
    ModelUtils contains utility methods for working with models.

    Methods:
        load_model: Loads a pre-trained model from a checkpoint.
        validation: Performs validation on a dataset using a model.
        get_pred_genres: Obtains predicted genres based on validation results.
    '''
    @staticmethod
    def load_model(path):
        '''
        Loads a pre-trained model from the specified path.

        Parameters:
            path (str): Path to the pre-trained model checkpoint.

        Returns:
            tuple: Tuple containing the loaded model and the device.
        '''
        # Create an instance of the BERTClass
        model = BERTClass()
        # Load the model state dictionary from the specified path
        model.load_state_dict(torch.load(
            path, map_location=torch.device('cpu')))
        # Set the device to CPU
        device = 'cpu'
        return model, device

    @staticmethod
    def validation(pred_loader, model):
        '''
        Performs validation on the provided data loader using the given model.

        Parameters:
            pred_loader (DataLoader): Data loader for prediction.
            model (BERTClass): Model for genre classification.

        Returns:
            list: List of predicted outputs.
        '''
        fin_outputs = []
        # Disable gradient calculation
        with torch.no_grad():
            # Iterate through the data loader
            for _, data in enumerate(pred_loader, 0):
                # Retrieve input IDs, attention mask, and token type IDs
                ids = data['ids']
                mask = data['mask']
                token_type_ids = data['token_type_ids']
                # Perform forward pass through the model
                outputs = model(ids, mask, token_type_ids)
                # Append the predicted outputs to the list
                fin_outputs.extend(torch.sigmoid(
                    outputs).cpu().detach().numpy().tolist())
        return fin_outputs

    @staticmethod
    def get_pred_genres(validation, pred_loader, model, device, threshold=0.5):
        '''
        Obtains predicted genres based on the provided validation results.

        Parameters:
            validation (function): Function for performing validation.
            pred_loader (DataLoader): Data loader for prediction.
            model (BERTClass): Model for genre classification.
            device (str): Device to use for inference.
            threshold (float): Threshold for binary classification.

        Returns:
            list: List of predicted genres.
        '''
        # Perform validation to get predicted outputs
        outputs = validation(pred_loader, model)
        # Apply thresholding for binary classification
        outputs = np.array(outputs) >= threshold
        # Define genre labels
        genres = ["Drama", "Comedy", "Romance", "Thriller", "Action",
                  "Crime", "Horror", "Family Film", "Adventure", "Animation"]
        # Map predicted outputs to genre labels
        values_array = np.array(outputs)
        pred_genres = [np.array(genres)[value_row]
                       for value_row in values_array]
        return pred_genres


def load_model_and_tokenizer():
    '''
    Loads the pre-trained model and tokenizer.

    Returns:
        tuple: Tuple containing the loaded model, device, tokenizer,
        and maximum length.
    '''
    # Path to the pre-trained model checkpoint
    path = 'model.bin'
    # Maximum length of input sequence
    MAX_LEN = 200
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # Load the pre-trained model and device
    model, device = ModelUtils.load_model(path)
    return model, device, tokenizer, MAX_LEN


def predict_genre(text, model, device, tokenizer, MAX_LEN):
    '''
    Predicts the genre of a given text using a pre-trained model.

    Parameters:
        text (str): Input text to predict the genre for.
        model (BERTClass): Pre-trained model for genre classification.
        device (str): Device to use for inference.
        tokenizer: Tokenizer to encode the text.
        MAX_LEN (int): Maximum length of the input sequence.

    Returns:
        str: Predicted genre for the input text.
    '''
    # Define batch size for prediction
    TRAIN_BATCH_SIZE = 64
    # Preprocess the input text
    text = TextPreprocessing.pipeline(text)
    # Create a DataFrame with the preprocessed text
    df = pd.DataFrame({'summary': [text]})
    # Create a BERTTestDataset with the preprocessed text
    pred_data = BERTTestDataset(df, tokenizer, MAX_LEN)
    # Create a DataLoader for prediction
    pred_loader = DataLoader(pred_data, batch_size=TRAIN_BATCH_SIZE,
                             num_workers=4, shuffle=True, pin_memory=True)
    # Get predicted genres for the input text
    pred_genres = ModelUtils.get_pred_genres(
        ModelUtils.validation, pred_loader, model, device)
    # Return the predicted genre
    return pred_genres[0]
