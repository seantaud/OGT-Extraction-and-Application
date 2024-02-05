import nltk
nltk.download('punkt')
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

def sent_slipter():
    punkt_param = PunktParameters()
    abbreviation = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u',
                    'v', 'w', 'x', 'y', 'z', 'et al', 'spp', 'sp', 'fig', 'nov', 'e.g', 'Dr']
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)
    return tokenizer


if __name__ == '__main__':
    tokenizer = sent_slipter()
