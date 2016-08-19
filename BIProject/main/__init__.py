import os
import nltk
from nltk import PunktSentenceTokenizer
from nltk.corpus import state_union

dictionary_path = os.path.join(os.path.dirname(__file__), '../dictionaries/positive.txt')
with open(dictionary_path, 'r') as f:
    list_of_positive_lines = [word.strip('\n') for word in f.readlines()]

dictionary_path = os.path.join(os.path.dirname(__file__), '../dictionaries/negative.txt')
with open(dictionary_path, 'r') as f:
    list_of_negative_lines = [word.strip('\n') for word in f.readlines()]

output_file_path = os.path.join(os.path.dirname(__file__), 'output.arff')
output_file = open(output_file_path, 'w')

output_file.write('@RELATION type_of_sentence\n')
output_file.write('\n')
output_file.write('@ATTRIBUTE freq_positive_adj         NUMERIC\n')
output_file.write('@ATTRIBUTE freq_negative_adj         NUMERIC\n')
output_file.write('@ATTRIBUTE freq_positive_noun        NUMERIC\n')
output_file.write('@ATTRIBUTE freq_negative_noun        NUMERIC\n')
output_file.write('@ATTRIBUTE freq_positive_verb        NUMERIC\n')
output_file.write('@ATTRIBUTE freq_negative_verb        NUMERIC\n')
output_file.write('@ATTRIBUTE class                     {positive, negative, neutral}\n')
output_file.write('\n')
output_file.write('@DATA\n')


######################################################################################################################################
######################################################################################################################################

# function that compare the word with the text of the dictionary
def comparison_text(list_of_words, word):
    if word in list_of_words:
        return True
    else:
        return False


######################################################################################################################################
######################################################################################################################################
# Function that start the comparison
def comparison_dictionary(word):
    word = word.lower().strip('\n')
    global return_word

    if comparison_text(list_of_positive_lines, word):
        return_word = 'POSITIVE'
    else:

        if comparison_text(list_of_negative_lines, word):
            return_word = 'NEGATIVE'
        else:
            return_word = 'NEUTRAL'

    return return_word


######################################################################################################################################
######################################################################################################################################

# Read the file text
file_path = os.path.join(os.path.dirname(__file__), '../books/LoveUnderFire.txt')
file = open(file_path, 'r')
text_file = file.read()

# Train the sentence tokenizer
train_text = state_union.raw("2005-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Set the list of sentences
sentences = custom_sent_tokenizer.tokenize(text_file)

# list of tagged sentences
tagged_sentences_list = []

try:
    for sentence in sentences:
        # output_file.write(sentence + ' ' + '\n')
        words = nltk.word_tokenize(sentence)
        tagged_word_list = nltk.pos_tag(words)
        tagged_sentences_list.append(tagged_word_list)
        # output_file.close()
except Exception as e:
    print(str(e))

# For every sentence
for sentence in tagged_sentences_list:

    freq_positive_adjectives = 0
    freq_negative_adjectives = 0
    freq_positive_nouns = 0
    freq_negative_nouns = 0
    freq_positive_verbs = 0
    freq_negative_verbs = 0

    # For every word in the  sentence
    for word in sentence:

        if word[1] in ["VBD", 'VBN', 'JJ']:

            type = comparison_dictionary(word[0])
            if type == 'POSITIVE':
                freq_positive_adjectives += 1
            else:
                freq_negative_adjectives += 1

        elif word[1] in ['NN', 'NNP', 'NNPS', 'NNS']:

            type = comparison_dictionary(word[0])
            if type == 'POSITIVE':
                freq_positive_nouns += 1
            else:
                freq_negative_nouns += 1

        elif word[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:

            type = comparison_dictionary(word[0])
            if type == 'POSITIVE':
                freq_positive_verbs += 1
            else:
                freq_negative_verbs += 1

    output_file.write(
        str(freq_positive_adjectives) + ',' +
        str(freq_negative_adjectives) + ',' +
        str(freq_positive_nouns) + ',' +
        str(freq_negative_nouns) + ',' +
        str(freq_positive_verbs) + ',' +
        str(freq_negative_verbs) + ',' +
        '?' + '\n'
    )

    # print(
    #     str(freq_positive_adjectives) + ',' +
    #     str(freq_negative_adjectives) + ',' +
    #     str(freq_positive_nouns) + ',' +
    #     str(freq_negative_nouns) + ',' +
    #     str(freq_positive_verbs) + ',' +
    #     str(freq_negative_verbs) + ',' +
    #     '?' + '\n'
    # )

output_file.close()
