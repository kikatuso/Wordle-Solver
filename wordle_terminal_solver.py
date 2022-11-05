import numpy as np 
from main import get_words_dna,get_red_list,get_entropy_list,load_data
import time 
import pandas as pd 

def player_move(word,lexicon,wordle_output):
    """
    wordle_output in the format : 
    
    wordle_output = list/np.array with each cell being 'gn'/'gy'/'y' for 'green'/'grey'/'yellow'
    e.g. wordle_output = np.array(['gn','gy','y','gy','y']) being first letter green, second letter grey etc. 
    
    """
    
    sel_word_list = np.array(list(word))

    words_dna_digitised = get_words_dna(lexicon)
    
    wordle_output = np.array(wordle_output)
    green_mask = wordle_output=='gn'
    yellow_mask = wordle_output=='y'
    grey_mask =  wordle_output =='gy'

    yellow_letters = sel_word_list[yellow_mask.astype(bool)]
    green_letters = sel_word_list[green_mask.astype(bool)]
    grey_letters = sel_word_list[grey_mask.astype(bool)]

    # resulting new lexicon
    red_list = get_red_list(green_mask,green_letters,yellow_mask,yellow_letters,
                            grey_letters,words_dna_digitised,lexicon)
    # best next move 
    entropy_list = get_entropy_list(red_list)

    best_moves = entropy_list.sort_values(by='entropy',ascending=False).head(5).reset_index(drop=True)
    
    new_lexicon = red_list
    return new_lexicon,best_moves


def user_word(lexicon):
    word = input("Your choice: ")
    while word not in lexicon:
        print('Please provide a valid word.')
        time.sleep(1)
        word = input("Your choice: ")
    return word 

def user_score():
    print('What score did you get with the word "{}"?'.format(word))
    wordle_output = []
    for i in range(1,5+1):
        letter = input('Colour for letter no.{}? format:[gn,y,gy]: '.format(i))
        while letter not in ['gn','y','gy']:
            print('Please write either "gn" for green, "y" for yellow or "gy" for grey.')
            time.sleep(0.5)
            letter = input('Colour for the letter no.{}? format:[gn,y,gy]: '.format(i))
        wordle_output.append(letter)
    return wordle_output



if __name__ == "__main__":

    init_lexicon,_= load_data()
    print("Let's play Wordle!")
    time.sleep(1)
    print('Here are the best 5 words to start with:')
    initial_prob = pd.read_csv('initial_prob.csv')
    print(initial_prob.sort_values(by='entropy',ascending=False).head(5).reset_index(drop=True)[['word','entropy']])
    time.sleep(1)

    lexicon  = init_lexicon
    nlexicon = len(lexicon)
    turns = 0 
    game = True
    while game:
        turns = turns+1 
        word = user_word(lexicon)
        time.sleep(1)
        wordle_output = user_score()

        if (np.array(wordle_output)=='gn').all():
                outcome = 'won'
                break
        if turns==6:
            outcome = 'lost'
            break
        print('Please wait...')
        new_lexicon,best_moves = player_move(word,lexicon,wordle_output)
        nlexicon = len(new_lexicon)
        if  nlexicon==0:
            outcome = 'error'
            break
        print('Well done! You cut the dataset to {} possible word(s)!'.format(nlexicon))
        time.sleep(0.5)
        print('Your new best choices are:')  
        print(best_moves)
        lexicon = new_lexicon
 
    
    time.sleep(1.0)
    if outcome=='won':
        print('Well done, you finished the game in {} moves!'.format(turns))
    elif outcome=='lost':
        print('The number of turns has been exceeded. Try again next time.')
    elif outcome=='error':
        text= """There are no words matching your criteria. Please verify your input and try again next time."""
        print(text)


