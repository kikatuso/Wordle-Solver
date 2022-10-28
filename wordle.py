import pandas as pd
import urllib.request, json 
import numpy as np
from math import log2
from string import ascii_lowercase
from tqdm import tqdm
import time


# for combining word_freq with entropy include a scale feature going from 0 to 1, where 0.5 
#corresponds to both 0.5*scaled01(word_freq)+0.5*scaled01(entropy). 

def load_data():
    short_data="https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/possible_words.txt"
    #long_data = 'https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/allowed_words.txt'

    lexicon = pd.read_csv(short_data,header=None).squeeze().to_numpy()

    with urllib.request.urlopen("https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/freq_map.json") as url:
        word_freq = json.load(url)

    word_freq = {key:word_freq[key] for key in lexicon}
    return lexicon,word_freq



alphabet_map=dict((idx,let) for (let,idx) in enumerate(ascii_lowercase))

def digitise(word,alphabet_map=alphabet_map):
    return list(map(alphabet_map.get,list(word)))


def calc_bit(p):
    bit = -p*log2(p)
    return(bit)

def per_word(words_vec,x,words): 
    words_vec[x,[j for j in range(5)],digitise(words[x])]=1
    return words_vec



def get_red_list(green_mask,green_letters,yellow_mask,yellow_letters,
                 grey_letters,words_dna_digitised,words):
    
    Nwords = len(words)
    green_vec = np.zeros([5,26],dtype=np.int8)
    green_vec[(green_mask==True).nonzero(),digitise(green_letters)]=1

    # need to include yellow_mask in the function 
    # perform operation similar to operation for green_bool to find out the yellow_bool
    yellow_vec=np.zeros([5,26],dtype=np.int8)
    yellow_vec[(yellow_mask==True).nonzero(),digitise(yellow_letters)]=1

    grey_vec = np.zeros([1,26],dtype=np.int8)
    grey_vec[0,digitise(grey_letters)]=1

    words_vec = np.zeros([Nwords*5,26],dtype=np.int8)
    words_vec[[i for i in range(Nwords*5)],words_dna_digitised]=1
    
    words_vec = words_vec.reshape(Nwords,5,26)


    green_bool = np.all((words_vec-green_vec>=0).reshape(-1,5*26),axis=1)
    yellow_bool1 = np.all((words_vec.sum(axis=1)-yellow_vec.sum(axis=0)>=0),axis=1)
    yellow_bool2 = (words_vec*yellow_vec).reshape(-1,5*26).sum(axis=1)==0

    xx=(words_vec.sum(axis=1)-yellow_vec.sum(axis=0)-green_vec.sum(axis=0))
    grey_bool=(np.dot(xx,grey_vec.T)==0).flatten()
    
    bools = yellow_bool1*yellow_bool2*green_bool*grey_bool
    red_list = words[bools]
    
    return red_list


def get_words_dna(words):
    words_dna = []
    for i in range(len(words)):
        words_dna.extend(list(words[i]))
    words_dna_digitised = list(map(alphabet_map.get,words_dna))
    return words_dna_digitised 
    
    
def get_entropy(word,lexicon):
    
    words_dna_digitised = get_words_dna(lexicon)
    sel_word_list = np.array(list(word))
    
    red_dist={}
    red_sets = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        green_mask  = np.zeros(5)
                        yellow_mask = np.zeros(5)

                        if i==2: green_mask[0]=True
                        if i==1: yellow_mask[0]=True
                        if j==2: green_mask[1]=True
                        if j==1: yellow_mask[1]=True
                        if k==2: green_mask[2]=True
                        if k==1: yellow_mask[2]=True
                        if l==2: green_mask[3]=True
                        if l==1: yellow_mask[3]=True
                        if m==2: green_mask[4]=True
                        if m==1: yellow_mask[4]=True

                        grey_mask = ~(np.array(yellow_mask)+np.array(green_mask)).astype(bool)

                        yellow_letters = sel_word_list[yellow_mask.astype(bool)]
                        green_letters = sel_word_list[green_mask.astype(bool)]
                        grey_letters = sel_word_list[grey_mask.astype(bool)]

                        red_list = get_red_list(green_mask,green_letters,yellow_mask,yellow_letters,
                                                           grey_letters,words_dna_digitised,lexicon)
                        
                        if len(red_list)==0:
                            continue
                            
                        red_dist[(i,j,k,l,m)]=red_list
                        red_sets.append(set(red_list))
                        
    unique_sets = set(map(frozenset,red_sets))
    counts=list(map(len,unique_sets))

    counts = np.array(counts)/np.sum(counts)
    entropy = sum(map(calc_bit,counts))
                                                
    return  entropy,red_dist



def get_entropy_list(lexicon):
    entropies = []
    dists = []
    words = []
    for word in tqdm(lexicon):
        entropy,dist=get_entropy(word,lexicon)
        entropies.append(entropy)
        words.append(word)
        dists.append(dist)
    output = {'word':words,'entropy':entropies}
    df = pd.DataFrame.from_dict(output)
    return df    

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
    while nlexicon>1:

        word = user_word(lexicon)
        time.sleep(1)
        wordle_output = user_score()
        print('Please wait...')
        new_lexicon,best_moves = player_move(word,lexicon,wordle_output)
        nlexicon = len(new_lexicon)
        print('Well done! You cut the dataset to {} possible word(s)!'.format(nlexicon))
        time.sleep(0.5)
        print('Your new best choices are:')  
        print(best_moves)
        lexicon = new_lexicon


