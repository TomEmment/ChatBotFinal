import random
import aiml
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
import csv
import math
import re
import tweepy
from collections import Counter
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wav
import time
import cv2
import sys
from simpful import *
import pygame
from pygame.locals import *
import enchant
import requests, uuid, json
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials
import IPython
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from playsound import playsound


pygame.init()
####Azure

cog_key = '20da385e8547423288a84a543238e8be'
cog_endpoint = 'https://coursework.cognitiveservices.azure.com/'
cog_region = 'westeurope'

print('Ready to use cognitive services in {} using key {}'.format(cog_region, cog_key))
text_analytics_client = TextAnalyticsClient(cog_endpoint, AzureKeyCredential(cog_key))


def translate_text(cog_region, cog_key, text, to_lang, from_lang):

    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text': text
    }]

    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]







def PlayText(response_text):
    

    speech_config = SpeechConfig(cog_key, cog_region)
    output_file = os.path.join("Test.wav")
    audio_output = AudioConfig(filename=output_file) 
    speech_synthesizer = SpeechSynthesizer(speech_config, audio_output)

    speech_synthesizer.speak_text(response_text)


    data, fs = sf.read("Test.wav", dtype='float32')  
    sd.play(data, fs)
    status = sd.wait()

d = enchant.Dict("en_GB")

stopwords = stopwords.words('english')
warnings.filterwarnings('ignore')
## connect to Twitter
consumer_key = 'X4GdKJsPC5FwUypV4R4yQWlWk'
consumer_secret = 'zF9m8Sra9lDnXC7iNkLG8AzT9ZlN7OblRqR64GdSuiPFiOg4TE'
access_token = '1318227698762874884-FPwVWzuBVEPZVRujJ0PQBoYPygWlt2'
access_token_secret = '8ERA3WilxXFPK0TtheA66Y54OPmah97klkuKg1rzXmKCk'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

WORD = re.compile(r"\w+")

#Get bag of words

BagOfWords = []
###Cosine Similairity###
with open('CourseworkPart4Data.csv', newline='') as csvfile:
    Sentances = csv.reader(csvfile)
    for row in Sentances:
        TempWords = row[0].lower()
        Words = TempWords.split()
        for w in Words:
            if w in BagOfWords:
                Position = BagOfWords.index(w) +1
                BagOfWords[Position] = BagOfWords[Position] +1
            else:
                BagOfWords.append(w)
                BagOfWords.append(1)
    csvfile.close()


def ImplementFrequency(Vector):
    for x in BagOfWords:
        if BagOfWords.index(x)%2 == 0:
            if Vector[x] != 0:
                Vector[x] = Vector[x]/BagOfWords[BagOfWords.index(x) +1]
    return Vector
    
def CleanString(text):
    text = text.lower()
    temp = word_tokenize(text)
    temp1 = temp
    #temp = {w for w in temp if not w in stopwords}
    #if temp == "":
        #temp = temp1
    text = repr(temp)
    words = WORD.findall(text)
    return Counter(words)

def CreateVectors(Message,Orginal = False):
    vectors = CleanString(Message)
    if not Orginal:
        vectors = ImplementFrequency(vectors)
    return vectors

def CalculateCosine(vec1,vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def FindBestResponseInCSV(Message):
    MessageVector = CreateVectors(Message,True)
    Response = ""
    Highest = 0
    Choice = random.randint(1,3)
    with open('CourseworkPart4Data.csv', newline='') as csvfile:
        Sentances = csv.reader(csvfile)
        for row in Sentances:
            MessageVector1 = CreateVectors(row[0].lower())
            Result = CalculateCosine(MessageVector,MessageVector1)
            if Result>Highest:
                Highest = Result
                Response = row[Choice]
        csvfile.close()
    return Response

def FindBestUserInCSV(Message):
    MessageVector = CreateVectors(Message,True)
    Response = ""
    Highest = 0
    with open('TwitterCheck.csv', newline='') as csvfile:
        Sentances = csv.reader(csvfile)
        for row in Sentances:
            MessageVector1 = CreateVectors(row[0].lower())
            Result = CalculateCosine(MessageVector,MessageVector1)
            if Result>Highest:
                Highest = Result
                Response = row[1]
        csvfile.close()
    return Response



modelAudio = keras.models.load_model("AudioModel.keras")
modelImage = keras.models.load_model("ImageModel.keras")

#cam = cv2.VideoCapture(0)
#ret, frame = cam.read()

###CNN###
def ConvertData1Audio():
    y, sr = librosa.load("Test.wav", mono=True, duration=2)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
    plt.axis('off');
    plt.savefig("Test.png")
    plt.clf()

def TestAudio():
    image = tf.keras.preprocessing.image.load_img("Test.png")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = modelAudio.predict(input_arr)
    if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
        print("Paper")
        return 1
    if predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
        print("Rock")
        return 0
    if predictions[0][2] > predictions[0][1] and predictions[0][2] > predictions[0][0]:
        print("Scissors")
        return 2
                                
def GetInputAudio(Time):
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print ("Recording Audio")
    #time.sleep(0.3)
    myrecording = sd.rec(Time * 44100, samplerate=44100, channels=2,dtype='float64')
    sd.wait()
    Name = "Test.wav"
    sf.write(Name, myrecording, 44100)
    sd.wait()

def CropImage():
    picture = Image.open("Test.png")


    CutStarty = 9999
    CutStartx = 9999
    CutEndx = 0
    CutEndy = 0
    width, height = picture.size
    for x in range(0,width):
         for y in range(0,height):
            current_color = picture.getpixel( (x,y) )
            if current_color[0] >= 160 and current_color[1] >= 160 and current_color[2] >= 160:
                if x >= CutEndx:
                    CutEndx = x
                if y >= CutEndy:
                    CutEndy = y
                if y <= CutStarty:
                    CutStarty = y                        
                if x <= CutStartx:
                    CutStartx = x
    picture = picture.crop((CutStartx,CutStarty,CutEndx,CutEndy))
    picture = picture.resize((400,250))
    picture.save("Test.png","png")

def TestImage():
    image = tf.keras.preprocessing.image.load_img("Test.png")
    
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = modelImage.predict(input_arr)
    if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
        print("Paper")
        return 1
    if predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
        print("Rock")
        return 0
    if predictions[0][2] > predictions[0][1] and predictions[0][2] > predictions[0][0]:
        print("Scissors")
        return 2


                                
def GetInputImage():
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    ret, frame = cam.read()
    print ("Taking Photo")
    img_name = "Test.png"
    cv2.imwrite(img_name, frame)
    print("Image Saved!")
    CropImage()


###logic###
kb=[]
data = pd.read_csv('new.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

print("Checking Knowladge base...")
Check = "xxx is yyy"
object,subject=Check.split(' is ')
subject = subject.replace(" ", "")
expr=read_expr(subject + '(' + object + ')')
answer=ResolutionProver().prove(expr, kb)
if answer == True:
    print("Error found")
    sys.exit(0)

###Akinator###

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['small', 'average', 'big'], universe_of_discourse=[0,100])
FS.add_linguistic_variable("life", TLV)
FS.add_linguistic_variable("height", TLV)
FS.add_linguistic_variable("feathers", TLV)
FS.add_linguistic_variable("layeggs", TLV)
FS.add_linguistic_variable("swim", TLV)
FS.add_linguistic_variable("exoskeleton", TLV)
FS.add_linguistic_variable("scales", TLV)

O1 = TriangleFuzzySet(1,5,9,   term="reptile")
O2 = TriangleFuzzySet(10,15,19,  term="amphibian")
O3 = TriangleFuzzySet(20,25,29, term="mammal")
O4 = TriangleFuzzySet(30,35,39, term="fish")
O5 = TriangleFuzzySet(40,45,49, term="bird")
O6 = TriangleFuzzySet(50,55,60, term="invertebrate")
FS.add_linguistic_variable("animal", LinguisticVariable([O1, O2, O3, O4, O5, O6], universe_of_discourse=[1,60]))

FS.add_rules([
            "IF (layeggs IS small) THEN (animal IS mammal)",
            "IF (swim IS big)  AND  (scales IS big) THEN (animal IS fish)",
            "IF (feathers IS big) AND (layeggs IS big) THEN (animal IS bird)",
            "IF (exoskeleton IS big) THEN (animal IS invertebrate)",
            "IF (layeggs IS big) AND (swim IS big) AND (scales IS small) THEN (animal IS amphibian)",
            "IF (layeggs IS big) AND (swim IS small) AND (scales IS big) THEN (animal IS reptile)"
            
            ])    

def Akinator():
    pygame.init()
    pygame.display.set_caption("Akinator")
    win = pygame.display.set_mode((1400,800))
    Mediumfont = pygame.font.SysFont('Calibri', 30, True, False)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    GREY = (192,192,192)
    DARKGREY = (220,220,220)
    Akinator = pygame.image.load("Akinator.jpg")

    Questions = ["Does your animal lay eggs?","How long does your animal live on a scale of 1-100?","Can your animal swim?","Does your animal have feathers?","How big is your animal on a scale of 1-100?","Does your animal have an exoskeleton?","Does your animal have scales?","Does your animal have scales?"]
    Answers = [["Yes","No"],["Short","Average","Long"],["Yes","No"],["Yes","No"],["Small","Average","Large"],["Yes","No"],["Yes","No"],["Yes","No"],["Yes","No"]]
    User =[0,0,0,0,0,0,0,0]
    Number = 0
    run = True
    Answer = 0

    while run:
        win.fill(WHITE)
        win.blit(Akinator, (400, 100))
        pygame.time.delay(100)
        for event in pygame.event.get():
            MousePos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and Number != 4 and Number != 1:
                n = 0
                while n< len(Answers[Number]):
                    if (MousePos[0] in range (30,200)) and (MousePos[1] in range (420 + (n*120),470 + (n*120))):
                        if n == 0:
                            Answer = 100
                        if n == 1:
                            Answer = 0
                        User[Number] = Answer
                        Number = Number +1
                        Answer = 0
                    n = n+1
            elif event.type == pygame.MOUSEBUTTONDOWN and (Number == 4 or Number == 1):
                if (MousePos[0] in range (170,200)) and (MousePos[1] in range (420, 445)):
                    Answer = Answer + 1
                if (MousePos[0] in range (170,200)) and (MousePos[1] in range (450, 480)):
                    Answer = Answer - 1
                if (MousePos[0] in range (30,200)) and (MousePos[1] in range (540, 580)):
                    User[Number] = Answer
                    Number = Number + 1
            else:
                Yes = True
                
        if Answer > 100:
            Answer = 0
        if Answer < 0:
            Answer = 100

        pygame.draw.polygon(win, BLACK, [(0,400),(1400,400),(1400,800),(0,800)])
        pygame.draw.polygon(win, GREY, [(0,400),(1400,400),(1400,800),(0,800)],20)

        n = 0
        if Number != 1 and Number != 4:
            while n< len(Answers[Number]):
                pygame.draw.polygon(win, GREY, [(30,420 + (n*120)),(200,420 + (n*120)),(200,470 + (n*120)),(30,470 + (n*120))])
                text = Mediumfont.render(Answers[Number][n], True, BLACK)
                win.blit(text,(35,425 + (n*120)))
                n=n+1
        else:
            pygame.draw.polygon(win, GREY, [(30,420),(200,420 ),(200,480 ),(30,480)])
            text = Mediumfont.render(str(Answer), True, BLACK)
            win.blit(text,(35,430))
            pygame.draw.polygon(win, WHITE, [(170,420),(200,420 ),(200,445 ),(170,445)])
            pygame.draw.polygon(win, DARKGREY, [(170,420),(200,420 ),(200,445 ),(170,445)],5)
            pygame.draw.polygon(win, WHITE, [(170,450),(200,450 ),(200,480 ),(170,480)])
            pygame.draw.polygon(win, DARKGREY, [(170,450),(200,450 ),(200,478 ),(170,478)],5)
            text = Mediumfont.render("+", True, BLACK)
            win.blit(text,(175,420))
            text = Mediumfont.render("-", True, BLACK)
            win.blit(text,(180,450))
            pygame.draw.polygon(win, GREY, [(30,540),(200,540 ),(200,580 ),(30,580)])
            text = Mediumfont.render("Submit", True, BLACK)
            win.blit(text,(35,545))

        if len(Questions[Number]) > 100 and Number <6:
            Runs = 1
            Sentance = Questions[Number][0:100]
            while len(Questions[Number]) > (100*Runs):
                Sentance = Questions[Number][(100*(Runs-1)):100*Runs:]
                text = Mediumfont.render(Sentance, True, BLACK)
                win.blit(text,(50,30+(35*(Runs-1))))
                Runs = Runs +1
            Sentance = Questions[Number][(100*(Runs-1))::]
            text = Mediumfont.render(Sentance, True, BLACK)
            win.blit(text,(50,30+(35*(Runs-1))))
        else:
            text = Mediumfont.render(Questions[Number], True, BLACK)
            win.blit(text,(50,30))
        pygame.display.update()
        
        if Number >6:
            FS.set_variable("life", User[1]) 
            FS.set_variable("height", User[4])
            FS.set_variable("feathers", User[3]) 
            FS.set_variable("layeggs", User[0])
            FS.set_variable("swim", User[2]) 
            FS.set_variable("exoskeleton", User[5])
            FS.set_variable("scales", User[6])
            Gusse = FS.inference()
            Gusse = Gusse["animal"]

            if True:
                if Gusse >= 0 and Gusse <=10:
                    print("I gusse reptile")
                if Gusse > 10 and Gusse <=20:
                    print("I gusse amphibian")
                if Gusse > 20 and Gusse <=30:
                    print("I gusse mammal")
                if Gusse > 30 and Gusse <=40:
                    print("I gusse fish")
                if Gusse > 40 and Gusse <=50:
                    print("I gusse bird")
                if Gusse > 50 and Gusse <=60:
                    print("I gusse invertebrate")
            run = False
            pygame.quit()

###Audio mode##


speech_config = SpeechConfig(cog_key, cog_region)


Choice = input("Would you like to enter audio mode? ->")
Choice = Choice.lower()
if Choice == "yes":
    AudioMode = True
else:
    AudioMode = False







        
###Main###

kern = aiml.Kernel()
kern.setTextEncoding(None)

kern.bootstrap(learnFiles="CourseworkPart4.xml")

print("Welcome to this Twitter chat bot. Please feel free to ask for me to search for tweets or people for you, if your bored I can also play a game with you")

while True:
    try:
        if AudioMode:
            PlayText("Press enter to start recording:")
            Imdumb = input("Press enter to start recording:")
            

            print("You will have 5 seconds to speak")
            PlayText("You will have 5 seconds to speak")
            GetInputAudio(5)
            audio_config = AudioConfig(filename="Test.wav") # Use file instead of default (microphone)
            speech_recognizer = SpeechRecognizer(speech_config, audio_config)

            # Use a one-time, synchronous call to transcribe the speech
            speech = speech_recognizer.recognize_once()
            print(speech.text)
            Message = speech.text
        else:
            Message = input("Enter your Message: ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    
    ### Language detection and translation



    #####IMPORTANT ###IMPORTANT

    reviews = [Message + " " + Message]
    language_analysis = text_analytics_client.detect_language(documents=reviews)
    result = [doc for doc in language_analysis if not doc.is_error]
    for doc in result:
        LanguageName = doc.primary_language.name
        LanguageCode = doc.primary_language.iso6391_name
    if LanguageName != "English":
        
        Message = translate_text(cog_region, cog_key, Message, "en", LanguageCode)

    #####IMPORTANT ###IMPORTANT

    ###Sentiment analysis###
    Analysis = [Message]
    EmotionDetection = text_analytics_client.analyze_sentiment(Analysis, language="en")
    EmotionResult = [doc for doc in EmotionDetection if not doc.is_error]
    PositiveResponses = ["Wow whats got you in a good mood?","Feeling particularly good today?", "I want what you had for breakfast!"]
    NegativeResponses = ["Who shit in your cornfalkes?","Woke up on the wrong side of bed did you?", "How can we cheer you up?","Whats the matter?"]
    ResponseHappened = False
    for doc in EmotionResult:
        if doc.confidence_scores.positive >=0.8:
            Choice = random.randint(1,len(PositiveResponses)-1)
            if AudioMode:
                PlayText(PositiveResponses[Choice])
            print(PositiveResponses[Choice])
            ResponseHappened = True
        if doc.confidence_scores.negative >=0.8:
            Choice = random.randint(1,len(NegativeResponses)-1)
            if AudioMode:
                PlayText(NegativeResponses[Choice])
            print(NegativeResponses[Choice])
            ResponseHappened = True
    
    answer = kern.respond(Message)
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        

        if cmd == 0:
            
            if LanguageName != "English":
                    
                params[1] = translate_text(cog_region, cog_key, params[1], LanguageCode, "en")
            if AudioMode:
                PlayText(params[1])
            print(params[1])
            break
        elif cmd ==1:
            Search = params[1]
            Number = 0
            for tweet in tweepy.Cursor(api.search,q=Search,count=1,lang="en").items():
                if Number == 0:
                    Reply = WORD.findall(tweet.text)
                    Reply = " ".join(Reply)
                    if AudioMode:
                        PlayText("My search for "+ Search +" found the following most recent tweet "+ Reply)
                    print("My search for ", Search ," found the following most recent tweet: ", Reply)
                Number = Number +1


                
        ##Twitter search##
        elif cmd == 2:
            TwitterName = params[1]
            Return = FindBestUserInCSV(TwitterName)
            if Return == "":
                if AudioMode:
                    PlayText("I couldnt find that user")
                print("I couldnt find that user")
                Users = api.search_users(TwitterName,15)
                if AudioMode:
                    PlayText("Is it any of the following?")
                print("Is it any of the following?")
                Number = 0
                with open('TwitterCheck.csv','a', newline='') as csvfile:
                    NewNames = csv.writer(csvfile)
                    while Number <5:
                        try:
                            print(Users[Number].name)
                            NewNames.writerow([Users[Number].name,Users[Number].id])
                            Number = Number +1
                        except:
                            Number = Number +1
                    csvfile.close()
                Choice = input("->")
                Choice = Choice.lower()
                Return = FindBestUserInCSV(Choice)
            Return = api.user_timeline(Return,count=1)
            if Return == "":
                print("Unable to pull tweet, user could be on priavate or tweet contains unique text (1)")
            for status in Return:
                Reply = WORD.findall(status.text)
                Reply = " ".join(Reply)
                print("Most recent tweet by ", TwitterName ,": ", Reply)

        ##Game##
        elif cmd == 3:
            print("Do you want to play rock, paper, scissors(0) or a gusseing game(1)")
            GameType = input("")
            if GameType == "0":
                Choice = ""
                Options = ["Rock","Paper","Scissors"]

                print("We are about to play a game of rock, paper, scissors I can take your choice via audio(0) or image(1) input")

                GameType = input("")
                if GameType == "0":
                    print("Say your choice in:")
                    GetInputAudio(2)
                    ConvertData1Audio()
                    RandomNumber = random.randint(0,2)
                    print("I choose ", Options[RandomNumber])
                    print("You chose: ")
                    Number = TestAudio()
                    if Number == RandomNumber:
                        print("Draw")
                    elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                        print("You win!")
                    else:
                        print("I win!")
                elif GameType == "1":
                    print("Present choice to camera in:")
                    GetInputImage()
                    RandomNumber = random.randint(0,2)
                    print("I choose ", Options[RandomNumber])
                    print("You chose: ")
                    Number = TestImage()
                    if Number == RandomNumber:
                        print("Draw")
                    elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                        print("You win!")
                    else:
                        print("I win!")
            else:
                Akinator()

        ###Logic###
        #User knows
        elif cmd == 4:
            params[1] = params[1].lower()
            object,subject=params[1].split(' is ')
            if subject[0] == "a" and subject[1] == " ":
                subject = subject[1:]
            subject = subject.replace(" ", "")

            if subject[len(subject)-1] == "s":
                subject = subject[:-1]
            if subject[len(subject)-1] == "d" and subject[len(subject)-2] == "e":
                subject = subject[:-2]
            expr=read_expr(subject + '(' + object + ')')

            answer = ResolutionProver().prove(expr,kb)

            if answer:
                if AudioMode:
                    PlayText("I already knew this!")
                print("I already knew this!")
            else:
                if not(d.check(subject)):
                    subject = d.suggest(subject)[0]
                    expr=read_expr(subject + '(' + object + ')')
                    answer=ResolutionProver().prove(expr, kb)                
                if answer:
                    if AudioMode:
                        PlayText('As far as I am aware this is corret.')
                    print('As far as I am aware this is corret.')
                else:
                    expr=read_expr("-"+subject + '(' + object + ')')
                    answer=ResolutionProver().prove(expr, kb)
                    if answer:
                        if AudioMode:
                            PlayText("I know this is not true")
                            PlayText("Check this is correct, I have found a contridiction in my knowladgebase")
                        print("I know this is not true")
                        print("Check this is correct, I have found a contridiction in my knowladgebase")
                    else:
                        if AudioMode:
                            PlayText("Okay I will remember that!")
                        print('OK, I will remember that',object,'is', subject)
                        expr=read_expr(subject + '(' + object + ')')
                        kb.append(expr)
            
        #User Checking
        elif cmd == 5:
            params[1] = params[1].lower()
            object,subject=params[1].split(' is ')
            
            if subject[0] == "a" and subject[1] == " ":
                subject = subject[1:]
            subject = subject.replace(" ", "")
            if subject[len(subject)-1] == "s":
                subject = subject[:-1]
            if subject[len(subject)-1] == "d" and subject[len(subject)-2] == "e":
                subject = subject[:-2]
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb)
            if answer:
                if AudioMode:
                    PlayText('As far as I am aware this is corret.')
                print('As far as I am aware this is corret.')
            else:
                if not(d.check(subject)):
                    subject = d.suggest(subject)[0]
                    expr=read_expr(subject + '(' + object + ')')
                    answer=ResolutionProver().prove(expr, kb)                
                if answer:
                    if AudioMode:
                        PlayText('As far as I am aware this is corret.')
                    print('As far as I am aware this is corret.')
                else:

                    expr=read_expr("-"+subject + '(' + object + ')')
                    answer=ResolutionProver().prove(expr, kb)
                    if answer:
                        if AudioMode:
                            PlayText("I know this is not true")
                        
                        print("I know this is not true")
                    else:
                        if AudioMode:
                            PlayText("I do not know this")
                        print("I do not know this")

               
        ##Consine Similairty##                
        elif cmd == 99:
            Message = Message.lower()
            Return = FindBestResponseInCSV(Message)
            if Return == "TWEETSEARCH":
                Return = FindBestUserInCSV(Message)
                if Return == "":
                    print("I couldnt find that user")
                else:
                    Name = Return
                    Return = api.user_timeline(Return,count=1)
                    if Return == "":
                        print("Unable to pull tweet, user could be on priavate or tweet contains unique text (2)")
                    for status in Return:
                        Reply = WORD.findall(status.text)
                        Reply = " ".join(Reply)
                        print("Most recent tweet by ", Name ,": ", status.text.encode())
            elif Return == "GAME":
                print("Do you want to play rock, paper, scissors(0) or a gusseing game(1)")
                GameType = input("")
                if GameType == "0":
                    Choice = ""
                    Options = ["Rock","Paper","Scissors"]

                    print("We are about to play a game of rock, paper, scissors I can take your choice via audio(0) or image(1) input")

                    GameType = input("")
                    if GameType == "0":
                        print("Say your choice in:")
                        GetInputAudio(2)
                        ConvertData1Audio()
                        RandomNumber = random.randint(0,2)
                        print("I choose ", Options[RandomNumber])
                        print("You chose: ")
                        Number = TestAudio()
                        if Number == RandomNumber:
                            print("Draw")
                        elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                            print("You win!")
                        else:
                            print("I win!")
                    elif GameType == "1":
                        print("Present choice to camera in:")
                        GetInputImage()
                        RandomNumber = random.randint(0,2)
                        print("I choose ", Options[RandomNumber])
                        print("You chose: ")
                        Number = TestImage()
                        if Number == RandomNumber:
                            print("Draw")
                        elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                            print("You win!")
                        else:
                            print("I win!")
                else:
                    Akinator()
            #####IMPORTANT ###IMPORTANT        
            elif Return != "":
                if LanguageName != "English":
                    
                    Return = translate_text(cog_region, cog_key, Return, LanguageCode, "en")
                    if AudioMode:
                        if not ResponseHappened:
                            PlayText(Return)
                    if not ResponseHappened:
                        print(Return)
            else:
                output = "I dont understand, could you try rephrasing the question?"
                if LanguageName != "English":
                    output = translate_text(cog_region, cog_key, output, LanguageCode, "en")
                if AudioMode:
                    if not ResponseHappened:
                        PlayText(output)
                if not ResponseHappened:
                    print(output)

    #####IMPORTANT ###IMPORTANT
    else:
        if LanguageName!= "English":
                    
            answer = translate_text(cog_region, cog_key, answer, LanguageCode, "en")
        if AudioMode:
            if not ResponseHappened:
                PlayText(answer)
        if not ResponseHappened:
            print(answer)        

