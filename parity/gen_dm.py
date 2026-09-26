import json, os, random, string, sys
from doublemetaphone import doublemetaphone
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"..","dedupe-rs","tests","golden")
os.makedirs(OUT, exist_ok=True)
rnd=random.Random(99)
def rw(n): return ''.join(rnd.choice(string.ascii_letters) for _ in range(n))
words=set(["i","donald","goofy","cipciop","9301 S. State St.","thomas","thoms","smith","schmidt","xavier","cher","wright","gnu","knight","psalm","aeon","bajador","michael","chemistry","schlesinger","wachtler","wechsler","tichner","mcHugh","czerny","focaccia","McClellan","bellocchio","bacchus","accident","accede","succeed","bacci","bertucci","edgar","ghislane","ghiradelli","hugh","bough","broughton","laugh","McLaughlin","cough","gough","rough","tough","cagney","tagliaro","biaggi","sugar","resnais","artois","zhao","filipowicz","cabrillo","gallegos","hochmeier","rogier","jose","san jacinto","breaux","campbell","raspberry","island","isle","carlisle","carlysle","Yankelovich","Jankelowicz"])
for _ in range(120000):
    n=rnd.randint(1,30)
    s=list(rw(n))
    # sprinkle punctuation/digits/spaces
    for i in range(len(s)):
        if rnd.random()<0.12: s[i]=rnd.choice(" .,-'/&0123456789")
    words.add(''.join(s))
for _ in range(20000):
    words.add(rw(rnd.randint(1,40)))
words=sorted(words)
cases=[{"w":w,"expected":[doublemetaphone(w)[0], doublemetaphone(w)[1]]} for w in words]
json.dump(cases, open(os.path.join(OUT,"double_metaphone.json"),"w"))
print("double_metaphone:",len(cases))
