import pandas as pd
import csv

def createSubCSV(topId,downId):
    with open("sub.csv",'w',newline='') as csvfile:
        spamwritter = csv.writer(csvfile)
        spamwritter.writerow(['id','rank'])
        for i in range(len(df_files.sentiment)):
            if(i in topId):
                spamwritter.writerow([i,(10-topId.index(i))])
            elif(i in downId):
                spamwritter.writerow([i,(downId.index(i)-10)])
            else:
                spamwritter.writerow([i,0])

def getSubmission(df):
    dfCleaned = df[df['sentiment']>0.01]
    top = list(dfCleaned.sort_values('sentiment',ascending=False).head(10).id)
    down = list(dfCleaned.sort_values('sentiment',ascending=True).head(10).id)
    return top,down