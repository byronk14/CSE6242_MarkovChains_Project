import PyPDF2
import csv

#import pdf
PDFfilename = "CAMEO.Manual.1.1b3.pdf" #filename of your PDF/directory where your PDF is stored
pfr = PyPDF2.PdfFileReader(open(PDFfilename, "rb")) #PdfFileReader object

#EventCode code dictionary
eventCodedict = {}

#iterate first 94 pages to get codes
codebookPageNum = 95
for i in range(12,codebookPageNum):
    contents = pfr.getPage(i).extractText().split('\n')
    for i in range(len(contents)-1):
        if contents[i].find("CAMEO") != -1:
            code = contents[i].replace('CAMEO','')
            description = contents[i+1].replace('Name','')
            eventCodedict[code] = description


with open('EventCodeLookup.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in eventCodedict.items()]


