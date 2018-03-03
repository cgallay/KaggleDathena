import PyPDF2
import docx

def apply(file_path):
    file_type = get_file_type(file_path)
    try:
        if file_type == 'EXCEL':
            return exctract_excel(file_path)
        if file_type == 'PDF':
            return exctract_pdf(file_path)
        if file_type == 'WORD':
            return exctract_word(file_path)
    except:
        print('Error while reading file ' + file_path)
        return ''

def get_file_type(file_path):
    if file_path[-3:] == 'pdf':
        return 'PDF'
    if file_path[-3:] == 'xls' or file_path[-4:] == 'xlsx':
        return 'EXCEL'
    if file_path[-3:] == 'doc' or file_path[-4:] == 'docx':
        return 'WORD'
    raise Exception('Unexpected file type')


def exctract_excel(path):
    raise NotImplementedError('Please implement the excel extraction function')
def exctract_pdf(path):
    pdf = PyPDF2.PdfFileReader(path)
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text+= ' ' + page.extractText()
    return text 
def exctract_word(path):
    doc = docx.Document(path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)