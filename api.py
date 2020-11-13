
import jira
import re
import nltk
import string
from sklearn.metrics.pairwise import cosine_similarity
from os.path import isfile
from os.path import join
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
import getpass as pwd

password = pwd.getpass()

userid = 'Your PSID'

# print(len(allissues))
NewIssue = input("Enter the Issue ID")  # for now new jira is manually input by the user

authTuple = (userid, password)

jiraOpts = {
    'server': 'http://jira',
    'verify': False,
    'rest_api_version': 2,
    'check_update': False
}
authed_jira = jira.JIRA(basic_auth=authTuple, options=jiraOpts, validate=False, get_server_info=False)
# issue = authed_jira.issue('ABCD-333')
# issue
# issue = authed_jira.issue('ABCD-333', fields='summary,comment')
# issue
projects = authed_jira.projects()
projects  # list of all projects that fall under the http://jira.hk.hsbc url
block_size = 4000
block_num = 0
allissues = []
# projectName = str(input("Enter the project you want to access data from"))
while (True):
    start_idx = block_num * block_size
    issues = authed_jira.search_issues('project=ABCD', start_idx, block_size)
    if len(issues) == 0:
        # Retrieve issues until there are no more to come
        break
    block_num += 1
    for issue in issues:
        # log.info('%s: %s' % (issue.key, issue.fields.summary))

        allissues.append(issue)
# print(type(issue.key))

new_issues = authed_jira.issue(NewIssue)
allissues.append(new_issue)  # new jira will be appended to the rest of the issues
# allissues
import pandas as pd

issues = pd.DataFrame()

allissues  # lists all issues

# fetching the description and the summary and the jira number from the jira application
for issue in allissues:
    d = {
        'summary': issue.fields.summary,
        'description': issue.fields.description,
        'key': issue.key
    }
    issues = issues.append(d, ignore_index=True)

issues

issues['Data'] = issues['summary'] + issues['description'].astype(str)
issues = issues.drop('description', 1)
issues = issues.drop('summary', 1)

issues.head()  # final dataframe after the data extraction

# converting all the issues into separate text files with names as the jira
# numbers and keeping them in a separate folder named(JiraData).
# Currently testing on small set later multiprocessing will be used
i = 0
for index, row in issues.iterrows():
    if i > len(issues):
        break
    else:
        f = open('folder path' + row[0] + '.txt', 'w', encoding='utf-8')
        f.write(issues.iloc[i][1])
        f.close()
        i += 1

# Preprocessing the data, Applying NLP


BASE_FOLDER = "./JiraData/"


# preprocessing
def filePathList(folderPath):
    fileInfo = []
    listOfFileNames = [fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    listOfFilePaths = [join(folderPath, fileName) for fileName in listdir(folderPath) if
                       isfile(join(folderPath, fileName))]
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo


fileNames, filePaths = filePathList(BASE_FOLDER)


def dict_of_docContent(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        with open(filePath, "r", encoding="utf8", errors='ignore') as ifile:
            fileContent = ifile.read()
        rawContentDict[filePath] = fileContent
    return rawContentDict


rawContentDict = dict_of_docContent(filePaths)


def tokenizing(raw_contents):
    tokenized = nltk.tokenize.word_tokenize(raw_contents)
    return tokenized


def removing_stop_words(tokenized_content):
    stop_words_set = set(nltk.corpus.stopwords.words('english'))
    refined_content = [word for word in tokenized_content if word not in stop_words_set]
    return refined_content


def porter_stemming(tokenized_content):
    porterStemmer = nltk.stem.PorterStemmer()
    refined_content = [porterStemmer.stem(word) for word in tokenized_content]
    return refined_content


def removing_punctuation(tokenized_content):
    removed_punctuation = set(string.punctuation)
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'
    boldMarks = '*'


removed_punctuation.add(doubleSingleQuote)
removed_punctuation.add(doubleDash)
removed_punctuation.add(doubleTick)
removed_punctuation.add(boldMarks)

refined_content = [word for word in tokenized_content if word not in removed_punctuation]
return refined_content


def converting_to_lower(raw_contents):
    refined_content = [term.lower() for term in raw_contents]
    return refined_content


# testing
# content_test = rawContentDict[filePaths[0]]
# content_test
# content_test_tokenized = tokenizing(content_test)
# print(content_test_tokenized[:20])
def processData(rawContents):
    processed_content = tokenizing(rawContents)
    processed_content = removing_stop_words(processed_content)
    processed_content = porter_stemming(processed_content)
    processed_content = removing_punctuation(processed_content)
    processed_content = converting_to_lower(processed_content)
    return processed_content


def print_TFIDF_for_all(term, values, fileNames):
    values = values.transpose()
    numValues = len(values[0])
    print('                ', end="")
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')
        for j in range(numValues):
            print('{0:.12f}'.format(values[i][j]), end='   ')
        print()


def write_TFIDF_for_all(term, values, fileNames):
    filePath = "../results/tfid.txt"
    outFile = open(filePath, 'a')
    title = "TFIDF\n"
    outFile.write(title)
    values = values.transpose()
    numValues = len(values[0])
    outFile.write('               \t')
    for n in range(len(fileNames)):
        outFile.write('{0:18}'.format(fileNames[n]))
    outFile.write("\n")
    for i in range(len(term)):
        outFile.write('{0:15}'.format(term[i]))
        outFile.write('\t|  ')
        for j in range(numValues):
            outFile.write('{0:.12f}'.format(values[i][j]))
            outFile.write('   ')
        outFile.write("\n")

    outFile.close()


def calc_and_print_CosineSimilarity_for_all(tfs, fileNames):
    max_count = {}
    # print("\n\n\nCOSINE SIMILARITY\n")
    numFiles = len(fileNames)
    names = []
    # print('                   ', end="")
    for i in range(numFiles):
        if i == 0:
            for k in range(numFiles):
                print(fileNames[k], end='     ')
            print()

        print(fileNames[i], end='     ')
        for n in range(numFiles):
            if fileNames[n] == (NewIssue + '.txt'):
                matrixValue = cosine_similarity(tfs[i], tfs[n])
                # max_count[fileNames[n]] = matrixValue

            else:
                continue

            numValue = matrixValue[0][0]
            print(numValue)
            # print(n)
            max_count[fileNames[i]] = numValue
            names.append(fileNames[n])
            # print(" {0:.8f}".format(numValue), end='         ')

        print()
    # print("\n\n\n")
    # print(max_count)

    import operator
    sorted_d = sorted(max_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_d)
    for i in range(1, len(sorted_d)):
        print(sorted_d[i])
    similar_content = sorted_d[1:4]
    similar_names = []
    for name, score in similar_content:
        similar_names.append((name[:-4]))
    print(similar_names)
    similar_names = "\n".join(similar_names)
    # authed_jira.add_comment('CRDS-333', "Below are the the old JIRA References")
    authed_jira.add_comment('CRDS-333', similar_names)


def main(printResults=True):
    baseFolderPath = "./JiraData/"

    fileNames, filePathList = filePathList(baseFolderPath)

    rawContentDict = dict_of_docContent(filePathList)

    tfidf = TfidfVectorizer(tokenizer=processData, stop_words='english')
    tfs = tfidf.fit_transform(rawContentDict.values())
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()

    if printResults:
        # print_TFIDF_for_all(tfs_Term, tfs_Values, fileNames)
        calc_and_print_CosineSimilarity_for_all(tfs, fileNames)
    else:
        # write results to file
        write_TFIDF_for_all(tfs_Term, tfs_Values, fileNames)
        calc_and_write_CosineSimilarity_for_all(tfs, fileNames)


main()
