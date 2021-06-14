# clean the doc to consider just the first line of it
import json
import os
import sys
import re
import  utils


if __name__=='__main__':
    apis = json.load(open('./API+JAVADOC+CODE.json', 'r'))
    apis_code = []
    apis_doc = []
    for api in apis:
        doc_first_line = api["api_doc"].split('.', 1)[0]
        doc_first_line = re.sub('</\w+>', '', doc_first_line)
        if doc_first_line != "" and api["api_code"] != "":
            doc_first_line_cleaned = utils.preprocess_sentence((doc_first_line))
            api_code_cleaned = utils.preprocess_sentence(api['api_code'])

            apis_code.append(api_code_cleaned)
            apis_doc.append(doc_first_line_cleaned)