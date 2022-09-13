import numpy as np
import pandas as pd
import torch
import random
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reportQL_to_json(reportQL):

    parsed_input = ''
    num_brace = 0

    i = 0
    while i < len(reportQL):
        if reportQL[i]=='{':
            num_brace += 1
            if num_brace==1:
                parsed_input += "{"
            elif num_brace==2:
                parsed_input += ":{"
            elif num_brace==3:
                parsed_input += ":["
            i += 1
        elif reportQL[i]=='}':
            num_brace -= 1
            if num_brace==2:
                j = i + 1
                while reportQL[j]==' ':
                    j += 1
                if reportQL[j]=='}':
                    parsed_input += "]"
                else:
                    parsed_input += "],"
            elif num_brace==1:
                j = i + 1
                while reportQL[j]==' ':
                    j += 1
                if reportQL[j]=='}':
                    parsed_input += "}"
                else:
                    parsed_input += "},"
            elif num_brace==0:
                parsed_input += "}"
            i += 1
        else:
            s = ""
            while reportQL[i]!='{' and reportQL[i]!='}':
                s += reportQL[i]
                i += 1
            if s.strip()=='':
                pass
            else:
                parsed_input += f'"{s.strip()}"'

    return json.loads(parsed_input)


def json_to_reportQL(json_in):
    parsed_input = ''
    s = json.dumps(json_in)

    i = 0
    while i < len(s):
        if s[i]==':' or s[i]==',':  # Put " " instead of ", " or ": "
            parsed_input += " "
            i += 2
        elif s[i]=="{" or s[i]=="[":  # Put "{ " instead of "{" or "["
            parsed_input += "{ "
            i += 1
        elif s[i]=="}" or s[i]=="]":  # Put " }" instead of "}" or "]"
            parsed_input += " }"
            i += 1
        else:
            i += 1  # skip first "
            temp = ""
            while s[i]!='"':
                temp += s[i]
                i += 1
            i += 1  # skip second "
            parsed_input += temp
    reportQL_to_json(parsed_input)
    return parsed_input


def add_special_tokens(tokenizer):
    special_tokens_dict = tokenizer.special_tokens_map
    special_tokens_dict['mask_token'] = '<mask>'
    special_tokens_dict['additional_special_tokens'] = ['<t>', '</t>', '<a>', '</a>']
    tokenizer.add_tokens(['{', '}', '<c>', '</c>', '<size>'])
    tokenizer.add_special_tokens(special_tokens_dict)
