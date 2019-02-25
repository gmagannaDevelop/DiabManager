
import json

def list_to_string(ls: list) -> str:
    _str = ''
    for val in ls:
        _str += f'{val.strip()} '
    _str = _str.strip()
    return _str


def file_filter(in_file: str):
    """ Save only the entries which are either:
        'data' or 'event'
        to a file called 
        filename_filtered.extension
    """
    out_file = in_file.replace('.', '_filtered.')
    with open(in_file, 'r') as fsock1, open(out_file, 'w') as fsock2:
        for line in fsock1:
            _tmp = json.loads(line)
            if _tmp['type'] in 'data event':
                fsock2.write(f'{json.dumps(_tmp)}\n')


