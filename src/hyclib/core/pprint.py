import collections

def pformat(d, indent=0, spaces=4, depth=-1, formatters=None, verbose=True):
    output = ''
    for key, value in d.items():
        output += f"{' ' * spaces * indent}'{str(key)}':\n"
        if depth != 0 and isinstance(value, dict):
            output += f"{pformat(value, indent=indent+1, spaces=spaces, depth=depth - 1, formatters=formatters, verbose=verbose)}\n"
        else:
            if formatters and type(value) in formatters:
                value = formatters[type(value)](value)
            elif verbose:
                value = str(value)
            else:
                value = str(type(value))
            output += '\n'.join([f"{' ' * spaces * (indent+1)}{line}" for line in value.split('\n')]) + '\n'
    return output.rstrip('\n')

def pprint(d, **kwargs):
    print(pformat(d, **kwargs))

def iter_str(l):
    if isinstance(l, str):
        return repr(l)
        
    if not isinstance(l, (tuple, list, set, dict)):
        return str(l)

    if isinstance(l, dict):
        s = ', '.join(f'{repr(k)}: {v}' for k, v in l.items())
        return '{' + s + '}'
        
    s = ', '.join(iter_str(v) for v in l)

    if isinstance(l, set):
        return '{' + s + '}'

    if isinstance(l, tuple):
        return '(' + s + ')'
    
    return '[' + s + ']'
    
def pformat_english(*args):
    """
    Formats arguments in a way that follows English grammatical rules.
    """
    length = len(args)
    
    if length == 0:
        return ""
    
    if length == 1:
        return str(args[0])
    
    if length == 2:
        return f'{args[0]} and {args[1]}'
    
    return f"{', '.join(map(str, args[:-1]))}, and {args[-1]}"