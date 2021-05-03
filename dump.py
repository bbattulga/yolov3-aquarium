
with open('./requirements.txt') as f:

    content = f.read().split('\n')
    for x in content:
        print('\"' + x + '\"')
