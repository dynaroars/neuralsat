import os

def order(logfile):
    D = dict()
    D['boxes'] = 0
    D['symbolic'] = 36
    D['deeppoly'] = 45
    A = dict()
    A['LtEno'] = 0
    A['LtE0.20'] = 9
    A['Gtno'] = 18
    A['Gt0.20'] = 27
    name = logfile.split('-')       # ['german', 'Gtno', 'bias3', 'symbolic', '0', '2.log']
    return A[name[1]] + D[name[3]] + int(name[2][-1])

# current directory
directory = os.fsencode('.').decode('utf-8')

header = ['File Name', 'Input', '|C|', 'Bias', '|F|p', '|F|f', 'Time']
print('\t '.join(header))

logs = [logfile for logfile in os.listdir(directory) if logfile.endswith('.log')]

# for each logfile in the current directory
for logfile in sorted(logs, key=order):
    # for each line of the current logfile
    with open(logfile, 'r', encoding='latin-1') as filename:
        found = None
        compressed = None
        pre = None
        result = None
        total = None
        for line in filename:
            if line.startswith('Found: '):
                found = line.split()
            if line.startswith('Compressed'):
                compressed = line.split()
            if line.startswith('Pre-Analysis Result: '):
                pre = line.split()
            if line.startswith('Result: '):
                result = line.split()
            if line.startswith('Total'):
                total = line.split()

        patterns = found[1]
        zipped = compressed[2] if compressed else ''
        rest = found[4].split('[')
        feasible = rest[0]
        completed = rest[1].strip(']')

        _space = float(pre[4].strip('()').strip('%'))
        space = '{0:.2f}%'.format(_space)

        _biased = float(result[4].strip('()').strip('%'))
        biased = '{0:.2f}%'.format(_biased)

        seconds = float(total[-2].strip('s'))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        time =  '|{}h |{}m |{}s'.format(int(hours), int(minutes), int(seconds)).replace('|0h ', '').replace('|0m ', '').replace('|', '')

        if compressed:
            fetched = [logfile, space, completed, biased, zipped, feasible, time]
        else:
            fetched = [logfile, space, completed, biased, patterns, feasible, time]
        print('\t '.join(fetched))
