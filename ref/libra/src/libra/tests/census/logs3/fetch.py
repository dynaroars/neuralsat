import os

def order(logfile):
    D = dict()
    D['boxes'] = 0
    D['symbolic'] = 250
    D['deeppoly'] = 500
    A = dict()
    A['20F'] = 0
    A['20E'] = 10
    A['20D'] = 20
    A['20C'] = 30
    A['20B'] = 40
    A['20A'] = 50
    A['80F'] = 60
    A['80E'] = 70
    A['80D'] = 80
    A['80C'] = 90
    A['80B'] = 100
    A['80A'] = 110
    A['320F'] = 120
    A['320E'] = 130
    A['320D'] = 140
    A['320C'] = 150
    A['320B'] = 160
    A['320A'] = 170
    A['1280F'] = 180
    A['1280E'] = 190
    A['1280D'] = 200
    A['1280C'] = 210
    A['1280B'] = 220
    A['1280A'] = 230
    name = logfile.split('-')       # ['census', '20A', 'boxes', '0.25', '2.log']
    return A[name[1]] + D[name[2]]

# current directory
directory = os.fsencode('.').decode('utf-8')

header = ['File Name', 'Input', '|C|', '|F|p', '|F|f', 'Time']
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

        if found:

            patterns = found[1]
            zipped = compressed[2] if compressed else ''
            rest = found[4].split('[')
            feasible = rest[0]
            completed = rest[1].strip(']')

            _space = float(pre[4].strip('()').strip('%'))
            space = '{0:.2f}%'.format(_space)

            if result:

                seconds = float(total[-2].strip('s'))
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                time =  '|{}h |{}m |{}s'.format(int(hours), int(minutes), int(seconds)).replace('|0h ', '').replace('|0m ', '').replace('|', '')

                if compressed:
                    fetched = [logfile, space, completed, zipped, feasible, time]
                else:
                    fetched = [logfile, space, completed, patterns, feasible, time]
                print('\t '.join(fetched))

            else:

                if compressed:
                    fetched = [logfile, space, completed, zipped, feasible, '>13h']
                else:
                    fetched = [logfile, space, completed, patterns, feasible, '>13h']
                print('\t '.join(fetched))

        else:

            fetched = [logfile, '-', '-', '-', '-', '>13h']
            print('\t '.join(fetched))
