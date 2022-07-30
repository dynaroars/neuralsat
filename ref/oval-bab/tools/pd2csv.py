import glob
import pandas as pd
import argparse
import os


def pd2csv(path, result_path, keyword):
    files_prev = glob.glob(path+f'*{keyword}*')
    print(f'dealing with {len(files_prev)} pandas pickle files')
    for fname in files_prev:
        temp = pd.read_pickle(fname).dropna(how="all")
        # some modificatin to our tables
        for key in temp.keys():
            if key=='Idx':
                temp = temp.rename(columns={key: "Image"})

            elif key == 'prop' and ("mnist_colt" in keyword or "cifar_colt" in keyword):
                temp = temp.drop(columns=key)

            elif key == 'Eps':
                pass

            elif "SAT" in key:
                # as our method returns SAT/UNSAT on the counter-example search, flip True/False for ETH's datasets
                # (which, differently from ours, assume SAT is given if robust)
                if "mnist_colt" in keyword or "cifar_colt" in keyword:
                    temp.loc[temp[key] == 'False', key] = 'true'
                    temp.loc[temp[key] == 'True', key] = 'false'
                    temp.loc[temp[key] == 'false', key] = 'False'
                    temp.loc[temp[key] == 'true', key] = 'True'
                temp = temp.rename(columns={key: "SAT"})

            elif "BBran" in key:
                temp = temp.rename(columns={key:"Branches"})

            elif "BTime" in key:
                temp = temp.rename(columns={key:"Time(s)"})

            else:
                temp = temp.drop(columns=key)

        fname = fname.split('/')[-1]
        fname_csv = result_path+fname[:-4]+".csv"
        temp.to_csv(fname_csv, index=False)

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, help='modify files whose names containing the keyword')
    parser.add_argument('--path', type=str, default='./cifar_results/', help="path of files to be modified")
    args = parser.parse_args()
    path = args.path
    result_path = path+'csv/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    pd2csv(path, result_path, args.keyword)
